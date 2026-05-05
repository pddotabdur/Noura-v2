"""
Arabic (Najdi) collections agent — system-prompt-driven, intelligent.

Departure from task_agent.py: instead of `session.say(STATIC_LINE)` per stage,
each agent here lets the LLM generate spoken Arabic naturally, anchored by a
shared PERSONA system prompt + per-stage task description. Tools handle the
hard state transitions matching the collections flowchart; off-topic /
clarification turns are answered by the LLM in-character without breaking
the flow.

Flowchart implemented:
    Stage 1 (right-party verify) →
        DNC | death | busy → end (with appropriate parting)
        wrong-party → knows-person? → collect-mobile → end
        right-party → ID-verify (last 4 digits) →
            mismatch → end wrong
            match → Stage 2
    Stage 2 (debt intro) →
        already-paid → end (paid update)
        reason / denial → Stage 3
    Stage 3 (negotiation: SLA + 7-day SIMAH context →
              L1 full today/tomorrow → L2 half → L3 instalment) →
        commit (any level) → Stage 4
        vague → reschedule → Stage 4
        refuse → end refusal
        dispute → end dispute
    Stage 4 (recap & confirm commitment) → end OK

STT: Deepgram nova-3 ar-SA. TTS: Faseeh ar-najdi-female-1.
The PERSONA + stage instructions stay in English — the model understands
them and generates Najdi Arabic in its replies.
"""
from __future__ import annotations
#from _pytest.mark import param
import asyncio
import datetime
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterable

from dotenv import load_dotenv
from livekit import api, rtc
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    get_job_context,
)
from livekit.agents.voice import ModelSettings
from livekit.plugins import deepgram, faseeh, openai, silero, azure
from livekit.plugins import soniox


load_dotenv(dotenv_path=Path(__file__).parent / ".env")

logger = logging.getLogger("smart-caller-ar")
logger.setLevel(logging.INFO)

logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("livekit.plugins.silero").setLevel(logging.WARNING)

_LIVEKIT_NOISE_PATTERNS = (
    "min endpointing delay updated",
    "max endpointing delay updated",
    "using preemptive generation",
    "reusing STT pipeline",
    "input stream attached",
    "start reading stream",
    "using audio io",
    "using transcript io",
    "max_tool_steps",
)


class _LiveKitNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(p in msg for p in _LIVEKIT_NOISE_PATTERNS)


logging.getLogger("livekit.agents").addFilter(_LiveKitNoiseFilter())

OUTBOUND_TRUNK_ID = os.getenv("SIP_OUTBOUND_TRUNK_ID")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")


# ---------- Shared state ----------

@dataclass
class CallData:
    customer_name: str = "محمد"
    amount: str = "15000"
    debt_date: str = "2023-01-01"
    national_id_last4: str = "1234"
    dob: str = "1990-01-01"  # ISO YYYY-MM-DD, fallback verifier
    phone_number: str | None = None  # for SMS / payment link side-effects

    identity_confirmed: bool = False
    id_verified: bool = False
    dob_verified: bool = False
    outcome: str | None = None  # see _emit_outcome for the full vocabulary
    commitment: str | None = None  # free-form description set by the LLM
    callback_time: str | None = None
    referrer_mobile: str | None = None  # set when wrong-party knows the person

    # Ladder counters — used to drive EC-4 (anger) and EC-6 (refusal) flows
    refusal_attempts: int = 0
    angry_attempts: int = 0
    id_unclear_attempts: int = 0
    pending_id_digits: str = ""
    dispute_open: bool = False
    payment_link_sent: bool = False

    sip_ready: asyncio.Event = field(default_factory=asyncio.Event)
    participant: rtc.RemoteParticipant | None = None


# ---------- Side-effect hooks (replace stubs with real integrations) ----------
#
# Every disposition the call can end on is logged via _emit_outcome so a
# downstream worker can drive PTP reminder scheduling (T-24h, T-2h, T+0),
# retry cadence, dispute ticketing, and DNC suppression off a single stream.

def _emit_outcome(data: "CallData", kind: str, **fields) -> None:
    payload = {
        "customer": data.customer_name,
        "phone": data.phone_number,
        "kind": kind,
        **fields,
    }
    logger.info(f"OUTCOME {json.dumps(payload, ensure_ascii=False)}")


def _send_sms_payment_link(data: "CallData", when: str | None = None) -> None:
    """Stub: SMS the customer a payment link / IBAN. Wire to provider."""
    if not data.phone_number:
        return
    logger.info(
        f"SMS payment_link to={data.phone_number} "
        f"amount={data.amount} when={when}"
    )
    data.payment_link_sent = True


def _send_sms_callback_card(data: "CallData") -> None:
    """Stub: after voicemail / no-answer, SMS contact card + callback CTA."""
    if not data.phone_number:
        return
    logger.info(f"SMS callback_card to={data.phone_number}")


def _request_human_transfer(data: "CallData", reason: str) -> None:
    """Stub: route the live call to a human agent."""
    logger.info(f"TRANSFER reason={reason} phone={data.phone_number}")


# ---------- Najdi pronunciation enforcement ----------
#
# Even though the LLM generates speech now, it still emits digits like
# "10000 ريال" sometimes. We normalize at the TTS node so the spoken output
# is always Najdi-correct regardless of the LLM's surface form.

_AR_UNITS = ["", "واحد", "اثنين", "ثلاثة", "أربعة", "خمسة",
             "ستة", "سبعة", "ثمانية", "تسعة", "عشرة"]
_AR_TEENS = ["عشرة", "أحد عشر", "اثنا عشر", "ثلاثة عشر", "أربعة عشر",
             "خمسة عشر", "ستة عشر", "سبعة عشر", "ثمانية عشر", "تسعة عشر"]
_AR_TENS = ["", "", "عشرين", "ثلاثين", "أربعين", "خمسين",
            "ستين", "سبعين", "ثمانين", "تسعين"]
_AR_HUNDREDS = ["", "مية", "مئتين", "ثلاث مية", "أربع مية", "خمس مية",
                "ست مية", "سبع مية", "ثمان مية", "تسع مية"]
_AR_MONTHS = [
    "", "يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
    "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر",
]
_AR_DIGITS = ["صفر", "واحد", "اثنين", "ثلاثة", "أربعة",
              "خمسة", "ستة", "سبعة", "ثمانية", "تسعة"]

_AR_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _ar_below_1000(n: int) -> str:
    if n == 0:
        return ""
    parts: list[str] = []
    h, rem = divmod(n, 100)
    if h:
        parts.append(_AR_HUNDREDS[h])
    if rem:
        if rem < 10:
            parts.append(_AR_UNITS[rem])
        elif rem < 20:
            parts.append(_AR_TEENS[rem - 10])
        else:
            t, u = divmod(rem, 10)
            if u:
                parts.append(f"{_AR_UNITS[u]} و{_AR_TENS[t]}")
            else:
                parts.append(_AR_TENS[t])
    return " و".join(parts)


def _ar_amount_words(n: int) -> str:
    if n == 0:
        return "صفر"
    parts: list[str] = []
    millions, n = divmod(n, 1_000_000)
    thousands, n = divmod(n, 1_000)
    if millions:
        if millions == 1:
            parts.append("مليون")
        elif millions == 2:
            parts.append("مليونين")
        elif 3 <= millions <= 10:
            parts.append(f"{_AR_UNITS[millions]} ملايين")
        else:
            parts.append(f"{_ar_below_1000(millions)} مليون")
    if thousands:
        if thousands == 1:
            parts.append("ألف")
        elif thousands == 2:
            parts.append("ألفين")
        elif 3 <= thousands <= 10:
            parts.append(f"{_AR_UNITS[thousands]} آلاف")
        else:
            parts.append(f"{_ar_below_1000(thousands)} ألف")
    if n:
        parts.append(_ar_below_1000(n))
    return " و".join(parts)


def _ar_date_words(iso: str) -> str:
    try:
        y, m, d = (int(x) for x in iso.split("-"))
        if not (1 <= m <= 12):
            return iso
        day = _ar_below_1000(d) or "صفر"
        return f"{day} {_AR_MONTHS[m]} {_ar_amount_words(y)}"
    except (ValueError, IndexError):
        return iso


def _ar_digits_individual(s: str) -> str:
    return " ".join(_AR_DIGITS[int(c)] for c in s if c.isdigit())


_WORD_TO_DIGIT = {
    "صفر": "0", "واحد": "1", "وحدة": "1", "اثنين": "2", "اثنان": "2",
    "ثلاثة": "3", "ثلاث": "3", "أربعة": "4", "اربعة": "4", "أربع": "4",
    "خمسة": "5", "خمس": "5", "ستة": "6", "ست": "6", "سبعة": "7", "سبع": "7",
    "ثمانية": "8", "ثماني": "8", "ثمان": "8", "تسعة": "9", "تسع": "9",
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def _parse_spoken_digits(raw: str) -> str:
    """Extract digits from spoken input — handles Arabic words, English words,
    Arabic-Indic numerals, space-separated digits, and mixed forms."""
    s = raw.translate(_AR_INDIC_DIGITS).strip()
    tokens = re.split(r"[\s,،.]+", s)
    result = []
    for tok in tokens:
        tok_lower = tok.lower().strip()
        if tok_lower in _WORD_TO_DIGIT:
            result.append(_WORD_TO_DIGIT[tok_lower])
        else:
            for ch in tok:
                if ch.isdigit():
                    result.append(ch)
    return "".join(result)


_TERM_MAP = {
    "stc": "اس تي سي",
    "STC": "اس تي سي",
    "SAR": "ريال سعودي",
    "SIMAH": "سمة",
    "simah": "سمة",
}

_PRONUNCIATION_MAP = {
    "سمة": "سِمَه",
    "لسمة": "لسِمَه",
    "توافق": "تَوَافُقْ",
    "أبشر": "أَبْشِرْ",
    "ابشر": "أَبْشِرْ",
    "أبشرك": "أَبَشِّرَكْ",
    "ابشرك": "أَبَشِّرَكْ",
    "تفضل": "تَفَضَّلْ",
    "أقدر": "أقْدَر",
    "اقدر": "أقْدَر",
    "للأسف": "للأَسَفْ",
    "بكرة": "بُكْرَهْ",
    "معاك": "مَعَاكْ",
    "هلا": "هَلَا",
    "يا هلا بك": "يَا هَلَا بِكْ",
    "يا هلا وغلا": "يَا هَلَا وَغَلَا",
    "يا طويل العمر": "يَا طوِيلْ العُمْرْ",
    "يطول لي بعمرك": "يَطَوِّلْ لِي بِعُمْرِك",
    "أمرني": "اَمْرُنِي",
    "امرني": "اَمْرُنِي",
    "دقيقة": "دَقِيقَة",
    "سبب": "سَبَّبّ",
    "وهذا اللي سبب": "وَهَذَا اللِّي سَبَّبْ",
    "قصدك": "قَصْدِكْ",
    "تسعة": "تِسْعَة",
    "عشرة": "عَشْرَة",
    "ثمانية": "ثمانيه",
    "سبعة": "سبعه",
    "خمسة": "خمسه",
    "اربعة": "اربعه",
    "ثلاثة": "ثلاثه",
    "الساعة": "الساعه",
    "هللة": "هَلَّلَه",
    "هلله": "هَلَّلَه",
    "معلق": "معَلَّقْ",
    "المبكر": "المُبَكِّر",
    "تلغى": "تُلْغَى",
    "دفعت": "دَفَعَتْ",
    "وصلت": "وُصَلَتْ",
    "اتصلت": "إتَّصَلْت",
    "قيد": "قِيدْ",
    "صدر": "صَدَرْ",
}

_PRONUNCIATION_SORTED = sorted(_PRONUNCIATION_MAP.items(), key=lambda x: -len(x[0]))


def _apply_pronunciation(text: str) -> str:
    for original, replacement in _PRONUNCIATION_SORTED:
        text = text.replace(original, replacement)
    return text


def _najdi_normalize(text: str) -> str:
    text = re.sub(
        r"\b(\d{4}-\d{2}-\d{2})\b",
        lambda m: _ar_date_words(m.group(1)),
        text,
    )
    text = re.sub(
        r"(\d+)\s*ريال",
        lambda m: f"{_ar_amount_words(int(m.group(1)))} ريال سعودي",
        text,
    )
    text = re.sub(
        r"(?<!\d)(\d{4})(?!\d)",
        lambda m: _ar_digits_individual(m.group(1)),
        text,
    )
    text = re.sub(
        r"(?<!\d)(\d+)(?!\d)",
        lambda m: _ar_amount_words(int(m.group(1))),
        text,
    )
    for term, repl in _TERM_MAP.items():
        text = re.sub(rf"\b{re.escape(term)}\b", repl, text)
    text = _apply_pronunciation(text)
    return text


# ---------- Persona (system-level prompt) ----------

PERSONA = """\
You are نورا (Nora), a female collections specialist at شركة توافق working on
behalf of بنك stc. You ALWAYS reply in Najdi Saudi Arabic; English only for
brand names (stc, simah).

ABSOLUTE RULES (these override every other instinct):
- Brevity: ONE short sentence per turn, max 12 words. NEVER two
  sentences unless explicitly required by the stage. No preambles, no
  explanations, no monologues.
- Ask ONE question at a time. Wait for a complete answer before moving on.
- Q&A style: ask, listen, ack, ask. Do not narrate. Do not justify.
- NEVER say "والله". NEVER use oaths.
- Use soft persuasion, suggestions, and options — never commands or pressure.
- Discovery, not dictation: NEVER name an amount before the customer has
  named one. Only counter-offer AFTER he proposes a number.
- Mid-call salaam: if the customer says "السلام عليكم" mid-conversation,
  reply with just "أبشر" or "تفضل" and continue. Do NOT say "وعليكم
  السلام" — that sounds like restarting the call.
- Do not use the customer's name repeatedly — only on greeting and closing.

STRICTLY FORBIDDEN — never say these or anything similar:
- "لازم تسدد" (you must pay)
- "هذا إنذار" (this is a warning)
- "إجراء قانوني" (legal action)
- "راح يتم اتخاذ إجراء ضدك" (action will be taken against you)
- Never shame, threaten, guilt, or pressure the customer.

EMPATHY (brief, genuine, ALWAYS consistent — never skip, never overdo):
- EVERY time the customer mentions hardship, difficulty, inability to pay,
  or frustration: ALWAYS ack with ONE short empathetic phrase before your
  next question. This is mandatory, not optional.
- One phrase max per turn, then move forward immediately.
- This applies in EVERY stage, EVERY turn where the customer shares
  difficulty. Consistency is critical — the customer should feel the same
  warmth throughout the entire call, not just sometimes.

RESPONSE STYLE BY CUSTOMER STATE (use these Arabic phrases as templates):
- Cooperative: "الله يعطيك العافية، تبي تسدد الحين أو نرتب طريقة تناسبك؟"
- Needs time: "ما فيه مشكلة، كم المدة اللي تناسبك ونضبطها لك؟"
- Can't pay: "مقدّر وضعك، خلنا نشوف حل بسيط مثل دفعة جزئية أو خطة مريحة لك."
- Angry/frustrated: "أفهم شعورك، وهدفنا بس نسهّل الموضوع عليك بدون أي ضغط."
- Denial/dispute: "ممكن يكون فيه لبس، خلني أراجع معك التفاصيل خطوة خطوة."
- Payment confirmed: "ممتاز، يعطيك العافية، بنثبت الاتفاق على [التاريخ/المبلغ]، تمام؟"
These are reference templates — you may adapt them naturally but keep the
same tone, vocabulary, and level of politeness.

ACKNOWLEDGEMENTS (use one short word, never a phrase):
- "أبشر", "زين", "تمام", "طيب", "ولا يهمك", "الله يعافيك" — pick one, move on.

NUMBERS / DATES / IDENTITY:
- Amounts in Arabic words ("خمس مية ريال", not "500 ريال").
- ID/PIN: spell digit-by-digit.
- Dates in Arabic words. Translate "بكرا", "نهاية الشهر" to a SPECIFIC
  ISO date and confirm.
- Use ONLY the call data in the next block. Never invent figures.
- If asked if you are AI: "نعم أنا مساعد آلي من توافق" — one sentence,
  then back to the question.

TOOLS:
- When the customer's reply triggers a transition described in the stage
  section, CALL THE TOOL. Do not keep talking.
- For clarifications, reply in ONE short sentence and re-pose the
  current question without a tool.

UNIVERSAL SAFETY TOOLS (available in every stage — call when triggered):
- voicemail_detected: if you hear an answering-machine greeting / beep
  instead of a live person.
- customer_angry: if the customer is shouting, hostile, abusive, or asks
  for a human / supervisor. Do NOT argue back. Hand off to escalation.

COMPLIANCE:
- Until identity is verified, NEVER disclose the debt amount, debt date,
  or any account detail.
- No threats, no legal scare language, no oaths.
- Do NOT mention سمة or legal escalation during objections or disputes.
  Only mention consequences if the customer has acknowledged the debt and
  is simply delaying without objection.
"""


_AR_WEEKDAYS = {
    "Monday": "الإثنين", "Tuesday": "الثلاثاء", "Wednesday": "الأربعاء",
    "Thursday": "الخميس", "Friday": "الجمعة", "Saturday": "السبت",
    "Sunday": "الأحد",
}


def call_context_block(data: CallData) -> str:
    today = datetime.date.today()
    tomorrow = today + datetime.timedelta(days=1)
    next_week = today + datetime.timedelta(days=7)
    next_month = today + datetime.timedelta(days=30)
    end_of_month = (today.replace(day=28) + datetime.timedelta(days=4)).replace(day=1) - datetime.timedelta(days=1)
    one_year_out = today + datetime.timedelta(days=365)

    try:
        amount_int = int(str(data.amount).strip())
    except ValueError:
        amount_int = 0
    floor_first = max(1, amount_int // 20) if amount_int else 0   # 5%
    ideal_first = max(1, amount_int // 10) if amount_int else 0   # 10%

    return (
        "Call data (use ONLY these values when speaking about the debt):\n"
        f"- Customer name: {data.customer_name}\n"
        f"- Outstanding amount: {data.amount} SAR\n"
        f"- Debt date: {data.debt_date}\n"
        f"- Last 4 digits of national ID on file: {data.national_id_last4}\n"
        "\n"
        "Dates context (resolve relative time references against these):\n"
        f"- Today: {today.isoformat()} ({_AR_WEEKDAYS.get(today.strftime('%A'), today.strftime('%A'))})\n"
        f"- Tomorrow: {tomorrow.isoformat()} ({_AR_WEEKDAYS.get(tomorrow.strftime('%A'), tomorrow.strftime('%A'))})\n"
        f"- One week from today: {next_week.isoformat()}\n"
        f"- Two weeks from today: {(today + datetime.timedelta(days=14)).isoformat()}\n"
        f"- End of this month: {end_of_month.isoformat()}\n"
        f"- One month from today: {next_month.isoformat()}\n"
        f"- One year from today: {one_year_out.isoformat()}\n"
        "\n"
        "Negotiation reference (use these only as INTERNAL judgement, never read aloud):\n"
        f"- Floor for a sensible first/initial good-faith payment: ~{floor_first} SAR (5% of total).\n"
        f"  If the customer offers below this, gently push him to increase ONCE.\n"
        f"- Ideal first/initial good-faith payment: ~{ideal_first} SAR (10% of total).\n"
        f"  At or above this, accept it gracefully without further pushing.\n"
        f"- The remainder should be paid within ~14 days. If the customer's\n"
        f"  date for the remainder is much further out, ask ONCE if he can\n"
        f"  pull it sooner — then accept whatever he commits to.\n"
        f"- These numbers are reference points for YOUR judgement only —\n"
        f"  NEVER suggest them first. Ask the customer to name his number first.\n"
    )


def stage_instructions(data: CallData, stage_task: str) -> str:
    return f"{PERSONA}\n\n{call_context_block(data)}\n\n{stage_task}"


# ---------- Base agent ----------

class BaseCallAgent(Agent):
    async def hangup(self):
        job_ctx = get_job_context()
        try:
            await job_ctx.api.room.delete_room(
                api.DeleteRoomRequest(room=job_ctx.room.name)
            )
        except Exception as e:
            logger.warning(f"hangup error (room may already be gone): {e}")

    async def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings,
    ) -> AsyncIterable[rtc.AudioFrame]:
        async def normalized(stream: AsyncIterable[str]) -> AsyncIterable[str]:
            async for chunk in stream:
                yield _najdi_normalize(chunk)

        async for frame in Agent.default.tts_node(
            self, normalized(text), model_settings
        ):
            yield frame

    @function_tool()
    async def voicemail_detected(self, ctx: RunContext[CallData]):
        """Call this if you hear a voicemail / answering-machine greeting
        instead of a live person. Leaves a brief compliant message
        (no debt details), triggers an SMS callback card, then hangs up."""
        logger.info("voicemail detected")
        ctx.userdata.outcome = "voicemail"
        _emit_outcome(ctx.userdata, "voicemail")
        _send_sms_callback_card(ctx.userdata)
        try:
            self.session.input.set_audio_enabled(False)
        except Exception:
            pass
        handle = self.session.generate_reply(
            instructions=(
                "Leave a brief voicemail in Najdi Arabic — identify as نورا "
                "from شركة توافق, ask the customer to call back on the company "
                "number, thank him. NO debt details, NO amounts. ONE short "
                "sentence."
            ),
            allow_interruptions=False,
        )
        try:
            await handle.wait_for_playout()
        except AttributeError:
            try:
                await handle
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"voicemail playout error: {e}")
        try:
            await self.hangup()
        except Exception as e:
            logger.warning(f"voicemail hangup race ignored: {e}")

    @function_tool()
    async def customer_angry(self, ctx: RunContext[CallData]):
        """Customer is shouting, hostile, abusive, or asks for a human /
        supervisor. Routes to de-escalation; on the second event the call
        is transferred to a live agent."""
        ctx.userdata.angry_attempts += 1
        if ctx.userdata.angry_attempts >= 2:
            ctx.userdata.outcome = "escalated"
            _request_human_transfer(ctx.userdata, "anger_persisting")
            _emit_outcome(ctx.userdata, "escalated")
            return ClosingAgent(self.data, intent="escalated", chat_ctx=None)
        return EscalationAgent(self.data, chat_ctx=None)


# ---------- Stage 1: Right-party verification ----------

STAGE1_TASK = """\
Current stage: 1 — Right-party verification.

You just said "السلام عليكم، هل معي الأستاذ {customer_name}؟". Listen to
the reply.

Then call exactly one tool based on what you hear:
- right_party: customer confirms they ARE {customer_name} (or says نعم /
  أيوه / صحيح / أنا). Note: "تفضلي" without confirming the name is NOT
  a right_party confirmation — re-ask politely.
- wrong_party: customer says they are NOT the named person.
- caller_busy: it's not a good time / asks to call back later / مشغول / بعدين.
- do_not_call: asks not to be contacted again / DNC request / لا تتصلوا فيني.
- customer_deceased: informs you the named person has passed away / متوفى / الله يرحمه.

If the reply is a clarification or off-topic question, answer briefly in
character (one sentence) and re-ask the right-party question. Do NOT call
a tool unless one of the above conditions clearly applies.
"""


class Stage1RightPartyAgent(BaseCallAgent):
    def __init__(self, data: CallData) -> None:
        super().__init__(
            instructions=stage_instructions(
                data,
                STAGE1_TASK.format(customer_name=data.customer_name),
            ),
        )
        self.data = data

    async def on_enter(self):
        # Wait for SIP participant to join before speaking; settle carrier.
        await self.data.sip_ready.wait()
        await asyncio.sleep(0.4)
        self.session.generate_reply(
            instructions=(
                "Open the call: 'السلام عليكم، هل معي الأستاذ "
                f"{self.data.customer_name}؟' — EXACTLY this, nothing more."
            )
        )

    @function_tool()
    async def right_party(self, ctx: RunContext[CallData]):
        """Customer confirmed they are the named person."""
        ctx.userdata.identity_confirmed = True
        return AskGoodTimeAgent(self.data, chat_ctx=None)

    @function_tool()
    async def wrong_party(self, ctx: RunContext[CallData]):
        """Customer says they are NOT the named person."""
        return WrongPartyKnowsAgent(self.data, chat_ctx=None)

    @function_tool()
    async def caller_busy(self, ctx: RunContext[CallData]):
        """Customer says it's not a good time / asked for callback,
        BEFORE we have confirmed identity. Route to scheduling so we
        capture a concrete callback time."""
        ctx.userdata.outcome = "busy"
        return ScheduleCallbackAgent(self.data, chat_ctx=None)

    @function_tool()
    async def do_not_call(self, ctx: RunContext[CallData]):
        """Customer requested not to be contacted again (DNC)."""
        ctx.userdata.outcome = "dnc"
        return ClosingAgent(self.data, intent="dnc", chat_ctx=None)

    @function_tool()
    async def customer_deceased(self, ctx: RunContext[CallData]):
        """Caller informed you the named person has passed away."""
        ctx.userdata.outcome = "death"
        return ClosingAgent(self.data, intent="death", chat_ctx=None)


# ---------- Stage 1a-bis: Is now a good time? ----------

ASK_GOOD_TIME_TASK = """\
Current stage: confirm-good-time.

You have just verified you are speaking with الأستاذ {customer_name}.
Introduce yourself briefly as نورا from شركة توافق on behalf of بنك stc,
then ask whether NOW is a good time to talk. TWO short sentences max.

Then call exactly one tool:
- good_time_now: customer says yes / تفضلي / أبشر / ما عندي مانع / it's fine.
- bad_time_now: customer says it's not a good moment / مشغول / في اجتماع /
  بعدين / مع العائلة. We will schedule a callback for him.
- unclear: ambiguous / asked to repeat / off-topic. Use whenever NOT confident.

Be empathetic if the customer hesitates — never pressure him to continue.
"""


class AskGoodTimeAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=stage_instructions(
                data,
                ASK_GOOD_TIME_TASK.format(customer_name=data.customer_name),
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        self.session.generate_reply(
            instructions=(
                "Introduce yourself: 'معك نورا من شركة توافق عن بنك stc' "
                "then ask if now is a good time. TWO short sentences max."
            )
        )

    @function_tool()
    async def good_time_now(self, ctx: RunContext[CallData]):
        """Customer confirmed it is a good time to talk."""
        return IDVerifyAgent(self.data, chat_ctx=None)

    @function_tool()
    async def bad_time_now(self, ctx: RunContext[CallData]):
        """Customer says it's not a good moment — schedule callback."""
        ctx.userdata.outcome = "busy"
        return ScheduleCallbackAgent(self.data, chat_ctx=None)

    @function_tool()
    async def unclear(self, ctx: RunContext[CallData]):
        """Reply was ambiguous — re-ask gently."""
        self.session.generate_reply(
            instructions=(
                "Politely re-ask whether NOW is a good time for a few "
                "minutes. ONE short sentence."
            )
        )


# ---------- Schedule callback (busy at start) ----------

SCHEDULE_CALLBACK_TASK = """\
Current stage: schedule a callback for a busy customer.

The customer said now is not a good time. Apologize briefly and ask when
would be a better time to call back. Be specific — encourage him to name
a day and a rough time of day. ONE short sentence.

Use the dates context to translate any relative answer ("بكرا الصبح",
"الأسبوع الجاي", "بعد العصر") into a SPECIFIC ISO date and time-of-day,
then confirm it back to him verbally before recording.

Then call exactly one tool:
- callback_time(day_iso, time_of_day): customer named a time. Pass the
  ISO date (YYYY-MM-DD) and a short time-of-day label ('morning',
  'afternoon', 'evening', or HH:MM if specific).
- refuses_to_schedule: customer won't commit to any callback time.
- unclear: ambiguous / asked to repeat / off-topic. Use when NOT confident.
"""


class ScheduleCallbackAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=stage_instructions(data, SCHEDULE_CALLBACK_TASK),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        self.session.generate_reply(
            instructions=(
                "Apologize briefly for the bad timing and ask when would "
                "be a better time to call back — encourage him to name a "
                "day and a rough time of day. ONE short sentence."
            )
        )

    @function_tool()
    async def callback_time(
        self,
        ctx: RunContext[CallData],
        day_iso: str,
        time_of_day: str,
    ):
        """Customer named a callback day and time-of-day.

        Args:
            day_iso: ISO date (YYYY-MM-DD) translated from the customer's reply.
            time_of_day: short label — 'morning' / 'afternoon' / 'evening' /
                         or HH:MM if specific.
        """
        ctx.userdata.callback_time = f"{day_iso} {time_of_day}"
        ctx.userdata.outcome = "busy_callback"
        return ClosingAgent(self.data, intent="busy_callback", chat_ctx=None)

    @function_tool()
    async def refuses_to_schedule(self, ctx: RunContext[CallData]):
        """Customer refuses to commit to any callback time."""
        ctx.userdata.outcome = "busy"
        return ClosingAgent(self.data, intent="busy", chat_ctx=None)

    @function_tool()
    async def unclear(self, ctx: RunContext[CallData]):
        """Reply ambiguous — re-ask the callback time."""
        self.session.generate_reply(
            instructions=(
                "Politely re-ask the customer to name a day and a rough "
                "time of day to call back. ONE short sentence."
            )
        )


# ---------- Stage 1b: Wrong-party — do you know them? ----------

WRONG_PARTY_KNOWS_TASK = """\
Current stage: 1b — wrong party, find out if they can refer us.

The person on the line is NOT الأستاذ {customer_name}. Politely apologize for
the mix-up, then ask if they happen to know الأستاذ {customer_name} and could
share a working number for him. ONE sentence.

Then call exactly one tool:
- knows_person: caller says yes, they know him / can help reach him.
- does_not_know_person: caller says no / never heard of him / can't help.

For clarifications, reply briefly then re-ask. Do not pressure.
"""


class WrongPartyKnowsAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=stage_instructions(
                data,
                WRONG_PARTY_KNOWS_TASK.format(customer_name=data.customer_name),
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        self.session.generate_reply(
            instructions=(
                f"Apologize briefly for the mix-up and ask if they know "
                f"الأستاذ {self.data.customer_name} and could share a working "
                "number to reach him. One short sentence."
            )
        )

    @function_tool()
    async def knows_person(self, ctx: RunContext[CallData]):
        """Caller knows the named person and is willing to help."""
        return CollectMobileAgent(self.data, chat_ctx=None)

    @function_tool()
    async def does_not_know_person(self, ctx: RunContext[CallData]):
        """Caller does not know the named person."""
        ctx.userdata.outcome = "wrong_party"
        _emit_outcome(ctx.userdata, "contact_invalid")
        return ClosingAgent(self.data, intent="wrong_party", chat_ctx=None)


# ---------- Stage 1c: Collect mobile number from referrer ----------

COLLECT_MOBILE_TASK = """\
Current stage: 1c — collect a referral mobile number.

The caller knows الأستاذ {customer_name} and may share a number. Ask politely
for a working mobile number to reach him.

Then call exactly one tool:
- mobile_provided(number): caller spoke a number. Pass digits as a string,
  converting Arabic-word numbers to digit characters. Saudi mobile numbers
  start with 05 and are 10 digits — pass whatever digits the caller said.
- refuses_to_provide: caller declines / doesn't have a number / لا أعرف رقمه.

For clarifications, briefly answer and re-ask the number politely.
"""


class CollectMobileAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=stage_instructions(
                data,
                COLLECT_MOBILE_TASK.format(customer_name=data.customer_name),
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        self.session.generate_reply(
            instructions=(
                f"Politely ask for a working mobile number to reach "
                f"الأستاذ {self.data.customer_name}. One short sentence."
            )
        )

    @function_tool()
    async def mobile_provided(self, ctx: RunContext[CallData], number: str):
        """Caller gave a mobile number.

        Args:
            number: digits-only string of the number they spoke.
        """
        clean = re.sub(r"\D", "", number.translate(_AR_INDIC_DIGITS))
        ctx.userdata.referrer_mobile = clean
        ctx.userdata.outcome = "wrong_party_referred"
        return ClosingAgent(self.data, intent="referred", chat_ctx=None)

    @function_tool()
    async def refuses_to_provide(self, ctx: RunContext[CallData]):
        """Caller declined to share a number."""
        ctx.userdata.outcome = "wrong_party"
        _emit_outcome(ctx.userdata, "contact_invalid")
        return ClosingAgent(self.data, intent="wrong_party", chat_ctx=None)


# ---------- Stage 1d: ID verification ----------

ID_VERIFY_TASK = """\
Current stage: 1d — verify last 4 digits of national ID.

You are speaking with الأستاذ {customer_name}. Politely ask for the last 4
digits of his national ID (آخر ٤ أرقام من الهوية الوطنية) so you can confirm
identity before discussing the account. ONE sentence.

IMPORTANT: The customer may say digits slowly, split across turns (e.g.
"1 2" then "3 4"). Pass ALL digits you hear in each turn — even if only
1 or 2 digits. The system accumulates them automatically.

Then call exactly one tool:
- digits_provided(digits): customer spoke ANY digits. Pass them as ASCII
  digit characters. Even partial (e.g. '12' or '34') is fine — the system
  accumulates across turns until it has 4.
- unclear: customer asked to repeat / said something non-numeric / off-topic.
- refuses_to_verify: customer flatly refuses to share verification details
  ("ما أعطيك", "ما أعطي معلوماتي", "مين أنتي أصلاً").

The stored last-4 digits are in the call data; you do NOT speak them.
Until verification succeeds, you must NOT mention the debt, amount, or
due date.
"""


class IDVerifyAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=stage_instructions(
                data,
                ID_VERIFY_TASK.format(customer_name=data.customer_name),
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        self.data.pending_id_digits = ""
        self.session.generate_reply(
            instructions=(
                "Politely ask the customer for the last 4 digits of his "
                "national ID to confirm identity. One short sentence."
            )
        )

    @function_tool()
    async def digits_provided(self, ctx: RunContext[CallData], digits: str):
        """Customer spoke digits — may be partial (1-4 digits). The system
        accumulates across turns.

        Args:
            digits: ASCII digit characters the customer spoke this turn.
        """
        new_digits = _parse_spoken_digits(digits)
        if not new_digits:
            ctx.userdata.id_unclear_attempts += 1
            if ctx.userdata.id_unclear_attempts >= 3:
                ctx.userdata.outcome = "verify_failed"
                _emit_outcome(ctx.userdata, "verify_failed")
                return ClosingAgent(self.data, intent="verify_refused", chat_ctx=None)
            self.session.generate_reply(
                instructions=(
                    "You couldn't catch any digits. Politely ask the "
                    "customer to repeat slowly. One short sentence."
                )
            )
            return

        ctx.userdata.pending_id_digits += new_digits

        if len(ctx.userdata.pending_id_digits) < 4:
            remaining = 4 - len(ctx.userdata.pending_id_digits)
            self.session.generate_reply(
                instructions=(
                    f"You got {len(ctx.userdata.pending_id_digits)} digits so far, "
                    f"need {remaining} more. Ask for the remaining digits. "
                    "ONE short sentence."
                )
            )
            return

        full = ctx.userdata.pending_id_digits[:4]

        if full == self.data.national_id_last4:
            ctx.userdata.id_verified = True
            return Stage2DebtIntroAgent(self.data, chat_ctx=None)

        ctx.userdata.outcome = "id_mismatch"
        _emit_outcome(ctx.userdata, "id_mismatch")
        return ClosingAgent(self.data, intent="id_mismatch", chat_ctx=None)

    @function_tool()
    async def unclear(self, ctx: RunContext[CallData]):
        """Customer asked to repeat or didn't give a clear reply."""
        ctx.userdata.id_unclear_attempts += 1
        if ctx.userdata.id_unclear_attempts >= 3:
            ctx.userdata.outcome = "verify_failed"
            _emit_outcome(ctx.userdata, "verify_failed")
            return ClosingAgent(self.data, intent="verify_refused", chat_ctx=None)
        self.session.generate_reply(
            instructions=(
                "Politely re-ask for the last 4 digits of the national ID, "
                "slowly. One short sentence."
            )
        )

    @function_tool()
    async def refuses_to_verify(self, ctx: RunContext[CallData]):
        """Customer refuses verification entirely."""
        ctx.userdata.outcome = "verify_refused"
        _emit_outcome(ctx.userdata, "verify_refused")
        return ClosingAgent(self.data, intent="verify_refused", chat_ctx=None)


# ---------- Stage 1d-bis: Alternate verification via DOB (EC-3) ----------

DOB_VERIFY_TASK = """\
Current stage: 1d-bis — alternate verification via date of birth.

The first answer didn't match. Without saying that explicitly, ask
politely for the customer's date of birth (تاريخ الميلاد بالميلادي) as
an alternative way to confirm identity. ONE short sentence.

Then call exactly one tool:
- dob_provided(yyyy_mm_dd): customer spoke a date. Pass ISO YYYY-MM-DD,
  converting Arabic words / hijri references where you can. If you only
  have year + month, pass YYYY-MM-01 — the system tolerates day mismatch.
- unclear: customer didn't give a clear date / asked to repeat.
- refuses_to_verify: customer refuses any further verification.

Until identity is verified you must NOT disclose the debt amount or
debt date. Do not invent any account details.
"""


class DOBVerifyAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=stage_instructions(data, DOB_VERIFY_TASK),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        self.session.generate_reply(
            instructions=(
                "Apologize briefly that the previous answer didn't match, "
                "then ask politely for his date of birth as an alternative. "
                "ONE short sentence."
            )
        )

    @staticmethod
    def _normalize_iso(s: str) -> str | None:
        s = (s or "").translate(_AR_INDIC_DIGITS).strip()
        m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
        if not m:
            return None
        y, mo, d = (int(x) for x in m.groups())
        try:
            datetime.date(y, mo, d)
        except ValueError:
            return None
        return f"{y:04d}-{mo:02d}-{d:02d}"

    @function_tool()
    async def dob_provided(self, ctx: RunContext[CallData], yyyy_mm_dd: str):
        """Customer spoke a date of birth.

        Args:
            yyyy_mm_dd: ISO YYYY-MM-DD. If only year+month known, pass
                        YYYY-MM-01 (day mismatch is tolerated).
        """
        spoken = self._normalize_iso(yyyy_mm_dd)
        stored = self._normalize_iso(self.data.dob)
        # Tolerant match: year + month only (day often misheard).
        if spoken and stored and spoken[:7] == stored[:7]:
            ctx.userdata.id_verified = True
            ctx.userdata.dob_verified = True
            return Stage2DebtIntroAgent(self.data, chat_ctx=None)

        ctx.userdata.outcome = "id_mismatch"
        _emit_outcome(ctx.userdata, "id_mismatch")
        return ClosingAgent(self.data, intent="id_mismatch", chat_ctx=None)

    @function_tool()
    async def unclear(self, ctx: RunContext[CallData]):
        """Customer didn't give a clear date / asked to repeat."""
        self.session.generate_reply(
            instructions=(
                "Politely re-ask for date of birth, slowly. One short sentence."
            )
        )

    @function_tool()
    async def refuses_to_verify(self, ctx: RunContext[CallData]):
        """Customer refuses any further verification."""
        ctx.userdata.outcome = "verify_refused"
        _emit_outcome(ctx.userdata, "verify_refused")
        return ClosingAgent(self.data, intent="verify_refused", chat_ctx=None)


# ---------- Stage 2: Debt intro & reason ----------

STAGE2_TASK = """\
Current stage: 2 — Mention the debt and ask for payment capability.

You MUST ask the following question exactly word-for-word as written, as a single connected sentence:
"عليك مبلغ {amount} ريال من stc هل قادر على السداد الان طال عمرك ؟"
DO NOT add any greetings, DO NOT add any extra words, and DO NOT split it. Just this exact text.

Then call exactly one tool based on the customer's reply:
- commits_to_pay: customer says YES he will pay now / today / immediately
  ("أدفع الحين", "بسدد اليوم", "ما في مشكلة بدفع", "أنا جاهز", etc.).
  This SKIPS negotiation and goes straight to confirmation.
- already_paid: customer claims it's already paid (مدفوع / سددته / دفعت).
- proceed_to_negotiation: anything else — no, reason, hardship, partial offer,
  "will pay later", disputes, vague answers, etc.
"""


class Stage2DebtIntroAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=stage_instructions(
                data,
                STAGE2_TASK.format(amount=data.amount, debt_date=data.debt_date),
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        self.session.generate_reply(
            instructions=(
                "You must ask this exact question word-for-word, without any additions, as a single connected sentence: "
                f"\"عليك مبلغ {self.data.amount} ريال من stc هل قادر على السداد الان او وقت ثاني ؟\""
            )
        )

    @function_tool()
    async def commits_to_pay(self, ctx: RunContext[CallData]):
        """Customer agrees to pay now / today / immediately — full commitment."""
        today = datetime.date.today().isoformat()
        ctx.userdata.commitment = (
            f"full payment of {self.data.amount} SAR on {today}"
        )
        ctx.userdata.outcome = "committed"
        return Stage4RecapAgent(self.data, chat_ctx=None)

    @function_tool()
    async def already_paid(self, ctx: RunContext[CallData]):
        """Customer claims the debt is already paid."""
        ctx.userdata.outcome = "paid"
        return ClosingAgent(self.data, intent="paid", chat_ctx=None)

    @function_tool()
    async def disputes_debt(self, ctx: RunContext[CallData]):
        """Customer disputes the debt — wrong amount, not his, fraud."""
        ctx.userdata.dispute_open = True
        return DisputeAgent(self.data, chat_ctx=None)

    @function_tool()
    async def proceed_to_negotiation(self, ctx: RunContext[CallData]):
        """Customer gave a reason / denial / wants to discuss payment."""
        return Stage3NegotiationAgent(self.data, chat_ctx=None)


# ---------- Stage 3: Discovery-led negotiation ----------

STAGE3_NEGOTIATION_TASK = """\
Current stage: 3 — Negotiation. DISCOVERY-LED, never dictate.

Outstanding amount: {amount} SAR.

GOLDEN RULE: NEVER name an amount before the customer has named one.
Open by asking how much HE can pay, then judge his number.

ARITHMETIC RULES (critical — never skip):
- When customer names an amount X, ALWAYS compute: remainder = {amount} - X.
- State the remainder out loud in Arabic words: "الباقي [remainder] ريال"
- If customer offers X per month, compute EXACTLY how many months:
  months = {amount} / X. Tell him: "يعني حوالي [months] شهر للمبلغ كامل".
  NEVER round or guess. 500/month on a 10000 debt = 20 months, NOT 3.
- NEVER say "you'll finish in N months" without computing N = total / monthly.
- Track running totals: initial offer + remainder = {amount}. Always.

Q&A FLOW (one short sentence each turn):

1. Open: ask how much he can pay today / soon.
   Example: "كم تقدر تدفع اليوم طال عمرك؟"

2. Customer names a number A:
   • If A covers the full debt ({amount} SAR): ack, ask when
     (today/tomorrow), call full_payment(when_iso).
   • If A is at or above the IDEAL first payment ({ideal_first} SAR):
     ack ("زين"), ask WHEN he can transfer it. Move to step 4.
   • If A is between the FLOOR ({floor_first} SAR) and IDEAL ({ideal_first} SAR):
     ack briefly, ONE soft push to stretch a bit higher
     ("تقدر توصلها لـ {ideal_first} ريال؟"). Whatever he answers next, accept
     it (do NOT push twice). Move to step 4.
   • If A is BELOW the FLOOR ({floor_first} SAR):
     point out it's small vs the total in ONE short sentence and ask if he
     can increase. Do NOT name a number yourself yet.
     If he raises but is still below FLOOR, you MAY now suggest a
     stretch range like "تقدر توصلها لـ {floor_first} أو {ideal_first} ريال؟"
     — only after he has named at least one number. Whatever he commits
     to next, accept and move on. Never push more than twice.

3. (After step 2) Customer is still vague after 2 nudges, OR refuses, OR
   disputes — call vague_response / refuses_payment / disputes_debt.

4. You now have an initial amount X he agreed to. Ask WHEN he can pay it.
   Translate "اليوم", "بكرا", "بعد يومين" to a SPECIFIC ISO date using
   the dates context. ONE short sentence.

5. With initial X + initial date locked, compute remainder = {amount} - X.
   Ask when the REST will be paid:
   "والباقي [remainder] ريال متى تقدر تسدده؟"
   Aim for ~14 days from today. If he names a date much later, ask ONCE
   if he can pull it sooner. Then accept whatever he commits to.
   (If he says he can pay the WHOLE debt at once on a single date,
   call full_payment(when_iso) instead.)

   IMPORTANT: if the customer is vague or undecided about the remainder
   after 2 tries, but has ALREADY committed to the initial payment,
   call initial_agreed_remainder_later(initial_amount, initial_date_iso)
   to lock in what's agreed and schedule a follow-up for the rest.
   Do NOT call vague_response — that would lose the initial commitment.

6. With all four pieces (initial_amount, initial_date_iso, rest_amount,
   rest_date_iso) agreed, call partial_committed.

TOOLS:
- partial_committed(initial_amount, initial_date_iso, rest_amount, rest_date_iso):
  full deal captured — both initial and remainder amounts + dates agreed.
- initial_agreed_remainder_later(initial_amount, initial_date_iso):
  initial payment agreed but customer is vague about the remainder.
  Records the initial commitment and schedules a follow-up call for the rest.
- full_payment(when_iso): customer commits to pay the whole {amount} SAR
  in a single transfer on this ISO date.
- already_paid: customer asserts the debt is already paid.
- vague_response: customer has NOT committed to ANY amount at all — no
  concrete number, no concrete date, after 2 nudges.
- refuses_payment: flat refusal ("ما أدفع", "ما تستحقون").
- disputes_debt: claims it's not his / wrong amount / fraud.
- unclear: ambiguous reply — re-ask the SAME question.

HARD RULES (override every other instinct):
- ONE short sentence per turn (8–14 words). Q&A, not monologue.
- NEVER explain SIMAH, grace periods, or company policy unless he asks.
- NEVER name an amount first.
- NEVER use "والله". NEVER overuse "طال عمرك".
- Acknowledgements are ONE word: "أبشر" / "زين" / "تمام" / "طيب".
- Hardship gets a one-word ack and the next question — no empathy speech.
"""


class Stage3NegotiationAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        try:
            self._amount_int = int(str(data.amount).strip())
        except ValueError:
            self._amount_int = 0
        self._floor_first = max(1, self._amount_int // 20) if self._amount_int else 0
        self._ideal_first = max(1, self._amount_int // 10) if self._amount_int else 0
        super().__init__(
            instructions=stage_instructions(
                data,
                STAGE3_NEGOTIATION_TASK.format(
                    amount=data.amount,
                    floor_first=self._floor_first,
                    ideal_first=self._ideal_first,
                    remainder="the remainder",
                ),
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        self.session.generate_reply(
            instructions=(
                "Acknowledge his situation briefly with empathy, then offer "
                "options. Use this template: "
                "'مقدّر وضعك، خلنا نشوف حل بسيط — كم تقدر تدفع الحين أو قريب؟' "
                "DO NOT name any amount. TWO short sentences max."
            )
        )

    @function_tool()
    async def partial_committed(
        self,
        ctx: RunContext[CallData],
        initial_amount: float,
        initial_date_iso: str,
        rest_amount: float,
        rest_date_iso: str,
    ):
        """Customer agreed to a two-step plan: an initial good-faith
        payment plus a remainder on a later date.

        Args:
            initial_amount: SAR the customer commits to pay first.
            initial_date_iso: ISO date (YYYY-MM-DD) of the first payment.
            rest_amount: SAR the customer commits for the remainder.
            rest_date_iso: ISO date (YYYY-MM-DD) of the remainder payment.
        """
        ctx.userdata.commitment = (
            f"initial {int(initial_amount)} SAR on {initial_date_iso}, "
            f"remainder {int(rest_amount)} SAR on {rest_date_iso}"
        )
        ctx.userdata.outcome = "committed"
        return Stage4RecapAgent(self.data, chat_ctx=None)

    @function_tool()
    async def full_payment(self, ctx: RunContext[CallData], when_iso: str):
        """Customer commits to pay the FULL outstanding amount in a
        single transfer.

        Args:
            when_iso: ISO date (YYYY-MM-DD) of the single payment.
        """
        ctx.userdata.commitment = (
            f"full payment of {self.data.amount} SAR on {when_iso}"
        )
        ctx.userdata.outcome = "committed"
        return Stage4RecapAgent(self.data, chat_ctx=None)

    @function_tool()
    async def initial_agreed_remainder_later(
        self,
        ctx: RunContext[CallData],
        initial_amount: float,
        initial_date_iso: str,
    ):
        """Customer agreed to an initial payment but is vague about the
        remainder. Lock in the initial commitment and schedule a follow-up.

        Args:
            initial_amount: SAR the customer commits to pay first.
            initial_date_iso: ISO date (YYYY-MM-DD) of the first payment.
        """
        remainder = self._amount_int - int(initial_amount)
        ctx.userdata.commitment = (
            f"initial {int(initial_amount)} SAR on {initial_date_iso}, "
            f"remainder {remainder} SAR TBD"
        )
        ctx.userdata.outcome = "committed"
        return RescheduleAgent(self.data, chat_ctx=None)

    @function_tool()
    async def vague_response(self, ctx: RunContext[CallData]):
        """Customer would not name ANY concrete number at all after two nudges."""
        return RescheduleAgent(self.data, chat_ctx=None)

    @function_tool()
    async def refuses_payment(self, ctx: RunContext[CallData]):
        """Customer flatly refuses to pay. One empathetic attempt, then
        move to reschedule — never push more than once."""
        ctx.userdata.refusal_attempts += 1
        n = ctx.userdata.refusal_attempts
        if n >= 2:
            return RescheduleAgent(self.data, chat_ctx=None)
        instructions = (
            "Brief empathy ('أقدر ظرفك'), then ask softly if he could "
            "manage any small amount, even later this month — do NOT name "
            "a number yourself. ONE short sentence, max 12 words."
        )
        self.session.generate_reply(instructions=instructions)
        return None

    @function_tool()
    async def disputes_debt(self, ctx: RunContext[CallData]):
        """Customer disputes the debt — wrong amount, not his, fraud.
        Open a parallel dispute ticket and offer to start with any
        non-disputed portion (EC-5)."""
        ctx.userdata.dispute_open = True
        return DisputeAgent(self.data, chat_ctx=None)

    @function_tool()
    async def already_paid(self, ctx: RunContext[CallData]):
        """Customer claims the debt is already paid / settled."""
        ctx.userdata.outcome = "paid"
        return ClosingAgent(self.data, intent="paid", chat_ctx=None)

    @function_tool()
    async def unclear(self, ctx: RunContext[CallData]):
        """Reply is ambiguous, off-topic, or asks to repeat — re-ask same."""
        self.session.generate_reply(
            instructions=(
                "Re-ask the SAME question you just asked, slightly "
                "rephrased. ONE short sentence."
            )
        )



# ---------- Reschedule (vague exit from L3) ----------

RESCHEDULE_TASK = """\
Current stage: reschedule callback.

You are scheduling a FOLLOW-UP CALL (not a payment). Ask when would be a
good time for us to CALL HIM BACK — within the next two weeks.
ONE short sentence. Use "نكلمك" / "نتواصل" (we'll call you), NEVER "تدفع"
(you'll pay) — this is about scheduling a phone call, not a payment.

Then call exactly one tool:
- callback_scheduled(when): customer named a time. Pass a short description
  ('next Monday morning', 'tomorrow afternoon', or ISO date).
- refuses_to_schedule: customer won't commit to any callback time.

For clarifications, answer briefly and re-ask.
"""


class RescheduleAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=stage_instructions(data, RESCHEDULE_TASK),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        self.session.generate_reply(
            instructions=(
                "Ask when would be a good time to CALL HIM BACK. Use this "
                "style: 'ما فيه مشكلة، متى يناسبك نتواصل معك؟' "
                "ONE short sentence."
            )
        )

    @function_tool()
    async def callback_scheduled(self, ctx: RunContext[CallData], when: str):
        """Customer named a callback time.

        Args:
            when: short description of the time (e.g. 'next Monday morning').
        """
        ctx.userdata.callback_time = when
        if not ctx.userdata.outcome:
            ctx.userdata.outcome = "rescheduled"
        return Stage4RecapAgent(self.data, chat_ctx=None)

    @function_tool()
    async def refuses_to_schedule(self, ctx: RunContext[CallData]):
        """Customer refuses to commit to any callback time."""
        ctx.userdata.outcome = "refusal"
        return ClosingAgent(self.data, intent="refusal", chat_ctx=None)


# ---------- Stage 4: Recap & confirm ----------

STAGE4_TASK = """\
Current stage: 4 — Thank, recap, and confirm the commitment.

STEP-BY-STEP (one turn per step, max 12 words each):

Step A — Thank + positive reinforcement:
  Thank the customer for committing. Say it's a positive step / good sign
  ("خطوة ممتازة" or "شيء حلو"). ONE short sentence.

Step B — Recap the details:
  Use this template style: "ممتاز، يعطيك العافية، بنثبت الاتفاق على
  [المبلغ] بتاريخ [التاريخ]، تمام؟"
  Restate amounts and dates in Arabic words. ONE short sentence.

Step C — Offer a reminder:
  Ask if he would like a reminder call one day before the payment date.
  ONE short sentence.

Step D — Any questions:
  Ask if he has any other questions. ONE short sentence.

After Step D (or earlier if he says no questions), call recap_confirmed.

Commitment: {commitment}
Callback time (if rescheduled): {callback}

TOOLS:
- recap_confirmed: customer confirms and has no further questions.
- recap_minor_correction(correction): customer corrects a SMALL detail
  but the OVERALL plan stays the same.
- wants_to_renegotiate: customer is MATERIALLY changing the commitment —
  says he can't actually pay what was agreed.

For clarifications, answer briefly and continue through the steps.
"""


class Stage4RecapAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        commitment = data.commitment or "(no commitment recorded)"
        callback = data.callback_time or "(none)"
        super().__init__(
            instructions=stage_instructions(
                data,
                STAGE4_TASK.format(commitment=commitment, callback=callback),
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        has_commitment = self.data.commitment and "TBD" not in (self.data.commitment or "")
        has_partial = self.data.commitment and "TBD" in (self.data.commitment or "")
        has_callback = bool(self.data.callback_time)

        if has_partial and has_callback:
            hint = (
                "Thank the customer for his initial commitment — say it's a "
                "positive step ('خطوة ممتازة'). Recap the INITIAL payment "
                "amount and date in Arabic words, then mention you'll follow "
                "up about the remainder. Ask 'تمام كذا؟'. TWO short sentences.\n\n"
                f"Commitment: {self.data.commitment}.\n"
                f"Follow-up callback: {self.data.callback_time}.\n"
                "IMPORTANT: The callback is for FOLLOW-UP, not a payment date. "
                "Say 'بنتواصل معك' (we'll follow up) NOT 'بتدفع' (you'll pay)."
            )
        elif self.data.outcome == "rescheduled" and not self.data.commitment:
            hint = (
                "Thank the customer briefly. Recap ONLY the callback time — "
                "say 'بنكلمك' (we'll call you) on [date]. Ask to confirm. "
                "Do NOT say 'بتدفع' (you'll pay) — this is a CALLBACK, "
                f"not a payment. ONE short sentence.\n\n"
                f"Callback: {self.data.callback_time}."
            )
        else:
            hint = (
                "Thank the customer for his commitment — say it's a positive "
                "step ('خطوة ممتازة'). Then recap the amounts and dates in "
                "Arabic words and ask 'تمام كذا؟'. TWO short sentences.\n\n"
                f"Commitment: {self.data.commitment}."
            )
        self.session.generate_reply(instructions=hint)

    @function_tool()
    async def recap_confirmed(self, ctx: RunContext[CallData]):
        """Customer confirms the recap is correct and has no questions."""
        return ClosingAgent(self.data, intent="ok", chat_ctx=None)

    @function_tool()
    async def recap_minor_correction(
        self, ctx: RunContext[CallData], correction: str
    ):
        """Customer corrected a small detail but the overall plan stays.

        Args:
            correction: short description of what the customer corrected.
        """
        ctx.userdata.commitment = (
            f"{ctx.userdata.commitment or ''} | corrected: {correction}"
        ).strip(" |")
        return ClosingAgent(self.data, intent="ok", chat_ctx=None)

    @function_tool()
    async def wants_to_renegotiate(self, ctx: RunContext[CallData]):
        """Customer is materially backing out of the commitment.
        Move to reschedule — do NOT re-enter full negotiation."""
        ctx.userdata.commitment = None
        ctx.userdata.outcome = None
        return RescheduleAgent(self.data, chat_ctx=None)


# ---------- Dispute handling (EC-5) ----------

DISPUTE_TASK = """\
Current stage: dispute handling.

The customer disputes the debt (wrong amount, not mine, fraud).
Acknowledge calmly. Do NOT mention سمة, credit reporting, or legal action.
Do NOT push for payment. Inform him he can contact stc customer service
directly to verify. ONE or TWO short sentences.

Then call exactly one tool:
- accepts_undisputed(amount, when_iso): customer agrees to pay an
  undisputed amount today/soon. Pass SAR amount and ISO date.
- declines_partial: customer declines any payment until review.
- still_disputing_only: customer keeps disputing without engaging on
  partial — close with dispute outcome.
"""


class DisputeAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=stage_instructions(data, DISPUTE_TASK),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        self.session.generate_reply(
            instructions=(
                "Acknowledge his concern calmly. Tell him he can verify "
                "with stc customer service directly. Then ask if he wants "
                "to pay any non-disputed portion in good faith while "
                "checking. TWO short sentences max."
            )
        )

    @function_tool()
    async def accepts_undisputed(
        self,
        ctx: RunContext[CallData],
        amount: float,
        when_iso: str,
    ):
        """Customer agreed to pay an undisputed portion.

        Args:
            amount: SAR amount the customer commits to pay.
            when_iso: ISO YYYY-MM-DD payment date.
        """
        ctx.userdata.commitment = (
            f"undisputed partial {int(amount)} SAR on {when_iso} "
            f"(dispute under review)"
        )
        ctx.userdata.outcome = "dispute_partial"
        _emit_outcome(
            ctx.userdata, "dispute_partial",
            amount=int(amount), when=when_iso,
        )
        return Stage4RecapAgent(self.data, chat_ctx=None)

    @function_tool()
    async def declines_partial(self, ctx: RunContext[CallData]):
        """Customer declines any payment until the review concludes."""
        ctx.userdata.outcome = "dispute"
        _emit_outcome(ctx.userdata, "dispute_open")
        return ClosingAgent(self.data, intent="dispute", chat_ctx=None)

    @function_tool()
    async def still_disputing_only(self, ctx: RunContext[CallData]):
        """Customer keeps disputing without engaging on payment."""
        ctx.userdata.outcome = "dispute"
        _emit_outcome(ctx.userdata, "dispute_open")
        return ClosingAgent(self.data, intent="dispute", chat_ctx=None)


# ---------- Escalation / de-escalation (EC-4) ----------

ESCALATE_TASK = """\
Current stage: de-escalation.

The customer is upset / hostile. Goal: lower the temperature. Slow your
pace. Acknowledge feelings briefly without being defensive. Do NOT push
numbers. Do NOT argue back. ONE short sentence per turn.

Then call exactly one tool:
- de_escalated: customer calmed and we can continue the negotiation.
- escalate_to_human: customer is still upset / asks for a human or a
  supervisor / abusive language. Transfer the call.
- end_call_safely: customer demands the call end / has clearly hung up.
"""


class EscalationAgent(BaseCallAgent):
    def __init__(self, data: CallData, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=stage_instructions(data, ESCALATE_TASK),
            chat_ctx=chat_ctx,
        )
        self.data = data

    async def on_enter(self):
        self.session.generate_reply(
            instructions=(
                "Acknowledge the customer's frustration briefly without "
                "being defensive, slow your pace, and ask whether you can "
                "continue calmly or transfer him to a human agent. ONE "
                "short sentence."
            )
        )

    @function_tool()
    async def de_escalated(self, ctx: RunContext[CallData]):
        """Customer calmed — resume the appropriate stage."""
        if ctx.userdata.id_verified:
            return Stage3NegotiationAgent(self.data, chat_ctx=None)
        # Pre-verify anger: don't push further — close gracefully.
        ctx.userdata.outcome = "ended_by_customer"
        return ClosingAgent(self.data, intent="ended_by_customer", chat_ctx=None)

    @function_tool()
    async def escalate_to_human(self, ctx: RunContext[CallData]):
        """Transfer the call to a live human agent."""
        ctx.userdata.outcome = "escalated"
        _request_human_transfer(ctx.userdata, "angry_or_requested")
        _emit_outcome(ctx.userdata, "escalated")
        return ClosingAgent(self.data, intent="escalated", chat_ctx=None)

    @function_tool()
    async def end_call_safely(self, ctx: RunContext[CallData]):
        """Customer wants to end the call now."""
        ctx.userdata.outcome = "ended_by_customer"
        _emit_outcome(ctx.userdata, "ended_by_customer")
        return ClosingAgent(self.data, intent="ended_by_customer", chat_ctx=None)


# ---------- Closing (parameterized by intent) ----------

_CLOSING_HINTS = {
    "ok": (
        "Thank the customer warmly for cooperation. Use this style: "
        "'شاكرة لك تعاونك، الله يجزاك خير. وراح نتواصل معك قريب بإذن الله.' "
        "TWO short sentences max."
    ),
    "paid": (
        "Acknowledge his statement that the amount is paid, say you'll "
        "verify and update the record, apologize for the bother, and "
        "say goodbye. ONE or TWO short sentences."
    ),
    "busy": (
        "Politely acknowledge the bad timing. Use this style: "
        "'أبشر، بنتواصل معك في وقت مناسب لك بإذن الله.' ONE short sentence."
    ),
    "busy_callback": (
        "Confirm the callback time you just agreed on (use the date and "
        "time_of_day already recorded — speak the date in Arabic words), "
        "thank the customer, and say goodbye. ONE or TWO short sentences."
    ),
    "dnc": (
        "Acknowledge the do-not-call request, confirm it will be recorded, "
        "apologize for the disturbance, and say goodbye. ONE short sentence."
    ),
    "death": (
        "Express sincere condolences (الله يرحمه ويغفر له), thank the "
        "caller for letting you know, and say goodbye. ONE or TWO short "
        "sentences."
    ),
    "wrong_party": (
        "Apologize for the mix-up briefly and say goodbye. ONE short sentence."
    ),
    "referred": (
        "Thank the caller warmly for the help and say goodbye. ONE short "
        "sentence."
    ),
    "id_mismatch": (
        "Apologize that the data doesn't match this account, say you "
        "won't take more of his time, and say goodbye. ONE or TWO short "
        "sentences."
    ),
    "refusal": (
        "Acknowledge his decision politely. Use this style: "
        "'أشكرك على وقتك، وبنتواصل معك في وقت مناسب لك بإذن الله.' "
        "ONE short sentence. Do NOT threaten or pressure."
    ),
    "dispute": (
        "Acknowledge that he disputes the debt, say the case will be "
        "reviewed by the relevant team and someone will follow up, and "
        "say goodbye. ONE or TWO short sentences."
    ),
    "hard_refusal": (
        "Acknowledge his decision respectfully. Use this style: "
        "'أشكرك على وقتك، وبنتواصل معك في وقت مناسب لك بإذن الله.' "
        "ONE short sentence. NO threats, NO legal language."
    ),
    "escalated": (
        "Reassure briefly that a colleague will follow up with him "
        "shortly, thank him, and say goodbye. ONE short sentence."
    ),
    "ended_by_customer": (
        "Close politely with a brief goodbye. ONE short sentence."
    ),
    "verify_refused": (
        "Apologize briefly that you cannot continue without verification, "
        "invite him to call back on the official company number, and say "
        "goodbye. ONE or TWO short sentences. Do NOT mention any debt "
        "details."
    ),
}


# Intents that should trigger an SMS payment link after the goodbye plays.
# Only "ok" — meaning a confirmed commitment from Stage 4 recap. "paid"
# means the customer claims it's already settled, so we verify, not nudge.
_PAYMENT_LINK_INTENTS = {"ok"}


class ClosingAgent(BaseCallAgent):
    def __init__(
        self,
        data: CallData,
        *,
        intent: str,
        chat_ctx: ChatContext | None = None,
    ) -> None:
        hint = _CLOSING_HINTS.get(intent, _CLOSING_HINTS["ok"])
        super().__init__(
            instructions=stage_instructions(
                data,
                f"Current stage: closing (intent={intent}).\n{hint}\n"
                "Do not call any tool. Just say the parting line.",
            ),
            chat_ctx=chat_ctx,
        )
        self.data = data
        self.intent = intent

    async def on_enter(self):
        # Stop any in-flight speech / LLM turn so the goodbye is the next
        # thing the customer hears. Without this, a user reply (e.g. asking
        # a follow-up after recap_minor_correction) starts a new LLM turn
        # that races with hangup and the goodbye never plays.
        try:
            self.session.interrupt()
        except Exception:
            pass

        # Disable mic input so further user audio doesn't spawn a new turn
        # while we're playing the goodbye + tearing down the room.
        try:
            self.session.input.set_audio_enabled(False)
        except Exception:
            pass

        hint = _CLOSING_HINTS.get(self.intent, _CLOSING_HINTS["ok"])
        handle = self.session.generate_reply(
            instructions=hint,
            allow_interruptions=False,
        )
        try:
            await handle.wait_for_playout()
        except AttributeError:
            # Older API surface — fall back to awaiting the handle directly.
            try:
                await handle
            except Exception as e:
                logger.warning(f"closing speech await error: {e}")
        except Exception as e:
            logger.warning(f"closing speech playout error: {e}")

        # Side effects fire AFTER the goodbye plays — never delay the audio.
        if self.intent in _PAYMENT_LINK_INTENTS:
            _send_sms_payment_link(self.data, when=self.data.commitment)

        # Final disposition — single record per call for downstream PTP
        # reminders / retry cadence / DNC suppression.
        _emit_outcome(
            self.data,
            f"close_{self.intent}",
            outcome=self.data.outcome,
            commitment=self.data.commitment,
            callback_time=self.data.callback_time,
            referrer_mobile=self.data.referrer_mobile,
            refusal_attempts=self.data.refusal_attempts,
            angry_attempts=self.data.angry_attempts,
            dispute_open=self.data.dispute_open,
            payment_link_sent=self.data.payment_link_sent,
            id_verified=self.data.id_verified,
            dob_verified=self.data.dob_verified,
        )

        # Even with the mic disabled, hangup can race with stream teardown.
        # Either way the call is ending — swallow the engine-closed error.
        try:
            await self.hangup()
        except Exception as e:
            logger.warning(f"hangup race ignored: {e}")


# ---------- Entrypoint ----------

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load(
        min_speech_duration=0.05, min_silence_duration=0.4
    )


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name} via {LIVEKIT_URL}")
    await ctx.connect()

    try:
        dial_info = json.loads(ctx.job.metadata or "{}")
        phone_number = dial_info["phone_number"]
    except (json.JSONDecodeError, KeyError):
        logger.error("No valid phone_number in job metadata.")
        ctx.shutdown()
        return

    if not OUTBOUND_TRUNK_ID:
        logger.error("SIP_OUTBOUND_TRUNK_ID is not set.")
        ctx.shutdown()
        return

    data = CallData(
        customer_name=dial_info.get("name", "محمد"),
        amount=dial_info.get("amount", "10000"),
        debt_date=dial_info.get("debt_date", "2023-01-01"),
        national_id_last4=dial_info.get("national_id_last4", "1234"),
        dob=dial_info.get("dob", "1990-01-01"),
        phone_number=phone_number,
    )

    participant_identity = f"sip-{phone_number}"

    options = soniox.STTOptions(
        language_hints=["ar"],
    )
    azure_stt = azure.STT(
        language="ar-SA",                    # Najdi Saudi Arabic
        segmentation_silence_timeout_ms=700,
    )

    session = AgentSession[CallData](
        userdata=data,
        turn_handling={
            "endpointing": {
                "mode": "dynamic",
                "min_delay": 0.1,
                "max_delay": 0.6,
            },
            "interruption": {
                "enabled": True,
                "mode": "vad",
                "min_words": 2,
            },
        },
        #stt=deepgram.STT(model="nova-3", language="ar-SA"),
        stt=soniox.STT(params=options),
        llm=openai.LLM(model="gpt-4.1", temperature=0.4),
        tts=faseeh.TTS(
            base_url="https://api.munsit.com/api/v1",
            voice_id="ar-hijazi-female-2",
            model="faseeh-v1-preview",
            stability=0.75,
            speed=1.0,
        ),
        vad=ctx.proc.userdata["vad"],
    )

    @session.on("error")
    def _on_error(err):
        logger.error(f"session error: {err}")

    @session.on("user_input_transcribed")
    def _on_user_transcript(ev):
        if getattr(ev, "is_final", True):
            logger.info(f"USER  {ev.transcript!r}")

    @session.on("function_tools_executed")
    def _on_tools(ev):
        for fc in ev.function_calls:
            args = (fc.arguments or "").strip()
            if args in ("", "{}"):
                logger.info(f"TOOL  {fc.name}")
            else:
                logger.info(f"TOOL  {fc.name}  args={args}")

    @session.on("conversation_item_added")
    def _on_conv_item(ev):
        item = ev.item
        if getattr(item, "type", None) != "message":
            return
        if getattr(item, "role", None) != "assistant":
            return
        text = getattr(item, "text_content", None)
        if not text:
            return
        logger.info(f"AGENT {text!r}")

    #@session.on("metrics_collected")
    #def _on_metrics(ev):
    #    logger.info(f"metrics: {ev.metrics}")

    # Per-turn latency rollup. We stash EOU + LLM TTFT and emit one summary
    # line per response when TTS finishes. VAD per-second pings and short STT
    # chunks are dropped — they were ~85% of log volume and added no signal.
    _turn_state: dict[str, float] = {}

    @session.on("metrics_collected")
    def _on_metrics(ev):
        m = ev.metrics
        mtype = getattr(m, "type", None) or m.__class__.__name__.lower()

        if mtype == "vad_metrics":
            return
        if mtype == "stt_metrics":
            # STT timing isn't actionable for us here — EOU already covers it.
            return

        if mtype == "eou_metrics":
            eou = getattr(m, "end_of_utterance_delay", 0.0)
            tdel = getattr(m, "transcription_delay", 0.0)
            _turn_state["eou"] = eou
            _turn_state["transcript"] = tdel
            logger.info(f"EOU  {eou:.2f}s  (transcript {tdel:.2f}s)")
            return

        if mtype == "llm_metrics":
            ttft = getattr(m, "ttft", 0.0)
            pt = getattr(m, "prompt_tokens", 0)
            pc = getattr(m, "prompt_cached_tokens", 0)
            ct = getattr(m, "completion_tokens", 0)
            _turn_state["ttft"] = ttft
            cache_pct = (pc / pt * 100) if pt else 0
            logger.info(
                f"LLM  ttft {ttft:.2f}s  "
                f"prompt={pt} (cached {pc}, {cache_pct:.0f}%)  "
                f"completion={ct}"
            )
            return

        if mtype == "tts_metrics":
            ttfb = getattr(m, "ttfb", 0.0)
            audio = getattr(m, "audio_duration", 0.0)
            chars = getattr(m, "characters_count", 0)
            logger.info(
                f"TTS  ttfb {ttfb:.2f}s  audio {audio:.1f}s  chars={chars}"
            )
            eou = _turn_state.pop("eou", None)
            ttft = _turn_state.pop("ttft", None)
            if eou is not None and ttft is not None:
                total = eou + ttft + ttfb
                _turn_state.clear()
                logger.info(
                    f"TURN total {total:.2f}s  "
                    f"(EOU {eou:.2f} + TTFT {ttft:.2f} + TTFB {ttfb:.2f})"
                )
            return

    try:
        logger.info(f"dialing {phone_number} via trunk {OUTBOUND_TRUNK_ID}")
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=OUTBOUND_TRUNK_ID,
                sip_call_to=phone_number,
                participant_identity=participant_identity,
                participant_name="Phone User",
                wait_until_answered=True,
            )
        )

        participant = await ctx.wait_for_participant(identity=participant_identity)
        logger.info(f"participant joined: {participant.identity}")
        data.participant = participant
        data.sip_ready.set()

        await session.start(
            agent=Stage1RightPartyAgent(data),
            room=ctx.room,
            room_input_options=RoomInputOptions(
                participant_identity=participant_identity,
            ),
        )

    except api.TwirpError as e:
        logger.error(
            f"error creating SIP participant: {e.message}, "
            f"SIP status: {e.metadata.get('sip_status_code')} "
            f"{e.metadata.get('sip_status')}"
        )
        ctx.shutdown()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=os.getenv("AGENT_NAME", "outbound-caller-aws-local"),
            num_idle_processes=1,
        )
    )
