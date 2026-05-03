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
from livekit.plugins import deepgram, faseeh, openai, silero
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
    amount: str = "10000"
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
_AR_HUNDREDS = ["", "مئة", "مئتين", "ثلاث مئة", "أربع مئة", "خمس مئة",
                "ست مئة", "سبع مئة", "ثمان مئة", "تسع مئة"]
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


_TERM_MAP = {
    "stc": "اس تي سي",
    "STC": "اس تي سي",
    "SAR": "ريال سعودي",
    "SIMAH": "سمة",
    "simah": "سمة",
}


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
    return text


# ---------- Persona (system-level prompt) ----------

PERSONA = """\
You are نورا (Nora), a female collections specialist at شركة توافق working on
behalf of بنك stc. You ALWAYS reply in Najdi Saudi Arabic; English only for
brand names (stc, simah).

ABSOLUTE RULES (these override every other instinct):
- Brevity: ONE short sentence per turn, target 8–14 words. NEVER two
  sentences unless explicitly required by the stage. No preambles, no
  explanations, no monologues.
- Q&A style: ask, listen, ack, ask. Do not narrate. Do not justify.
- NEVER say "والله". NEVER use oaths.
- Use "طال عمرك" ONLY in the very first greeting; never again.
- Discovery, not dictation: NEVER name an amount before the customer has
  named one. Only counter-offer AFTER he proposes a number.
- Mid-call salaam: if the customer says "السلام عليكم" mid-conversation,
  reply with just "أبشر" or "تفضل" and continue. Do NOT say "وعليكم
  السلام" — that sounds like restarting the call.

ACKNOWLEDGEMENTS (use one short word, never a phrase):
- "أبشر", "زين", "تمام", "طيب", "ولا يهمك" — pick one, move on.
- Skip flowery empathy. Hardship gets a one-word ack and a question.

NUMBERS / DATES / IDENTITY:
- Amounts in Arabic words ("خمس مئة ريال", not "500 ريال").
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

You have just been connected. Greet the customer briefly, identify yourself
as نورا from شركة توافق on behalf of بنك stc, and ask whether you are speaking
with الأستاذ {customer_name}. Listen carefully to their reply.

Then call exactly one tool based on what you hear:
- right_party: customer confirms they ARE {customer_name}.
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
                "Open the call now. Greet warmly and identify yourself as "
                "نورا from شركة توافق on behalf of بنك stc, then ask whether "
                f"you are speaking with الأستاذ {self.data.customer_name}. "
                "ONE short sentence."
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
Before discussing the account, briefly ask whether NOW is a good time to
talk for a few minutes. ONE short sentence, friendly and unhurried.

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
                "Briefly ask the customer if NOW is a good time to talk "
                "for a few minutes. Friendly, unhurried, ONE short sentence."
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

Then call exactly one tool:
- digits_provided(digits): customer spoke 4 digits. Pass as a 4-character
  ASCII string. Convert Arabic-word numbers to digit characters
  ('واحد اثنين ثلاثة أربعة' → '1234'). If the customer says them as one
  number (e.g. 'ألف ومئتين وأربعة وثلاثين' or '١٢٣٤'), still pass '1234'.
- unclear: customer asked to repeat / gave fewer than 4 digits / off-topic.
  Use whenever no clear 4-digit answer is given. Never guess.
- refuses_to_verify: customer flatly refuses to share verification details
  ("ما أعطيك", "ما أعطي معلوماتي", "مين أنتي أصلاً").

The stored last-4 digits are in the call data; you do NOT speak them.
On a single mismatch, the system silently switches to date-of-birth as
an alternate verification method. Until verification succeeds, you must
NOT mention the debt, amount, or due date.
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
        self.session.generate_reply(
            instructions=(
                "Politely ask the customer for the last 4 digits of his "
                "national ID to confirm identity. One short sentence."
            )
        )

    @function_tool()
    async def digits_provided(self, ctx: RunContext[CallData], digits: str):
        """Customer spoke 4 digits — verify against stored last-4.

        Args:
            digits: 4-character ASCII digit string, e.g. '1234'.
        """
        clean = re.sub(r"\D", "", digits.translate(_AR_INDIC_DIGITS))
        if len(clean) != 4:
            self.session.generate_reply(
                instructions=(
                    "You couldn't catch all 4 digits. Politely ask the "
                    "customer to repeat the last 4 digits of the national "
                    "ID, slowly. One short sentence."
                )
            )
            return

        if clean == self.data.national_id_last4:
            ctx.userdata.id_verified = True
            return Stage2DebtIntroAgent(self.data, chat_ctx=None)

        # Single mismatch — switch to DOB fallback (EC-3) instead of
        # making the customer re-guess his own ID. Do NOT disclose any
        # debt detail in the meantime.
        return DOBVerifyAgent(self.data, chat_ctx=None)

    @function_tool()
    async def unclear(self, ctx: RunContext[CallData]):
        """Customer asked to repeat or didn't give a clear 4-digit reply."""
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
Current stage: 2 — Mention the debt, listen.

In ONE short sentence: state that there is an outstanding amount of
{amount} SAR on his stc account. Do NOT ask why it isn't paid. Do NOT
explain SIMAH or company policy. Just a brief mention, then wait.

Then call exactly one tool:
- already_paid: customer claims it's already paid (مدفوع / سددته / دفعت).
- proceed_to_negotiation: anything else (reason, hardship, willingness to
  discuss, disputes, "will pay later", etc).
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
                f"State briefly that there is an outstanding amount of "
                f"{self.data.amount} SAR on his stc account. ONE short "
                "sentence, no question, no explanation."
            )
        )

    @function_tool()
    async def already_paid(self, ctx: RunContext[CallData]):
        """Customer claims the debt is already paid."""
        ctx.userdata.outcome = "paid"
        return ClosingAgent(self.data, intent="paid", chat_ctx=None)

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

Q&A FLOW (one short sentence each turn):

1. Open: ask how much he can pay today / soon.
   Example: "كم تقدر تدفع اليوم طال عمرك؟"

2. Customer names a number A:
   • If A covers the full debt: ack, ask when (today/tomorrow), call
     full_payment(when_iso).
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

5. With initial X + initial date locked, ask when the REST will be paid:
   "والباقي ({remainder} of {amount} SAR) متى تقدر تسدده؟"
   Aim for ~14 days from today. If he names a date much later, ask ONCE
   if he can pull it sooner. Then accept whatever he commits to.
   (If he says he can pay the WHOLE debt at once on a single date,
   call full_payment(when_iso) instead.)

6. With all four pieces (initial_amount, initial_date_iso, rest_amount,
   rest_date_iso) agreed, call partial_committed.

TOOLS:
- partial_committed(initial_amount, initial_date_iso, rest_amount, rest_date_iso):
  full deal captured.
- full_payment(when_iso): customer commits to pay the whole {amount} SAR
  in a single transfer on this ISO date.
- already_paid: customer asserts the debt is already paid.
- vague_response: after pushing twice, no concrete number / no concrete
  date ("بشوف", "إن شاء الله", "ما أدري").
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
                "Open Stage 3 by asking the customer how much he can pay "
                "today (or soon). DO NOT name any amount. ONE short "
                "sentence, e.g. 'كم تقدر تدفع اليوم طال عمرك؟' — but vary "
                "the phrasing naturally."
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
    async def vague_response(self, ctx: RunContext[CallData]):
        """Customer would not name a concrete number / date after two nudges."""
        return RescheduleAgent(self.data, chat_ctx=None)

    @function_tool()
    async def refuses_payment(self, ctx: RunContext[CallData]):
        """Customer flatly refuses to pay. Drives the EC-6 ladder:
        attempt 1 → empathy + smallest-entry ask, attempt 2 → soft
        consequence (no threats), attempt 3+ → HARD_REFUSAL close."""
        ctx.userdata.refusal_attempts += 1
        n = ctx.userdata.refusal_attempts
        if n >= 3:
            ctx.userdata.outcome = "hard_refusal"
            _emit_outcome(ctx.userdata, "hard_refusal", attempts=n)
            return ClosingAgent(self.data, intent="hard_refusal", chat_ctx=None)
        if n == 1:
            instructions = (
                "ONE-word empathy ('أتفهم'), then ask softly what the SMALLEST "
                "amount he could manage today as a good-faith gesture would "
                "be — do NOT name a number yourself. ONE short sentence."
            )
        else:  # n == 2
            instructions = (
                "Briefly mention that without any commitment now the case "
                "will be returned to the bank per policy — NO threats, NO "
                "legal language. Then ask one last time if a token small "
                "payment today is possible. ONE short sentence."
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

The customer was vague about a plan. Politely ask when would be a better
time to follow up — within the next two weeks. ONE short sentence.

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
                "Politely ask when would be a good time to follow up "
                "within the next two weeks. One short sentence."
            )
        )

    @function_tool()
    async def callback_scheduled(self, ctx: RunContext[CallData], when: str):
        """Customer named a callback time.

        Args:
            when: short description of the time (e.g. 'next Monday morning').
        """
        ctx.userdata.callback_time = when
        ctx.userdata.outcome = "rescheduled"
        return Stage4RecapAgent(self.data, chat_ctx=None)

    @function_tool()
    async def refuses_to_schedule(self, ctx: RunContext[CallData]):
        """Customer refuses to commit to any callback time."""
        ctx.userdata.outcome = "refusal"
        return ClosingAgent(self.data, intent="refusal", chat_ctx=None)


# ---------- Stage 4: Recap & confirm ----------

STAGE4_TASK = """\
Current stage: 4 — Recap and confirm the commitment.

You have a commitment from the customer. Briefly recap it back to him so he
can confirm — mention the amount and timing in Arabic words. ONE or TWO
short sentences. The exact commitment text is provided below.

Commitment to recap: {commitment}
Callback time (if rescheduled): {callback}

Then call exactly one tool:
- recap_confirmed: customer confirms the recap is correct.
- recap_minor_correction(correction): customer corrects a SMALL detail
  (different specific date, time of day, exact amount typo) but the
  OVERALL plan stays the same. Pass a short description.
- wants_to_renegotiate: customer is MATERIALLY changing the commitment —
  says he can't actually pay what was just agreed, asks to split the
  amount, asks for instalments, or asks for substantially more time.
  We will return to Stage 3 to find a workable plan. This is the
  correct tool when the customer is backing out of the commitment, even
  if politely.

For clarifications, answer briefly and re-pose the recap. Do NOT default
to recap_minor_correction when the customer is actually backing out.
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
        if self.data.outcome == "rescheduled":
            hint = (
                "Recap the agreed callback time in ONE short sentence and "
                "ask the customer to confirm. Speak the day in Arabic "
                f"words. Callback: {self.data.callback_time}."
            )
        else:
            hint = (
                "Recap the commitment in ONE short sentence — amounts in "
                "Arabic words, dates in Arabic words, then ask 'تمام كذا؟' "
                "or 'صح؟'. NO preamble. NO explanation.\n\n"
                f"Commitment recorded: {self.data.commitment}."
            )
        self.session.generate_reply(instructions=hint)

    @function_tool()
    async def recap_confirmed(self, ctx: RunContext[CallData]):
        """Customer confirms the recap is correct."""
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
        Reset and re-enter the discovery-led negotiation."""
        ctx.userdata.commitment = None
        ctx.userdata.outcome = None
        return Stage3NegotiationAgent(self.data, chat_ctx=None)


# ---------- Dispute handling (EC-5) ----------

DISPUTE_TASK = """\
Current stage: dispute handling.

The customer disputes the debt (wrong amount, not mine, fraud). Two
parallel actions:
  1. Acknowledge briefly — a review will be opened by the back office.
  2. Ask if he wants to pay any UNDISPUTED portion now in good faith
     while the review proceeds.

ONE short sentence. NEVER name a number first — let him propose.

Then call exactly one tool:
- accepts_undisputed(amount, when_iso): customer agrees to pay an
  undisputed amount today/soon. Pass SAR amount and ISO date.
- declines_partial: customer declines any payment until review.
- still_disputing_only: customer keeps disputing without engaging on
  partial — close with dispute outcome and back-office review.
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
                "Acknowledge briefly that a review will be opened, then ask "
                "if he can pay any non-disputed portion now in good faith. "
                "ONE short sentence."
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
        "Thank the customer warmly for his cooperation, wish him well, "
        "and say goodbye. ONE short sentence."
    ),
    "paid": (
        "Acknowledge his statement that the amount is paid, say you'll "
        "verify and update the record, apologize for the bother, and "
        "say goodbye. ONE or TWO short sentences."
    ),
    "busy": (
        "Politely acknowledge the bad timing, say you'll call back at a "
        "more convenient time, and say goodbye. ONE short sentence."
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
        "Acknowledge his decision politely, mention the case will proceed "
        "per company policy, and say goodbye respectfully. ONE or TWO "
        "short sentences. Do NOT threaten or pressure."
    ),
    "dispute": (
        "Acknowledge that he disputes the debt, say the case will be "
        "reviewed by the relevant team and someone will follow up, and "
        "say goodbye. ONE or TWO short sentences."
    ),
    "hard_refusal": (
        "Acknowledge his decision a final time, briefly note the case will "
        "be returned to the bank per policy, and close respectfully. ONE "
        "short sentence. NO threats, NO legal language."
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

    session = AgentSession[CallData](
        userdata=data,
        turn_handling={
            "endpointing": {
                "mode": "dynamic",
                "min_delay": 0.2,
                "max_delay": 1.0,
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
            voice_id="ybQaNl0nzt9TjN3Oh1zzyNgp",
            model="faseeh-v1-preview",
            stability=0.75,
            speed=0.9,
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
            agent_name=os.getenv("AGENT_NAME", "outbound-caller-smart"),
            num_idle_processes=1,
        )
    )
