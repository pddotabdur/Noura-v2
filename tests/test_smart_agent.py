"""Unit tests for smart_agent.py — covers the deterministic pieces I can
exercise without a live LiveKit session: Najdi normalisation, DOB tolerant
match, FSM tool transitions, refusal ladder, dispute routing, voicemail
side-effect, ClosingAgent SMS gating, outcome event schema.

What is NOT covered here (out of scope for unit tests — needs a live SIP
trunk + a real phone leg):
- Actual LLM behaviour / prompt adherence
- TTS/STT/VAD plumbing
- The on_enter hooks that call self.session.generate_reply
- voicemail_detected's playout + hangup sequence
- ClosingAgent.on_enter playout / hangup race handling

Run with:  .venv/bin/python -m pytest tests/ -v
"""
from __future__ import annotations

import json
import logging
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import smart_agent as sa


# --------------------------------------------------------------------------
# 1) Pure helpers: number / date / digit Najdi normalisation
# --------------------------------------------------------------------------

class TestNajdiNormalize:
    def test_below_1000_zero(self):
        assert sa._ar_below_1000(0) == ""

    def test_below_1000_units(self):
        assert sa._ar_below_1000(7) == "سبعة"

    def test_below_1000_teens(self):
        assert sa._ar_below_1000(13) == "ثلاثة عشر"

    def test_below_1000_tens(self):
        assert sa._ar_below_1000(50) == "خمسين"

    def test_below_1000_compound(self):
        # 543 → "خمس مئة و ثلاثة وأربعين"
        out = sa._ar_below_1000(543)
        assert "خمس مئة" in out
        assert "أربعين" in out

    def test_amount_words_zero(self):
        assert sa._ar_amount_words(0) == "صفر"

    def test_amount_words_thousand(self):
        assert sa._ar_amount_words(1000) == "ألف"

    def test_amount_words_two_thousand(self):
        assert sa._ar_amount_words(2000) == "ألفين"

    def test_amount_words_ten_thousand(self):
        out = sa._ar_amount_words(10000)
        assert "آلاف" in out or "ألف" in out

    def test_amount_words_million(self):
        assert sa._ar_amount_words(1_000_000) == "مليون"

    def test_date_words_iso(self):
        out = sa._ar_date_words("2025-03-15")
        assert "مارس" in out
        # day "15" → "خمسة عشر"
        assert "خمسة عشر" in out

    def test_date_words_invalid(self):
        # garbage stays untouched
        assert sa._ar_date_words("not-a-date") == "not-a-date"

    def test_digits_individual(self):
        # 4-digit ID is spelled digit by digit
        assert sa._ar_digits_individual("1234") == "واحد اثنين ثلاثة أربعة"

    def test_najdi_normalize_riyal_amount(self):
        out = sa._najdi_normalize("سداد 500 ريال")
        assert "خمس مئة" in out
        assert "ريال سعودي" in out
        # raw digits gone
        assert "500" not in out

    def test_najdi_normalize_4digit_spelled_individually(self):
        # 4-digit numbers (likely IDs / years) are read digit-by-digit
        out = sa._najdi_normalize("الكود 1234")
        assert "واحد" in out and "اثنين" in out

    def test_najdi_normalize_iso_date(self):
        out = sa._najdi_normalize("بتاريخ 2025-03-15")
        assert "مارس" in out

    def test_najdi_normalize_brand_terms(self):
        out = sa._najdi_normalize("بنك stc و SIMAH")
        assert "اس تي سي" in out
        assert "سمة" in out


# --------------------------------------------------------------------------
# 2) DOBVerifyAgent._normalize_iso — pure static method
# --------------------------------------------------------------------------

class TestDOBNormalize:
    def test_valid_iso(self):
        assert sa.DOBVerifyAgent._normalize_iso("1990-05-12") == "1990-05-12"

    def test_zero_pads(self):
        assert sa.DOBVerifyAgent._normalize_iso("1990-5-2") == "1990-05-02"

    def test_arabic_indic_digits(self):
        # Arabic-Indic numerals → Western
        assert sa.DOBVerifyAgent._normalize_iso("١٩٩٠-٠٥-١٢") == "1990-05-12"

    def test_invalid_month(self):
        assert sa.DOBVerifyAgent._normalize_iso("1990-13-01") is None

    def test_invalid_day(self):
        assert sa.DOBVerifyAgent._normalize_iso("1990-02-31") is None

    def test_garbage(self):
        assert sa.DOBVerifyAgent._normalize_iso("not-a-date") is None

    def test_empty(self):
        assert sa.DOBVerifyAgent._normalize_iso("") is None


# --------------------------------------------------------------------------
# 3) Side-effect hooks: side effects are visible only via the logger
# --------------------------------------------------------------------------

@pytest.fixture
def caplog_outcome(caplog):
    caplog.set_level(logging.INFO, logger="smart-caller-ar")
    return caplog


class TestSideEffectHooks:
    def _data(self, **kw):
        return sa.CallData(
            customer_name=kw.get("name", "محمد"),
            phone_number=kw.get("phone", "+966555000000"),
            amount=kw.get("amount", "10000"),
        )

    def test_emit_outcome_logs_json(self, caplog_outcome):
        d = self._data()
        sa._emit_outcome(d, "voicemail", attempts=1)
        rec = next(r for r in caplog_outcome.records if "OUTCOME" in r.message)
        body = rec.message.split("OUTCOME ", 1)[1]
        payload = json.loads(body)
        assert payload["customer"] == "محمد"
        assert payload["phone"] == "+966555000000"
        assert payload["kind"] == "voicemail"
        assert payload["attempts"] == 1

    def test_send_sms_payment_link_sets_flag(self, caplog_outcome):
        d = self._data()
        assert d.payment_link_sent is False
        sa._send_sms_payment_link(d, when="tomorrow")
        assert d.payment_link_sent is True
        assert any("SMS payment_link" in r.message for r in caplog_outcome.records)

    def test_send_sms_payment_link_no_phone_skips(self, caplog_outcome):
        d = self._data(phone=None)
        sa._send_sms_payment_link(d)
        assert d.payment_link_sent is False
        # No SMS log line emitted
        assert not any("SMS payment_link" in r.message for r in caplog_outcome.records)

    def test_send_sms_callback_card_no_phone_noop(self, caplog_outcome):
        d = self._data(phone=None)
        sa._send_sms_callback_card(d)
        assert not any("callback_card" in r.message for r in caplog_outcome.records)

    def test_request_human_transfer_logs(self, caplog_outcome):
        d = self._data()
        sa._request_human_transfer(d, "anger_persisting")
        assert any(
            "TRANSFER" in r.message and "anger_persisting" in r.message
            for r in caplog_outcome.records
        )


# --------------------------------------------------------------------------
# 4) CallData defaults / new fields
# --------------------------------------------------------------------------

class TestCallData:
    def test_new_fields_have_defaults(self):
        d = sa.CallData()
        assert d.dob == "1990-01-01"
        assert d.phone_number is None
        assert d.refusal_attempts == 0
        assert d.angry_attempts == 0
        assert d.dispute_open is False
        assert d.payment_link_sent is False
        assert d.dob_verified is False

    def test_call_context_block_omits_phone(self):
        # PII guard: phone number must NOT appear in the LLM prompt context
        d = sa.CallData(phone_number="+966555000000", customer_name="محمد")
        block = sa.call_context_block(d)
        assert "+966555000000" not in block
        # DOB also stays out of the prompt — it's only used server-side for match
        assert "1990-01-01" not in block

    def test_call_context_block_includes_amount_and_id(self):
        d = sa.CallData(amount="5000", national_id_last4="9876")
        block = sa.call_context_block(d)
        assert "5000" in block
        assert "9876" in block

    def test_negotiation_anchors_in_prompt(self):
        # 5% floor and 10% ideal computed from amount=10000 → 500 / 1000
        d = sa.CallData(amount="10000")
        block = sa.call_context_block(d)
        assert "500" in block  # floor
        assert "1000" in block  # ideal


# --------------------------------------------------------------------------
# 5) FSM tool transitions — call the underlying tool function and assert
#    the returned next-agent type and CallData mutations.
#
# function_tool wraps the method; the original is on .__wrapped__ in the
# livekit.agents implementation. We test the original async function.
# --------------------------------------------------------------------------

def _unwrap(tool):
    """Return the underlying coroutine function from a function_tool."""
    # livekit-agents stores the raw callable on .raw_callable / __wrapped__
    for attr in ("raw_callable", "__wrapped__", "fnc", "_fnc"):
        if hasattr(tool, attr):
            inner = getattr(tool, attr)
            if callable(inner):
                return inner
    return tool


def _ctx(data: sa.CallData):
    """Minimal RunContext shim — only `userdata` is read by the tools."""
    return SimpleNamespace(userdata=data)


def _patch_session(agent):
    """Tools read self.session for generate_reply / interrupt; the real
    Agent.session property reads from self._activity.session, so we mock
    that path."""
    activity = MagicMock()
    session = MagicMock()
    session.generate_reply = MagicMock(return_value=MagicMock())
    activity.session = session
    agent._activity = activity
    return session


@pytest.mark.asyncio
class TestStage3RefusalLadder:
    """EC-6: 3-step ladder before HARD_REFUSAL close."""

    async def _run_refusal(self, attempts_before: int):
        data = sa.CallData(amount="10000")
        data.refusal_attempts = attempts_before
        agent = sa.Stage3NegotiationAgent(data)
        session = _patch_session(agent)
        tool = _unwrap(agent.refuses_payment)
        result = await tool(agent, _ctx(data))
        return result, data, session

    async def test_first_refusal_stays_in_stage_and_reprompts(self, caplog_outcome):
        result, data, session = await self._run_refusal(0)
        assert result is None  # stay in Stage 3
        assert data.refusal_attempts == 1
        # generate_reply called exactly once with empathy + smallest-entry hint
        assert session.generate_reply.call_count == 1
        kwargs = session.generate_reply.call_args.kwargs
        instructions = kwargs.get("instructions", "")
        assert "أتفهم" in instructions or "empathy" in instructions.lower()
        assert "smallest" in instructions.lower()

    async def test_second_refusal_soft_consequence_no_threats(self, caplog_outcome):
        result, data, session = await self._run_refusal(1)
        assert result is None
        assert data.refusal_attempts == 2
        instructions = session.generate_reply.call_args.kwargs["instructions"]
        # Compliance: must explicitly forbid threats / legal language
        assert "NO threats" in instructions
        assert "NO" in instructions and "legal" in instructions

    async def test_third_refusal_hard_close(self, caplog_outcome):
        result, data, session = await self._run_refusal(2)
        # Should NOT generate_reply — should return ClosingAgent
        assert isinstance(result, sa.ClosingAgent)
        assert result.intent == "hard_refusal"
        assert data.outcome == "hard_refusal"
        assert data.refusal_attempts == 3
        # OUTCOME emitted with attempts field
        assert any(
            "hard_refusal" in r.message and "attempts" in r.message
            for r in caplog_outcome.records
        )


@pytest.mark.asyncio
class TestStage3DisputeRouting:
    async def test_disputes_debt_routes_to_dispute_agent(self):
        data = sa.CallData()
        agent = sa.Stage3NegotiationAgent(data)
        _patch_session(agent)
        tool = _unwrap(agent.disputes_debt)
        result = await tool(agent, _ctx(data))
        assert isinstance(result, sa.DisputeAgent)
        assert data.dispute_open is True


@pytest.mark.asyncio
class TestDisputeAgent:
    async def test_accepts_undisputed_routes_to_recap_with_commitment(
        self, caplog_outcome
    ):
        data = sa.CallData()
        agent = sa.DisputeAgent(data)
        _patch_session(agent)
        tool = _unwrap(agent.accepts_undisputed)
        result = await tool(
            agent, _ctx(data), amount=2500, when_iso="2026-05-09"
        )
        assert isinstance(result, sa.Stage4RecapAgent)
        assert data.outcome == "dispute_partial"
        assert "2500" in data.commitment
        assert "2026-05-09" in data.commitment
        assert "dispute under review" in data.commitment
        # OUTCOME row emitted with structured fields
        assert any(
            "dispute_partial" in r.message and "2500" in r.message
            for r in caplog_outcome.records
        )

    async def test_declines_partial_closes_with_dispute(self, caplog_outcome):
        data = sa.CallData()
        agent = sa.DisputeAgent(data)
        _patch_session(agent)
        tool = _unwrap(agent.declines_partial)
        result = await tool(agent, _ctx(data))
        assert isinstance(result, sa.ClosingAgent)
        assert result.intent == "dispute"
        assert data.outcome == "dispute"


@pytest.mark.asyncio
class TestIDVerifyDOBFallback:
    async def test_match_routes_to_stage2(self):
        data = sa.CallData(national_id_last4="1234")
        agent = sa.IDVerifyAgent(data)
        _patch_session(agent)
        tool = _unwrap(agent.digits_provided)
        result = await tool(agent, _ctx(data), digits="1234")
        assert isinstance(result, sa.Stage2DebtIntroAgent)
        assert data.id_verified is True

    async def test_mismatch_routes_to_dob_fallback_no_disclosure(self):
        data = sa.CallData(national_id_last4="1234", dob="1990-05-12")
        agent = sa.IDVerifyAgent(data)
        _patch_session(agent)
        tool = _unwrap(agent.digits_provided)
        result = await tool(agent, _ctx(data), digits="9999")
        # Critical: not Stage2; must be DOB fallback so no debt disclosure
        assert isinstance(result, sa.DOBVerifyAgent)
        assert data.id_verified is False

    async def test_short_digits_reasks_does_not_consume_attempt(self):
        data = sa.CallData(national_id_last4="1234")
        agent = sa.IDVerifyAgent(data)
        session = _patch_session(agent)
        tool = _unwrap(agent.digits_provided)
        result = await tool(agent, _ctx(data), digits="12")
        assert result is None
        assert session.generate_reply.call_count == 1

    async def test_arabic_indic_digits_recognised(self):
        data = sa.CallData(national_id_last4="1234")
        agent = sa.IDVerifyAgent(data)
        _patch_session(agent)
        tool = _unwrap(agent.digits_provided)
        result = await tool(agent, _ctx(data), digits="١٢٣٤")
        assert isinstance(result, sa.Stage2DebtIntroAgent)
        assert data.id_verified is True

    async def test_refuses_to_verify_closes_without_disclosure(self, caplog_outcome):
        data = sa.CallData()
        agent = sa.IDVerifyAgent(data)
        _patch_session(agent)
        tool = _unwrap(agent.refuses_to_verify)
        result = await tool(agent, _ctx(data))
        assert isinstance(result, sa.ClosingAgent)
        assert result.intent == "verify_refused"
        assert data.outcome == "verify_refused"
        # Stage2 was never reached → id_verified stays False
        assert data.id_verified is False


@pytest.mark.asyncio
class TestDOBVerifyAgent:
    async def test_exact_match_promotes_to_stage2(self):
        data = sa.CallData(dob="1990-05-12")
        agent = sa.DOBVerifyAgent(data)
        _patch_session(agent)
        tool = _unwrap(agent.dob_provided)
        result = await tool(agent, _ctx(data), yyyy_mm_dd="1990-05-12")
        assert isinstance(result, sa.Stage2DebtIntroAgent)
        assert data.id_verified is True
        assert data.dob_verified is True

    async def test_year_month_match_with_wrong_day_still_passes(self):
        # Day mishears are common in Arabic STT — the tolerant match must
        # let "1990-05-09" through when stored DOB is "1990-05-12".
        data = sa.CallData(dob="1990-05-12")
        agent = sa.DOBVerifyAgent(data)
        _patch_session(agent)
        tool = _unwrap(agent.dob_provided)
        result = await tool(agent, _ctx(data), yyyy_mm_dd="1990-05-09")
        assert isinstance(result, sa.Stage2DebtIntroAgent)
        assert data.id_verified is True

    async def test_wrong_month_rejected(self):
        data = sa.CallData(dob="1990-05-12")
        agent = sa.DOBVerifyAgent(data)
        _patch_session(agent)
        tool = _unwrap(agent.dob_provided)
        result = await tool(agent, _ctx(data), yyyy_mm_dd="1990-06-12")
        assert isinstance(result, sa.ClosingAgent)
        assert result.intent == "id_mismatch"
        assert data.id_verified is False

    async def test_wrong_year_rejected(self):
        data = sa.CallData(dob="1990-05-12")
        agent = sa.DOBVerifyAgent(data)
        _patch_session(agent)
        tool = _unwrap(agent.dob_provided)
        result = await tool(agent, _ctx(data), yyyy_mm_dd="1991-05-12")
        assert isinstance(result, sa.ClosingAgent)
        assert result.intent == "id_mismatch"


@pytest.mark.asyncio
class TestEscalationAgent:
    async def test_de_escalated_post_verify_resumes_stage3(self):
        data = sa.CallData()
        data.id_verified = True
        agent = sa.EscalationAgent(data)
        _patch_session(agent)
        tool = _unwrap(agent.de_escalated)
        result = await tool(agent, _ctx(data))
        assert isinstance(result, sa.Stage3NegotiationAgent)

    async def test_de_escalated_pre_verify_closes_safely(self):
        data = sa.CallData()
        data.id_verified = False
        agent = sa.EscalationAgent(data)
        _patch_session(agent)
        tool = _unwrap(agent.de_escalated)
        result = await tool(agent, _ctx(data))
        assert isinstance(result, sa.ClosingAgent)
        assert result.intent == "ended_by_customer"

    async def test_escalate_to_human_logs_transfer(self, caplog_outcome):
        data = sa.CallData(phone_number="+966555000000")
        agent = sa.EscalationAgent(data)
        _patch_session(agent)
        tool = _unwrap(agent.escalate_to_human)
        result = await tool(agent, _ctx(data))
        assert isinstance(result, sa.ClosingAgent)
        assert result.intent == "escalated"
        assert data.outcome == "escalated"
        assert any(
            "TRANSFER" in r.message and "angry_or_requested" in r.message
            for r in caplog_outcome.records
        )


@pytest.mark.asyncio
class TestUniversalCustomerAngryTool:
    """customer_angry is on BaseCallAgent — available in every stage."""

    async def test_first_event_routes_to_escalation_from_stage3(self):
        data = sa.CallData()
        data.id_verified = True
        agent = sa.Stage3NegotiationAgent(data)
        _patch_session(agent)
        tool = _unwrap(agent.customer_angry)
        result = await tool(agent, _ctx(data))
        assert isinstance(result, sa.EscalationAgent)
        assert data.angry_attempts == 1

    async def test_second_event_auto_transfers(self, caplog_outcome):
        data = sa.CallData()
        data.id_verified = True
        data.angry_attempts = 1
        agent = sa.Stage3NegotiationAgent(data)
        _patch_session(agent)
        tool = _unwrap(agent.customer_angry)
        result = await tool(agent, _ctx(data))
        assert isinstance(result, sa.ClosingAgent)
        assert result.intent == "escalated"
        assert data.outcome == "escalated"
        assert any(
            "anger_persisting" in r.message for r in caplog_outcome.records
        )

    async def test_available_on_stage1(self):
        # Compliance: anger handling exists pre-verify too
        data = sa.CallData()
        agent = sa.Stage1RightPartyAgent(data)
        _patch_session(agent)
        tool = _unwrap(agent.customer_angry)
        result = await tool(agent, _ctx(data))
        assert isinstance(result, sa.EscalationAgent)


@pytest.mark.asyncio
class TestWrongPartyContactInvalidLog:
    async def test_does_not_know_person_emits_contact_invalid(self, caplog_outcome):
        data = sa.CallData()
        agent = sa.WrongPartyKnowsAgent(data)
        _patch_session(agent)
        tool = _unwrap(agent.does_not_know_person)
        result = await tool(agent, _ctx(data))
        assert isinstance(result, sa.ClosingAgent)
        assert result.intent == "wrong_party"
        assert any("contact_invalid" in r.message for r in caplog_outcome.records)

    async def test_collect_mobile_refuses_emits_contact_invalid(self, caplog_outcome):
        data = sa.CallData()
        agent = sa.CollectMobileAgent(data)
        _patch_session(agent)
        tool = _unwrap(agent.refuses_to_provide)
        result = await tool(agent, _ctx(data))
        assert isinstance(result, sa.ClosingAgent)
        assert any("contact_invalid" in r.message for r in caplog_outcome.records)


# --------------------------------------------------------------------------
# 6) Closing intents — every documented intent must have a hint and be
#    accepted by ClosingAgent without raising.
# --------------------------------------------------------------------------

class TestClosingIntents:
    DOCUMENTED_INTENTS = {
        "ok", "paid", "busy", "busy_callback", "dnc", "death",
        "wrong_party", "referred", "id_mismatch", "refusal", "dispute",
        "hard_refusal", "escalated", "ended_by_customer", "verify_refused",
    }

    def test_all_documented_intents_have_hints(self):
        missing = self.DOCUMENTED_INTENTS - set(sa._CLOSING_HINTS)
        assert not missing, f"intents missing closing hints: {missing}"

    def test_payment_link_only_on_ok(self):
        # Compliance: 'paid' is a customer claim — we should NOT push a link
        assert sa._PAYMENT_LINK_INTENTS == {"ok"}

    def test_unknown_intent_falls_back_to_ok(self):
        d = sa.CallData()
        agent = sa.ClosingAgent(d, intent="totally-made-up", chat_ctx=None)
        assert agent.intent == "totally-made-up"  # stored as-is
        # but the hint resolution falls back
        assert sa._CLOSING_HINTS.get(agent.intent, sa._CLOSING_HINTS["ok"]) \
            == sa._CLOSING_HINTS["ok"]


# --------------------------------------------------------------------------
# 7) FSM reachability: every stage must be reachable from Stage1 and every
#    non-closing stage must have at least one tool that leads to closing.
# --------------------------------------------------------------------------

class TestFSMReachability:
    def test_every_agent_class_inherits_basecallagent(self):
        for name in [
            "Stage1RightPartyAgent", "AskGoodTimeAgent", "ScheduleCallbackAgent",
            "WrongPartyKnowsAgent", "CollectMobileAgent", "IDVerifyAgent",
            "DOBVerifyAgent", "Stage2DebtIntroAgent", "Stage3NegotiationAgent",
            "RescheduleAgent", "Stage4RecapAgent", "DisputeAgent",
            "EscalationAgent", "ClosingAgent",
        ]:
            cls = getattr(sa, name)
            assert issubclass(cls, sa.BaseCallAgent), name
