# Noura v2 — Agent Skills & Change Log

Reference for the Noura outbound-collections agent (`smart_agent.py`).
Documents every flow the agent can run, where the boundaries are between
LLM judgement and deterministic code, and which side-effects each
disposition triggers.

---

## 1. Architecture at a glance

The agent is a finite-state machine implemented as a chain of `Agent`
subclasses. Each stage owns one short prompt, a small set of
`@function_tool` transitions, and an `on_enter` hook that emits the
opening line of the stage. State transitions happen by **returning the
next agent from a tool** — LiveKit's framework swaps the active agent
in place, preserving the chat context.

### Why not `AgentTask` / `TaskGroup`

`TaskGroup` is the right primitive for *structured data collection* —
it summarises the group's chat history when the group exits and hands
the result back to the controlling agent. That summary is a full LLM
call. Our flow doesn't backtrack across stages and doesn't need a
post-hoc data extraction pass — the data is captured directly by tool
arguments (`partial_committed(initial_amount, initial_date_iso, ...)`).
Using `TaskGroup` would buy us nothing structural and cost an extra
LLM turn per group exit.

Sub-agents share `instructions` prefixes, so OpenAI prompt caching
keeps the PERSONA + `call_context_block` block hot across transitions.

### Latency budget per turn

| Component                       | Target  |
| ------------------------------- | ------- |
| EOU detection (dynamic)         | 0.2–1 s |
| LLM TTFT (gpt-4.1, cached prefix) | ~0.5 s |
| TTS TTFB (Faseeh streaming)     | ~0.3 s |
| **End-to-end turn target**      | < 1.5 s |

`metrics_collected` emits a `TURN total` line per turn for monitoring.

---

## 2. Stage map (the FSM)

```
Stage1RightPartyAgent
  ├─ right_party              → AskGoodTimeAgent
  ├─ wrong_party              → WrongPartyKnowsAgent
  │                              ├─ knows_person          → CollectMobileAgent
  │                              │                           ├─ mobile_provided    → ClosingAgent("referred")
  │                              │                           └─ refuses_to_provide → ClosingAgent("wrong_party")
  │                              └─ does_not_know_person  → ClosingAgent("wrong_party")
  ├─ caller_busy              → ScheduleCallbackAgent
  │                              ├─ callback_time          → ClosingAgent("busy_callback")
  │                              ├─ refuses_to_schedule    → ClosingAgent("busy")
  │                              └─ unclear (re-ask)
  ├─ do_not_call              → ClosingAgent("dnc")
  └─ customer_deceased        → ClosingAgent("death")

AskGoodTimeAgent
  ├─ good_time_now            → IDVerifyAgent
  └─ bad_time_now             → ScheduleCallbackAgent

IDVerifyAgent
  ├─ digits match             → Stage2DebtIntroAgent
  ├─ digits mismatch          → DOBVerifyAgent          ← EC-3 fallback (NEW)
  └─ refuses_to_verify        → ClosingAgent("verify_refused")  ← (NEW)

DOBVerifyAgent                                          ← (NEW)
  ├─ dob match (year+month)   → Stage2DebtIntroAgent
  ├─ dob mismatch             → ClosingAgent("id_mismatch")
  └─ refuses_to_verify        → ClosingAgent("verify_refused")

Stage2DebtIntroAgent
  ├─ already_paid             → ClosingAgent("paid")
  └─ proceed_to_negotiation   → Stage3NegotiationAgent

Stage3NegotiationAgent
  ├─ partial_committed        → Stage4RecapAgent
  ├─ full_payment             → Stage4RecapAgent
  ├─ vague_response           → RescheduleAgent
  ├─ refuses_payment          → EC-6 ladder (in-place, no transition) ← (NEW)
  │                              attempt 1: empathy + smallest-entry ask
  │                              attempt 2: soft policy consequence
  │                              attempt 3: ClosingAgent("hard_refusal")
  ├─ disputes_debt            → DisputeAgent             ← (NEW)
  └─ already_paid             → ClosingAgent("paid")

DisputeAgent                                              ← (NEW)
  ├─ accepts_undisputed       → Stage4RecapAgent
  ├─ declines_partial         → ClosingAgent("dispute")
  └─ still_disputing_only     → ClosingAgent("dispute")

Stage4RecapAgent
  ├─ recap_confirmed          → ClosingAgent("ok")        (SMS payment link fires)
  ├─ recap_minor_correction   → ClosingAgent("ok")        (SMS payment link fires)
  └─ wants_to_renegotiate     → Stage3NegotiationAgent

EscalationAgent                                           ← (NEW)
  ├─ de_escalated (verified)  → Stage3NegotiationAgent
  ├─ de_escalated (pre-verify)→ ClosingAgent("ended_by_customer")
  ├─ escalate_to_human        → ClosingAgent("escalated") + transfer hook
  └─ end_call_safely          → ClosingAgent("ended_by_customer")
```

Universal tools (defined on `BaseCallAgent`, available in every stage):

- `voicemail_detected` — leaves a 1-sentence compliant voicemail (no
  debt details), triggers `_send_sms_callback_card`, then hangs up.
- `customer_angry` — first event routes to `EscalationAgent`; second
  event auto-transfers to a human and closes.

---

## 3. Skills (the "what each agent does" reference)

### BaseCallAgent
Base class. Owns hangup, the Najdi-normalising `tts_node`, and the two
universal tools above. Subclasses set `self.data: CallData` in
`__init__` so the universal tools can route correctly.

### Stage1RightPartyAgent
Greet, identify, and confirm we're speaking to the named customer.
Routes DNC, deceased, busy, wrong-party, right-party.

### AskGoodTimeAgent
Confirms now is a good time to talk before any account discussion.

### ScheduleCallbackAgent
Captures a callback day + time-of-day when the customer is busy. Tool
argument shape: `callback_time(day_iso, time_of_day)` where
`time_of_day ∈ {'morning','afternoon','evening','HH:MM'}`.

### WrongPartyKnowsAgent → CollectMobileAgent
EC-2 path. If the wrong party knows the customer, captures a referral
number; either way the call ends. `_emit_outcome("contact_invalid")`
fires when no referral is given.

### IDVerifyAgent (revised)
Asks for last-4 of national ID. **Single mismatch silently switches
to DOB fallback** instead of re-asking. Refusal closes the call
**without** disclosing the debt — compliance hard requirement.

### DOBVerifyAgent (new)
Alternate verification when the ID digits didn't match. Uses a
**tolerant year+month match** because Arabic STT frequently misshears
the day. Mismatch → `id_mismatch` close (still no disclosure beyond
the original Stage 2 mention point — Stage 2 isn't reached without
verification).

### Stage2DebtIntroAgent
ONE sentence stating the outstanding amount. Routes to `paid` (claim)
or negotiation.

### Stage3NegotiationAgent (revised)
Discovery-led negotiation. Internal floor (5%) and ideal (10%) anchors
inform the LLM but are never spoken first. Captures either a full
single-payment commitment or a two-step plan (initial good-faith +
remainder within ~14 days).

**Refusal ladder (EC-6) is in-place** — `refuses_payment` increments
a counter and uses `session.generate_reply` to issue the next ladder
prompt without swapping agents. This avoids an extra LLM turn per
ladder step.

**Dispute** routes to `DisputeAgent` instead of closing immediately,
implementing EC-5's parallel ask.

### DisputeAgent (new)
Acknowledges the dispute (a back-office review will be opened) and
asks for any undisputed portion. If accepted, runs through Stage 4
recap so the customer hears confirmation.

### EscalationAgent (new)
EC-4 de-escalation. Slow pace, one acknowledgement, then route:
calmed → resume; still angry / asks for human → transfer; wants to
hang up → close. Pre-verification anger always closes safely without
disclosure.

### RescheduleAgent
Captures a follow-up time for vague non-commitments out of Stage 3.

### Stage4RecapAgent
Reads back the commitment for confirmation. `wants_to_renegotiate`
returns to Stage 3 (the only backtracking edge in the FSM).

### ClosingAgent (revised)
Plays the goodbye, then **after** the audio finishes:
1. Fires `_send_sms_payment_link` if `intent == "ok"` (a Stage 4
   confirmed commitment).
2. Emits one structured `OUTCOME close_<intent>` record carrying the
   complete `CallData` snapshot.
3. Hangs up.

The mic is disabled and the in-flight LLM turn is interrupted on
entry so the goodbye is the next thing the customer hears.

---

## 4. Side-effect hooks

All four are **stubs** at the top of `smart_agent.py`. Replace the
bodies with calls to your real providers.

```python
def _emit_outcome(data, kind, **fields)        # JSON log line
def _send_sms_payment_link(data, when=None)    # payment link / IBAN
def _send_sms_callback_card(data)              # post-voicemail / no-answer
def _request_human_transfer(data, reason)      # live-agent transfer
```

`_emit_outcome` is the single integration point a downstream worker
needs to subscribe to in order to drive PTP reminder scheduling
(T-24h, T-2h, T+0), retry cadence, dispute ticketing, and DNC
suppression.

---

## 5. Outcome event schema

Every outcome line is JSON inside a `logger.info("OUTCOME ...")`
record. Stable fields:

| field            | type  | always present | notes                                   |
| ---------------- | ----- | -------------- | --------------------------------------- |
| `customer`       | str   | yes            | from `CallData.customer_name`           |
| `phone`          | str   | yes            | E.164 from job metadata                 |
| `kind`           | str   | yes            | discriminator (see below)               |

Per-kind extras are documented inline. The closing record carries the
full snapshot:

```json
{
  "customer": "...", "phone": "+9665...",
  "kind": "close_ok",
  "outcome": "committed",
  "commitment": "initial 1000 SAR on 2026-05-05, remainder 9000 SAR on 2026-05-19",
  "callback_time": null,
  "referrer_mobile": null,
  "refusal_attempts": 0,
  "angry_attempts": 0,
  "dispute_open": false,
  "payment_link_sent": true,
  "id_verified": true,
  "dob_verified": false
}
```

### Kind vocabulary

| `kind`                 | emitted by                                  | meaning                                        |
| ---------------------- | ------------------------------------------- | ---------------------------------------------- |
| `voicemail`            | `BaseCallAgent.voicemail_detected`          | answering machine reached                      |
| `contact_invalid`      | wrong-party paths                           | dialer should retire / correct the number     |
| `verify_refused`       | ID/DOB verify refusal                       | retry on a future call; no disclosure made    |
| `id_mismatch`          | DOB also failed                             | verify offline; do not re-call without check  |
| `hard_refusal`         | Stage 3 ladder, attempt 3                   | return to bank per policy                     |
| `escalated`            | `EscalationAgent` / 2nd anger event         | live-agent transfer                           |
| `ended_by_customer`    | escalation `end_call_safely`                | customer asked to end                         |
| `dispute_open`         | `DisputeAgent` declines / still-disputing   | back-office review queue                      |
| `dispute_partial`     | `DisputeAgent.accepts_undisputed`           | partial PTP + parallel review                 |
| `close_<intent>`       | `ClosingAgent.on_enter`                     | final per-call snapshot (one per call)        |

`kind` values without a `close_` prefix are **mid-call** events. The
`close_<intent>` record is emitted **once** per call, last.

---

## 6. Compliance guarantees enforced in code

| Requirement                                       | Where enforced                                                                 |
| ------------------------------------------------- | ------------------------------------------------------------------------------ |
| No debt disclosure before identity verified       | Stage 2 only reachable after `id_verified=True`; PERSONA repeats the rule      |
| No threats / legal scare language                 | Refusal ladder + closing intents both call this out in their stage prompts     |
| Voicemail carries no sensitive data               | `voicemail_detected` instructions explicitly forbid amounts                   |
| DNC honored                                       | `do_not_call` immediately closes with `dnc` intent — no further outreach      |
| Identity refusal does not leak debt               | `verify_refused` intent has its own closing hint that omits all account detail |
| One-record-per-call audit trail                   | `_emit_outcome("close_<intent>", ...)` in `ClosingAgent.on_enter`              |

---

## 7. Mapping `call-flow.txt` → code

| Call-flow item                          | Implementation                                                                    |
| --------------------------------------- | --------------------------------------------------------------------------------- |
| HP-1 Immediate Full Payment             | `Stage3NegotiationAgent.full_payment(when_iso)` → recap → SMS link on close       |
| HP-2 Scheduled Full Payment             | same path; downstream worker reads `commitment` + `close_ok` to schedule reminders |
| HP-3 Partial + Plan                     | `Stage3NegotiationAgent.partial_committed(initial_amount, initial_date_iso, rest_amount, rest_date_iso)` |
| HP-4 Delay → salary date                | LLM resolves "بعد الراتب" against `call_context_block` dates → partial_committed   |
| EC-1 No-answer / voicemail              | `voicemail_detected` (compliant message + SMS callback card)                      |
| EC-2 Wrong number                       | `WrongPartyKnowsAgent` + `_emit_outcome("contact_invalid")`                       |
| EC-3 ID verification failed             | `IDVerifyAgent` mismatch → `DOBVerifyAgent`; refusal closes without disclosure    |
| EC-4 Customer angry / escalation        | `BaseCallAgent.customer_angry` → `EscalationAgent`; 2nd event auto-transfers      |
| EC-5 Dispute                            | `DisputeAgent` (parallel review + undisputed-partial offer)                       |
| EC-6 Repeated refusal                   | Stage 3 in-place refusal ladder → `hard_refusal` close                            |
| EC-7 Technical issues (ASR/TTS)         | Each stage has an `unclear` tool that re-poses the same question                  |
| Retry cadence + script rotation         | Out-of-call: downstream worker subscribes to `OUTCOME` stream                     |
| PTP enforcement loop (T-24h/T-2h/T+0)   | Out-of-call: same worker reads `commitment` + `close_ok`                          |
| Compliance pre-TTS filter               | `_najdi_normalize` (TTS) + per-stage prompt rules                                 |

---

## 8. Job metadata contract

`dispatch.py` sends JSON metadata to the agent. Recognised keys:

```json
{
  "phone_number":      "+9665...",       // required, E.164
  "name":              "محمد",            // optional, default "محمد"
  "amount":            "10000",          // optional, SAR string, default "10000"
  "debt_date":         "2023-01-01",     // optional ISO, default "2023-01-01"
  "national_id_last4": "1234",           // optional, default "1234"
  "dob":               "1990-05-12"      // optional ISO, default "1990-01-01"
}
```

`phone_number` is also propagated into `CallData.phone_number` so the
SMS hooks have a destination.

---

## 9. Latency-preserving choices

These are decisions worth keeping in mind before any future refactor:

1. **Sub-agent pattern over `TaskGroup`** — see §1. Keep stage
   transitions cheap (no summarisation LLM call).
2. **Identical PERSONA + `call_context_block` prefix across stages** —
   keeps GPT-4.1's automatic prompt-cache prefix hot. Don't reorder
   per-stage instructions before this prefix.
3. **In-place refusal ladder** — `refuses_payment` reuses the same
   agent and just emits a follow-up reply. Each ladder step costs one
   LLM turn, not two.
4. **Side effects after `wait_for_playout`** — `ClosingAgent` waits
   for the goodbye audio to finish before SMS / log / hangup. SMS or
   logging latency never delays the customer-perceived close.
5. **Soniox STT, Faseeh streaming TTS, Silero VAD** — preserved.
6. **Dynamic endpointing 0.2–1.0 s** — preserved.
7. **Najdi normalisation in `tts_node`** — runs as the LLM streams,
   not after, so no buffering overhead.

---

## 10. Where to wire real integrations

Search for the hook names in `smart_agent.py`:

| Stub                          | Replace with                                            |
| ----------------------------- | ------------------------------------------------------- |
| `_emit_outcome`               | structured logger / Kafka / webhook                     |
| `_send_sms_payment_link`      | SMS provider that sends `amount` + payment URL / IBAN   |
| `_send_sms_callback_card`     | SMS provider that sends a callback CTA                  |
| `_request_human_transfer`     | LiveKit SIP REFER / ACD bridge                          |

Each stub already has the data it needs on `CallData` — no signature
changes required.
