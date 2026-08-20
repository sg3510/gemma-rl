# Training against a simulated user: the distribution problem

Our agent is trained against a **simulated** customer — Gemini 3.7 Flash at
temperature 0. That raises a question worth stating precisely, because the
answer changes what our result means.

## The distinction that matters: which tokens carry loss

In multi-turn training, only the **agent's** tokens contribute to the loss.
Customer turns and tool results are masked out — they are context, not targets.
So the simulator's phrasing is *conditioned on* and never *produced*; the policy
never learns to speak like Gemini.

That is why an out-of-distribution **customer** is a different kind of problem
from an out-of-distribution **agent** turn. A human-written agent turn *would*
sit in the loss, teaching the model to imitate phrasing it would never generate.

(The masking itself is not free — it is one of the fiddlier parts of multi-turn
RL, and verl ships documentation specifically on multiturn tokenization and
masking.)

## But the conditioning distribution is still baked in

Masking protects the *targets*, not the *inputs*. We are optimising a policy
conditioned on Gemini-3.7-Flash-shaped customers: their verbosity, their habit
of volunteering an order id unprompted, their particular way of signalling
satisfaction. A policy tuned to that may degrade against real humans or against
a different simulator.

So the honest statement of any result we obtain is not

> "Gemma got better at telecom support"

but

> **"Gemma got better at serving Gemini-3.7-Flash-shaped customers on telecom."**

That is a real scope limit, and a deeper reason to keep the customer pinned
(D6/D7) than comparability alone: the simulator is not just a measurement
instrument, it is part of what gets optimised into the weights.

## How practitioners actually handle it

| approach | what it is | our position |
|---|---|---|
| **Freeze and scope** | Fix one simulator, treat it as part of the environment definition, state it in the paper. What τ-bench itself does. | **doing this** (D6) |
| **Persona diversity** | Vary the simulated user so the policy cannot overfit one style. | **available free** — τ2's telecom pool is balanced across three personas (769 `None` / 760 `Easy` / 756 `Hard`), varying verbosity and interrupt tendency |
| **KL regularisation** | Penalise divergence from the reference policy. Usually framed as training stability, but it is also the main practical brake on drifting into any environment-specific quirk, simulator idiosyncrasies included. | **standard in GRPO; keep it on** |
| **Held-out simulator** | Train against simulator A, evaluate against simulator B. The most direct test of simulator overfitting. | **rarely done, cheap for us** — see below |
| **Human validation** | After training against a sim, validate on a small human-in-the-loop sample. | out of scope for a toy run |
| **Domain randomisation** | Randomise simulator parameters, inherited from sim2real robotics. Persona sampling is a weak form. | partially, via personas |

The first row is what most published work does, and it is defensible provided
the scope is stated. The failure mode is not using a simulator — it is
reporting the result as though the simulator were not there.

## What we will do

1. **Keep the customer pinned** for every arm of the before/after (D6, D20).
   Changing it mid-experiment would contaminate the delta.
2. **Sample personas in the training mix**, matching the pool's natural balance.
   Free diversity that makes the policy less able to exploit one speaking style.
3. **Add a simulator-transfer check.** After training, re-evaluate the same
   policy with a *different* customer model on the same tasks. If the gain
   survives, it is about telecom competence. If it evaporates, we learned to
   serve one simulator — which is a finding worth reporting, not a failure to
   hide. This is cheap for us (one extra eval pass) and seldom done.

## Related: where demonstrations should come from

The same logic ranks the ways to obtain expert trajectories for tasks the policy
never solves (D25):

| source | context distribution | target distribution | verdict |
|---|---|---|---|
| hand-written offline | wrong — the customer cannot be scripted | wrong | not viable |
| human play mode | correct (live customer) | human phrasing, out of distribution | usable, imperfect |
| **`LLMGTAgent` with Gemma as the `llm`** | correct | **the model's own phrasing** | **preferred** |

τ2's `LLMGTAgent` takes an `llm` argument and is handed the expected actions, so
pointing it at Gemma yields the model's own phrasing with only the *action
choices* supplied by the expert — self-generated, expert-guided data, in the
STaR/rationalization family.

Residual concern with that route: a guided model may express a **confidence it
would not otherwise have**, having been told the answer. That is a
rationalization artifact, and it is testable by comparing token distributions of
guided versus naturally successful traces.
