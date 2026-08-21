"""Audit exactly what tau2 puts on the wire for the agent.

Run under .venv-tau2 (the tau2 editable install):

    .venv-tau2/bin/python -m tau2probe.wire_audit <results.json> [--sim-index N]

Takes a saved simulation, rebuilds the agent-visible history the way tau2's own
agent does (system prompt + `is_valid_agent_history_message` filter), pushes it
through the real `to_litellm_messages`, and reports:

  * structural checks -- every tool result carries role:"tool" + a tool_call_id
    that matches a prior assistant tool_call; no reasoning fields anywhere;
  * the two anomaly shapes worth testing against backends: `content: null` on
    assistant tool-call turns, the duplicate top-level "name" key inside each
    tool_call, and explicit `tool_calls: null` on text turns;
  * dumps the payload to JSON so a human can read what the model saw.

This audits OUR side of the wire. What NextBit's chat template does with it is
unobservable from here -- that is what the replay probe's variant D tests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tau2.agent.base_agent import is_valid_agent_history_message
from tau2.data_model.message import AssistantMessage, Message
from tau2.data_model.simulation import Results
from tau2.utils.llm_utils import to_litellm_messages


def agent_view(messages: list[Message]) -> list[Message]:
    """The message list the AGENT is conditioned on (tau2's own filter)."""
    return [m for m in messages if is_valid_agent_history_message(m)]


def audit_payload(payload: list[dict]) -> dict:
    issues: list[str] = []
    stats = {
        "messages": len(payload),
        "tool_results": 0,
        "assistant_tool_call_turns": 0,
        "null_content_tool_call_turns": 0,
        "explicit_null_tool_calls_field": 0,
        "duplicate_name_key": 0,
    }
    open_ids: set[str] = set()
    for i, msg in enumerate(payload):
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            stats["assistant_tool_call_turns"] += 1
            if msg.get("content") is None:
                stats["null_content_tool_call_turns"] += 1
            for tc in msg["tool_calls"]:
                open_ids.add(tc.get("id"))
                if "name" in tc:  # non-standard duplicate of function.name
                    stats["duplicate_name_key"] += 1
                if tc.get("function", {}).get("name") != tc.get(
                    "name", tc.get("function", {}).get("name")
                ):
                    issues.append(f"msg {i}: top-level name != function.name")
        elif role == "assistant" and "tool_calls" in msg and msg["tool_calls"] is None:
            stats["explicit_null_tool_calls_field"] += 1
        if role == "tool":
            stats["tool_results"] += 1
            tcid = msg.get("tool_call_id")
            if not tcid:
                issues.append(f"msg {i}: tool result without tool_call_id")
            elif tcid not in open_ids:
                issues.append(f"msg {i}: tool_call_id {tcid!r} matches no prior call")
            if msg.get("content") in (None, ""):
                issues.append(f"msg {i}: tool result with empty content")
        for bad in ("reasoning", "reasoning_content", "thinking", "signature"):
            if bad in msg:
                issues.append(f"msg {i}: unexpected field {bad!r}")
    return {"stats": stats, "issues": issues}


def normalize_payload(payload: list[dict]) -> list[dict]:
    """Variant-D payload: same content, standard-conformant shape.

    - assistant tool-call turns get content "" instead of null
    - the duplicate top-level "name" key is dropped from tool_calls
    - explicit `"tool_calls": null` fields are omitted
    """
    out = []
    for msg in payload:
        msg = dict(msg)
        if msg.get("role") == "assistant":
            if msg.get("tool_calls"):
                if msg.get("content") is None:
                    msg["content"] = ""
                msg["tool_calls"] = [
                    {k: v for k, v in tc.items() if k != "name"}
                    for tc in msg["tool_calls"]
                ]
            elif "tool_calls" in msg and msg["tool_calls"] is None:
                del msg["tool_calls"]
        out.append(msg)
    return out


def load_sim(results_path: Path, index: int):
    results = Results.load(results_path)
    sim = results.simulations[index]
    task = next(t for t in results.tasks if t.id == sim.task_id)
    return results, sim, task


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path)
    ap.add_argument("--sim-index", type=int, default=0)
    ap.add_argument("--dump", type=Path, help="write the payload JSON here")
    args = ap.parse_args()

    _, sim, task = load_sim(args.results, args.sim_index)
    view = agent_view(sim.messages)
    payload = to_litellm_messages(view)
    report = audit_payload(payload)

    print(f"simulation {sim.id}  task {sim.task_id}  trial {sim.trial}")
    print(f"termination {sim.termination_reason}  agent-visible messages {len(view)}")
    for k, v in report["stats"].items():
        print(f"  {k}: {v}")
    if report["issues"]:
        print("ISSUES:")
        for issue in report["issues"]:
            print(f"  - {issue}")
    else:
        print("no structural issues: every tool result is role:'tool' with a "
              "matching tool_call_id; no reasoning fields.")
    if args.dump:
        args.dump.write_text(json.dumps(payload, indent=1))
        print(f"payload -> {args.dump}")


if __name__ == "__main__":
    main()
