"""LLM response contract: prompts, JSON schema, and safe parsing into server Action models."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from server.models import Action, TaskId

# --- Public schema (mirrors server.models.Action) ---

ACTION_TYPE_ENUM = "isolate|patch|monitor|ignore|communicate|noop"

ACTION_JSON_SCHEMA = """{
  "action_type": "(%s)",
  "target": "<optional node name string, e.g. auth_server>",
  "classifications": { "<alert_id>": "real" | "fake" },
  "argument_text": "<only for communicate when debate is open>",
  "citations": [ "<alert_id strings>" ]
}""" % ACTION_TYPE_ENUM

SYSTEM_MESSAGE = f"""You are a cyber incident command agent. You MUST respond with exactly one JSON object and nothing else.
No markdown fences unless you wrap raw JSON. Allowed keys: action_type, target, classifications, argument_text, citations.
{ACTION_JSON_SCHEMA}
Rules:
- For task_id alert_triage: include classifications for every alert id, values real or fake.
- For stakeholder_argument: if pending_debate is null, use isolate, patch, or monitor to build evidence. If pending_debate is not null, use communicate and fill argument_text plus citations to real alert ids.
- For full_crisis_episode: use node targets on infrastructure actions. If a debate is pending, prefer communicate to resolve.
"""

REPAIR_PREFIX = "Your previous output was invalid. Return only valid JSON matching the schema, fixing errors:"


@dataclass(frozen=True)
class TurnContext:
    task_id: TaskId
    observation: Dict[str, Any]
    last_error: Optional[str] = None


def _compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def build_user_message(ctx: TurnContext) -> str:
    obs = ctx.observation
    block = {
        "task_id": ctx.task_id,
        "observation": obs,
        "strict_action_schema": ACTION_JSON_SCHEMA,
    }
    if ctx.last_error:
        return REPAIR_PREFIX + " " + ctx.last_error + "\ncontext=" + _compact_json(block)
    return (
        "Choose the next action for this observation.\ncontext=" + _compact_json(block)
    )


def build_chat_messages(
    task_id: TaskId, observation: Dict[str, Any]
) -> List[Dict[str, str]]:
    """Messages suitable for SFT/TRL chat training."""
    ctx = TurnContext(task_id=task_id, observation=dict(observation))
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": build_user_message(ctx)},
    ]


def extract_json_object(text: str) -> str:
    """Strip optional markdown fences; return the first top-level JSON object as string."""
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.DOTALL)
    if m:
        return m.group(1)
    rbrace = t.rfind("}")
    lbrace = t.find("{")
    if lbrace >= 0 and rbrace > lbrace:
        return t[lbrace : rbrace + 1]
    if t.startswith("{"):
        return t
    raise ValueError("no_json_object_in_text")


def _fallback_action() -> Action:
    return Action(action_type="noop", target=None)


def parse_text_to_action(text: str) -> Action:
    raw = extract_json_object(text)
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise TypeError("json_root_not_object")
    return Action.model_validate(data)


def parse_with_repair(assistant_text: str, repair_text: str | None) -> Tuple[Action, Optional[str]]:
    """
    Returns (action, error_string_if_used_repair).
    Tries once on assistant_text, then on repair_text if given.
    """
    try:
        return parse_text_to_action(assistant_text), None
    except (ValueError, json.JSONDecodeError, ValidationError) as exc:
        first = str(exc)
    if not repair_text:
        return _fallback_action(), first
    try:
        return parse_text_to_action(repair_text), first
    except (ValueError, json.JSONDecodeError, ValidationError) as exc2:
        return _fallback_action(), first + " | " + str(exc2)


def action_to_json_str(action: Action) -> str:
    return _compact_json(action.model_dump(exclude_none=True, exclude_unset=True))
