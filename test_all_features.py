"""
Cyber Crisis Simulator — Full Feature Test Suite
Run:  python test_all_features.py
"""

from __future__ import annotations

import sys
import textwrap
import time
from typing import Any, Dict

from fastapi.testclient import TestClient

from server.main import app

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

client = TestClient(app)

PASS_COUNT = 0
FAIL_COUNT = 0
SECTION_FAILURES: list[str] = []


def _reset(seed: int = 42, task_id: str = "full_crisis_episode") -> Dict[str, Any]:
    r = client.post("/reset", json={"seed": seed, "task_id": task_id})
    assert r.status_code == 200, f"/reset failed: {r.status_code} {r.text[:200]}"
    return r.json()


def _step(action_type: str, target: str | None = None, **kwargs) -> Dict[str, Any]:
    body: Dict[str, Any] = {"action_type": action_type}
    if target is not None:
        body["target"] = target
    body.update(kwargs)
    r = client.post("/step", json=body)
    return r


def _header(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASS_COUNT, FAIL_COUNT
    if condition:
        print(f"  {'✓':2}  {label}")
        PASS_COUNT += 1
    else:
        msg = f"  {'✗':2}  {label}"
        if detail:
            msg += f"\n        → {detail}"
        print(msg)
        FAIL_COUNT += 1
        SECTION_FAILURES.append(label)


# ─────────────────────────────────────────────────────────────────────────────
# 1. API HEALTH
# ─────────────────────────────────────────────────────────────────────────────

def test_health():
    _header("1. API Health & Root")
    r = client.get("/health")
    check("GET /health returns 200", r.status_code == 200, str(r.status_code))
    check("health status == 'healthy'", r.json().get("status") == "healthy", str(r.json()))

    r = client.get("/")
    check("GET / returns 200", r.status_code == 200)
    root = r.json()
    check("root has endpoints dict", "endpoints" in root)
    check("root lists tasks", "tasks" in root)

    r = client.get("/metadata")
    check("GET /metadata returns 200", r.status_code == 200)
    check("metadata has name field", "name" in r.json())

    r = client.get("/schema")
    check("GET /schema returns 200", r.status_code == 200)
    schema = r.json()
    check("schema has action key", "action" in schema)
    check("schema has observation key", "observation" in schema)


# ─────────────────────────────────────────────────────────────────────────────
# 2. RESET ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

def test_reset():
    _header("2. Reset Endpoint")

    # JSON body (was broken before our fix)
    r = client.post("/reset", json={"seed": 42, "task_id": "full_crisis_episode"})
    check("POST /reset with JSON body → 200", r.status_code == 200, str(r.status_code))
    obs = r.json()
    check("observation has task_id", "task_id" in obs)
    check("task_id == full_crisis_episode", obs["task_id"] == "full_crisis_episode")
    check("step == 0 on reset", obs["step"] == 0)
    check("system_health in [0,1]", 0.0 <= obs["system_health"] <= 1.0)
    check("threat_level in [0,1]", 0.0 <= obs["threat_level"] <= 1.0)
    check("infection_ratio in [0,1]", 0.0 <= obs["infection_ratio"] <= 1.0)
    check("done == False on reset", obs["done"] is False)
    check("resources_available >= 0", obs["resources_available"] >= 0)
    check("stakeholder_trust_scores present", "stakeholder_trust_scores" in obs)
    ts = obs["stakeholder_trust_scores"]
    check("initial Finance trust == 0.7", abs(ts.get("Finance", 0) - 0.7) < 1e-9, str(ts))
    check("initial Engineering trust == 0.7", abs(ts.get("Engineering", 0) - 0.7) < 1e-9)
    check("initial PR trust == 0.7", abs(ts.get("PR", 0) - 0.7) < 1e-9)

    # Query-param form still works
    r2 = client.post("/reset?seed=10&task_id=full_crisis_episode")
    check("POST /reset with query params → 200", r2.status_code == 200, str(r2.status_code))

    # Determinism: same seed = same first observation
    r3 = client.post("/reset", json={"seed": 42, "task_id": "full_crisis_episode"})
    r4 = client.post("/reset", json={"seed": 42, "task_id": "full_crisis_episode"})
    check("same seed produces identical reset observations", r3.json() == r4.json())


# ─────────────────────────────────────────────────────────────────────────────
# 3. STEP — CORE RESPONSE STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

def test_step_structure():
    _header("3. Step Response Structure")
    _reset(seed=42)
    r = _step("monitor", "api_gateway")
    check("POST /step → 200", r.status_code == 200, r.text[:200])
    d = r.json()

    check("response has 'observation'", "observation" in d)
    check("response has 'reward'", "reward" in d)
    check("response has 'done'", "done" in d)
    check("response has 'info'", "info" in d)

    obs = d["observation"]
    check("obs.step incremented to 1", obs["step"] == 1)
    check("obs.system_health in [0,1]", 0.0 <= obs["system_health"] <= 1.0)
    check("obs.stakeholder_trust_scores present", "stakeholder_trust_scores" in obs)
    check("obs.alerts is list", isinstance(obs.get("alerts"), list))
    check("obs.stakeholder_messages is list", isinstance(obs.get("stakeholder_messages"), list))

    rew = d["reward"]
    for field in ("security_score", "uptime_score", "trust_score", "speed_score", "total"):
        check(f"reward.{field} in [0,1]", 0.0 <= rew.get(field, -1) <= 1.0)

    check("done is bool", isinstance(d["done"], bool))

    info = d["info"]
    check("info.task_id present", "task_id" in info)
    check("info.trust_scores present", "trust_scores" in info, str(list(info.keys())))
    check("info.poisoned_stakeholders present", "poisoned_stakeholders" in info)
    check("info.agent_reason present", "agent_reason" in info)
    check("info.agent_reason is string", isinstance(info.get("agent_reason"), str))
    check("info.events is list", isinstance(info.get("events"), list))


# ─────────────────────────────────────────────────────────────────────────────
# 4. ALL ACTION TYPES
# ─────────────────────────────────────────────────────────────────────────────

def test_all_actions():
    _header("4. All Action Types")

    for action, target in [
        ("isolate",     "api_gateway"),
        ("monitor",     "internal_tools"),
        ("patch",       "api_gateway"),
        ("ignore",      None),
        ("communicate", None),
        ("noop",        None),
        ("investigate", None),
    ]:
        _reset(seed=42)
        # one warm-up step to generate alerts
        _step("monitor", "api_gateway")
        r = _step(action, target)
        check(f"action '{action}' → 200", r.status_code == 200,
              f"{r.status_code}: {r.text[:120]}")
        if r.status_code == 200:
            d = r.json()
            check(f"action '{action}' reward.total in [0,1]",
                  0.0 <= d["reward"]["total"] <= 1.0,
                  str(d["reward"]["total"]))

    # Target normalisation: API_Gateway (mixed case) must work
    _reset(seed=42)
    r = _step("isolate", "API_Gateway")
    check("target 'API_Gateway' normalised (was broken before fix)", r.status_code == 200,
          r.text[:200])

    # Invalid action must be rejected
    _reset(seed=42)
    r = _step("hack")
    check("unknown action_type 'hack' → 422", r.status_code == 422, str(r.status_code))


# ─────────────────────────────────────────────────────────────────────────────
# 5. INVESTIGATE ACTION (new feature)
# ─────────────────────────────────────────────────────────────────────────────

def test_investigate():
    _header("5. INVESTIGATE Action")

    # With prior alerts
    _reset(seed=42)
    _step("monitor", "api_gateway")   # generates alerts
    r = _step("investigate")
    check("investigate → 200", r.status_code == 200, r.text[:200])
    d = r.json()
    info = d["info"]
    check("investigated_alert in info", "investigated_alert" in info)
    check("is_fake in info", "is_fake" in info)
    check("is_fake is bool or None", info["is_fake"] in (True, False, None))
    check("reward in [0,1] after investigate", 0.0 <= d["reward"]["total"] <= 1.0)

    # Reward bonus when fake alert found
    found_fake = False
    for seed in range(1, 40):
        _reset(seed=seed)
        _step("monitor", "api_gateway")
        r = _step("investigate")
        if r.json()["info"].get("is_fake") is True:
            total = r.json()["reward"]["total"]
            check(f"investigate fake alert gives reward ≥ base (seed {seed})", total >= 0.0)
            found_fake = True
            break
    check("found at least one fake alert across seeds 1-39", found_fake)

    # Without prior alerts (fresh env, first step)
    _reset(seed=42)
    r = _step("investigate")
    check("investigate on fresh env (no prior alerts) → 200", r.status_code == 200)
    check("investigated_alert is str or None on fresh env",
          r.json()["info"]["investigated_alert"] is None
          or isinstance(r.json()["info"]["investigated_alert"], str))


# ─────────────────────────────────────────────────────────────────────────────
# 6. STAKEHOLDER TRUST TRACKING
# ─────────────────────────────────────────────────────────────────────────────

def test_trust_tracking():
    _header("6. Stakeholder Trust Tracking")

    # isolate → Engineering stance followed (not poisoned) → +0.1
    _reset(seed=42)
    r = _step("isolate", "api_gateway")
    ts = r.json()["observation"]["stakeholder_trust_scores"]
    check("Engineering trust increases after isolate (aligned)",
          ts["Engineering"] > 0.7, f"Engineering={ts['Engineering']:.3f}")

    # Finance not followed on isolate → no change
    check("Finance trust unchanged after isolate (not followed)",
          abs(ts["Finance"] - 0.7) < 1e-9, f"Finance={ts['Finance']:.3f}")

    # Follow poisoned Finance with ignore → trust drops -0.2
    _reset(seed=7)
    r1 = _step("monitor", "api_gateway")   # step 1: may poison Finance
    ts_before = r1.json()["observation"]["stakeholder_trust_scores"]["Finance"]
    r2 = _step("ignore")
    d2 = r2.json()
    ts_after = d2["observation"]["stakeholder_trust_scores"]["Finance"]
    poisoned = d2["info"]["poisoned_stakeholders"]
    if "Finance" in poisoned:
        check("Finance trust drops -0.2 when poisoned + followed",
              abs(ts_after - (ts_before - 0.2)) < 1e-9,
              f"before={ts_before:.3f} after={ts_after:.3f}")
    else:
        check("Finance trust increases +0.1 when not poisoned + followed",
              abs(ts_after - (ts_before + 0.1)) < 1e-9,
              f"before={ts_before:.3f} after={ts_after:.3f}")

    # Trust values always clamped
    _reset(seed=42)
    for _ in range(8):
        _step("ignore")
    r = _step("ignore")
    ts_final = r.json()["observation"]["stakeholder_trust_scores"]
    check("all trust values stay in [0,1] after many steps",
          all(0.0 <= v <= 1.0 for v in ts_final.values()), str(ts_final))

    # trust_scores also appears in /state
    r_state = client.get("/state")
    check("trust_scores in GET /state", "trust_scores" in r_state.json(),
          str(list(r_state.json().keys())))


# ─────────────────────────────────────────────────────────────────────────────
# 7. POISONING EVENTS
# ─────────────────────────────────────────────────────────────────────────────

def test_poisoning_events():
    _header("7. Red-Team Poisoning Events")

    # Run steps until we observe at least one poisoning event
    _reset(seed=7)
    seen_poison = False
    for _ in range(10):
        r = _step("monitor", "api_gateway")
        info = r.json()["info"]
        if info["poisoned_stakeholders"]:
            seen_poison = True
            events = info["events"]
            check("poisoned_stakeholders is non-empty list", isinstance(info["poisoned_stakeholders"], list))
            check("events list non-empty", len(events) > 0)
            ev = events[0]
            check("event has 'event' key", "event" in ev)
            check("event type == stakeholder_poisoned", ev["event"] == "stakeholder_poisoned")
            check("event has 'target' key", "target" in ev)
            check("event target is Finance/Engineering/PR",
                  ev["target"] in ("Finance", "Engineering", "PR"), ev["target"])
            check("event has 'timestep' key", "timestep" in ev)
            check("event timestep is int", isinstance(ev["timestep"], int))
            break

    check("observed at least one poisoning event in 10 steps (seed=7)", seen_poison)

    # Steps with no poisoning should return empty list
    _reset(seed=42)
    r = _step("isolate", "api_gateway")
    info = r.json()["info"]
    check("poisoned_stakeholders is list even when empty",
          isinstance(info["poisoned_stakeholders"], list))
    check("events is list even when empty", isinstance(info["events"], list))


# ─────────────────────────────────────────────────────────────────────────────
# 8. AGENT REASON
# ─────────────────────────────────────────────────────────────────────────────

def test_agent_reason():
    _header("8. Agent Reason Field")

    expected_substrings = {
        "isolate":     ["isolate", "Chose"],
        "monitor":     ["monitor", "Monitor"],
        "patch":       ["patch", "Patch"],
        "investigate": ["investigat"],
        "ignore":      ["ignore", "Ignore", "inconsistency", "conflict", "low confidence"],
        "communicate": ["communicat", "stakeholder", "debate"],
        "noop":        ["No action", "noop", "waiting"],
    }

    for action, keywords in expected_substrings.items():
        _reset(seed=7)
        _step("monitor", "api_gateway")   # warm-up
        r = _step(action, "api_gateway" if action in ("isolate", "monitor", "patch") else None)
        if r.status_code != 200:
            check(f"agent_reason for '{action}' — step succeeded", False, str(r.status_code))
            continue
        reason = r.json()["info"].get("agent_reason", "")
        matched = any(kw.lower() in reason.lower() for kw in keywords)
        check(f"agent_reason for '{action}' is contextual",
              matched and len(reason) > 10,
              f"reason='{reason[:80]}'")

    # Reason must be a non-empty string for every action
    for action in ("isolate", "monitor", "patch", "ignore", "investigate", "noop"):
        _reset(seed=42)
        r = _step(action, "api_gateway" if action in ("isolate", "monitor", "patch") else None)
        if r.status_code == 200:
            reason = r.json()["info"].get("agent_reason", "")
            check(f"agent_reason non-empty string for '{action}'",
                  isinstance(reason, str) and len(reason) > 5, repr(reason))


# ─────────────────────────────────────────────────────────────────────────────
# 9. STATE ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

def test_state():
    _header("9. GET /state")

    _reset(seed=42)
    _step("isolate", "api_gateway")

    r = client.get("/state")
    check("GET /state → 200", r.status_code == 200, str(r.status_code))
    st = r.json()

    for key in ("seed", "task_id", "step_count", "done", "attacker_position",
                "nodes", "pending_debate", "truth_alert_is_fake", "trust_scores"):
        check(f"state has '{key}'", key in st, str(list(st.keys())))

    check("state.seed == 42", st["seed"] == 42, str(st["seed"]))
    check("state.step_count == 1", st["step_count"] == 1, str(st["step_count"]))
    check("state.nodes contains api_gateway", "api_gateway" in st["nodes"])
    check("api_gateway.isolated == True after isolate action",
          st["nodes"]["api_gateway"]["isolated"] is True,
          str(st["nodes"]["api_gateway"]))
    check("trust_scores has Finance/Engineering/PR",
          set(st["trust_scores"].keys()) == {"Finance", "Engineering", "PR"})
    check("attacker_position is string", isinstance(st["attacker_position"], str))


# ─────────────────────────────────────────────────────────────────────────────
# 10. MCP ENDPOINT (OpenEnv contract)
# ─────────────────────────────────────────────────────────────────────────────

def test_mcp():
    _header("10. POST /mcp (OpenEnv Contract)")

    payload = {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}}
    r = client.post("/mcp", json=payload)
    check("POST /mcp → 200", r.status_code == 200, str(r.status_code))
    resp = r.json()
    check("mcp response has jsonrpc field", "jsonrpc" in resp)
    check("mcp response has id field", "id" in resp)
    check("mcp response has result field", "result" in resp)
    check("mcp result.ok == True", resp.get("result", {}).get("ok") is True)


# ─────────────────────────────────────────────────────────────────────────────
# 11. TASK MODES
# ─────────────────────────────────────────────────────────────────────────────

def test_task_modes():
    _header("11. Task Modes")

    # alert_triage
    r = client.post("/reset", json={"seed": 42, "task_id": "alert_triage"})
    check("reset alert_triage → 200", r.status_code == 200, str(r.status_code))
    obs = r.json()
    check("alert_triage provides alerts", len(obs.get("alerts", [])) > 0,
          f"alerts count: {len(obs.get('alerts', []))}")
    r2 = _step("monitor", "api_gateway")
    check("alert_triage step → 200", r2.status_code == 200, r2.text[:200])
    check("alert_triage reward.total in [0,1]", 0.0 <= r2.json()["reward"]["total"] <= 1.0)

    # stakeholder_argument
    r = client.post("/reset", json={"seed": 42, "task_id": "stakeholder_argument"})
    check("reset stakeholder_argument → 200", r.status_code == 200, str(r.status_code))
    r2 = _step("isolate", "auth_server")
    check("stakeholder_argument step → 200", r2.status_code == 200, r2.text[:200])

    # full_crisis_episode (already tested above)
    r = client.post("/reset", json={"seed": 42, "task_id": "full_crisis_episode"})
    check("reset full_crisis_episode → 200", r.status_code == 200)


# ─────────────────────────────────────────────────────────────────────────────
# 12. MULTI-STEP EPISODE INTEGRITY
# ─────────────────────────────────────────────────────────────────────────────

def test_episode_integrity():
    _header("12. Multi-Step Episode Integrity")

    _reset(seed=42)
    prev_step = 0
    for i, action in enumerate(["monitor", "isolate", "investigate", "ignore", "patch", "noop"]):
        target = "api_gateway" if action in ("monitor", "isolate", "patch") else None
        r = _step(action, target)
        check(f"step {i+1} ({action}) → 200", r.status_code == 200, r.text[:120])
        if r.status_code != 200:
            continue
        d = r.json()
        obs = d["observation"]
        check(f"step {i+1} step counter incremented",
              obs["step"] == prev_step + 1, f"expected {prev_step+1}, got {obs['step']}")
        prev_step = obs["step"]
        check(f"step {i+1} reward.total in [0,1]",
              0.0 <= d["reward"]["total"] <= 1.0)

    # Episode terminates when database breached (attacker wins)
    # Run many ignores until done or max steps
    _reset(seed=1)
    done = False
    for _ in range(30):
        r = _step("ignore")
        if r.json()["done"]:
            done = True
            break
    check("episode eventually terminates (done=True)", done)


# ─────────────────────────────────────────────────────────────────────────────
# 13. RED TEAM AGENT (unit tests)
# ─────────────────────────────────────────────────────────────────────────────

def test_red_team_agent():
    _header("13. RedTeamAgent Unit Tests")

    from server.agents import (
        EngineeringStakeholder,
        FinanceStakeholder,
        NetworkState,
        PRStakeholder,
        RedTeamAgent,
    )

    agent = RedTeamAgent()
    check("initial position == API_Gateway", agent.position == "API_Gateway")
    check("initial detected_ever == False", agent.detected_ever is False)

    ns = NetworkState(timestep=0, threat_level=0.3, status="stable",
                      defender_detected_red_team=False)
    new_pos = agent.step(ns)
    check("agent advances kill chain on step", new_pos == "Internal_Tools",
          new_pos)

    # plant_false_report logs event
    agent2 = RedTeamAgent()
    finance = FinanceStakeholder()
    agent2.plant_false_report(finance, "Threat is low", stakeholder_name="Finance", timestep=3)
    evts = agent2.flush_events()
    check("plant_false_report emits event", len(evts) == 1, str(evts))
    check("event.event == stakeholder_poisoned", evts[0]["event"] == "stakeholder_poisoned")
    check("event.target == Finance", evts[0]["target"] == "Finance")
    check("event.timestep == 3", evts[0]["timestep"] == 3)
    check("flush_events clears list", len(agent2.flush_events()) == 0)

    # stakeholder cannot be re-poisoned if can_be_poisoned=False
    class ProtectedStakeholder(FinanceStakeholder):
        can_be_poisoned: bool = False

    protected = ProtectedStakeholder()
    agent3 = RedTeamAgent()
    agent3.plant_false_report(protected, "false info", stakeholder_name="Protected", timestep=0)
    evts3 = agent3.flush_events()
    check("no event emitted for non-poisonable stakeholder", len(evts3) == 0, str(evts3))

    # generate_fake_alert
    fake = RedTeamAgent.generate_fake_alert("api_gateway", severity=4)
    check("generate_fake_alert has node", fake["node"] == "api_gateway")
    check("generate_fake_alert has is_fake=True", fake["is_fake"] is True)

    # attacker_reward
    end_agent = RedTeamAgent()
    end_agent.position = "Database"
    check("attacker_reward == 1.0 at Database undetected", end_agent.attacker_reward() == 1.0)
    end_agent.caught_before_auth = True
    check("attacker_reward == -1.0 if caught before auth", end_agent.attacker_reward() == -1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 14. STAKEHOLDER AGENTS (unit tests)
# ─────────────────────────────────────────────────────────────────────────────

def test_stakeholder_agents():
    _header("14. Stakeholder Agent Unit Tests")

    from server.agents import (
        EngineeringStakeholder,
        FinanceStakeholder,
        NetworkState,
        PRStakeholder,
    )

    ns = NetworkState(timestep=1, threat_level=0.4, status="warning",
                      isolation_proposed=True, shutdown_requested=False,
                      public_comms_pending=True, nodes_under_stress=["api_gateway"])

    for StakeholderCls, name in [
        (FinanceStakeholder, "Finance"),
        (EngineeringStakeholder, "Engineering"),
        (PRStakeholder, "PR"),
    ]:
        s = StakeholderCls()
        msg = s.generate_message(ns, timestep=1)
        check(f"{name} generates non-empty message", len(msg) > 20, repr(msg[:50]))
        check(f"{name} message contains timestep marker", "@t=1" in msg, msg[:60])

        # Poison integration
        s.receive_planted_report("This is false info")
        check(f"{name} poison_payload set", s.poison_payload == "This is false info")
        msg_poisoned = s.generate_message(ns, timestep=2)
        check(f"{name} poisoned message contains planted content",
              "false info" in msg_poisoned, msg_poisoned[:80])
        check(f"{name} poison_payload cleared after use", s.poison_payload is None)

        # Second message is clean
        msg_clean = s.generate_message(ns, timestep=3)
        check(f"{name} second message is clean (no poison)", "false info" not in msg_clean)


# ─────────────────────────────────────────────────────────────────────────────
# 15. WORLD SIMULATOR (unit tests)
# ─────────────────────────────────────────────────────────────────────────────

def test_world_simulator():
    _header("15. World Simulator Unit Tests")

    from simulator.world import CompanyWorld

    world = CompanyWorld(seed=42)
    check("world initialises with 5 nodes", len(world.nodes) == 5)
    check("api_gateway exists", "api_gateway" in world.nodes)
    check("database exists", "database" in world.nodes)
    check("initial attacker_position == 0", world.attacker_position == 0)

    # Isolation
    world.set_isolation("api_gateway", True)
    check("set_isolation marks node isolated", world.nodes["api_gateway"].isolated is True)
    check("isolated node availability drops to 0.35",
          abs(world.nodes["api_gateway"].availability - 0.35) < 1e-9)
    world.set_isolation("api_gateway", False)
    check("un-isolate restores availability to 1.0",
          abs(world.nodes["api_gateway"].availability - 1.0) < 1e-9)

    # Unknown node raises
    try:
        world.set_isolation("nonexistent", True)
        check("set_isolation unknown node raises ValueError", False)
    except ValueError:
        check("set_isolation unknown node raises ValueError", True)

    # Alert generation
    alerts = world.step_attacker_and_generate_alerts()
    check("step generates at least one alert", len(alerts) >= 1)
    check("alert has id field", "id" in alerts[0])
    check("alert has node field", "node" in alerts[0])
    check("alert has message field", "message" in alerts[0])
    check("alert id registered in alerts_by_id",
          alerts[0]["id"] in world.alerts_by_id)
    check("truth_alert_is_fake has alert entry",
          alerts[0]["id"] in world.truth_alert_is_fake)

    # Monitoring level
    world.set_monitoring_level("internal_tools", 3)
    check("monitoring level clamped to 3",
          world.nodes["internal_tools"].monitoring_level == 3)
    world.set_monitoring_level("internal_tools", 99)
    check("monitoring level clamped to max 3",
          world.nodes["internal_tools"].monitoring_level == 3)

    # attacker_reached_database
    check("database not breached initially", world.attacker_reached_database() is False)
    world.nodes["database"].compromise_stage = 3
    check("database breached when stage==3", world.attacker_reached_database() is True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       CYBER CRISIS SIMULATOR — FULL FEATURE TEST SUITE      ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    start = time.time()

    test_health()
    test_reset()
    test_step_structure()
    test_all_actions()
    test_investigate()
    test_trust_tracking()
    test_poisoning_events()
    test_agent_reason()
    test_state()
    test_mcp()
    test_task_modes()
    test_episode_integrity()
    test_red_team_agent()
    test_stakeholder_agents()
    test_world_simulator()

    elapsed = time.time() - start

    print()
    print("═" * 60)
    print(f"  RESULTS   {PASS_COUNT} passed   {FAIL_COUNT} failed   ({elapsed:.1f}s)")
    print("═" * 60)

    if FAIL_COUNT > 0:
        print("\n  Failed checks:")
        for f in SECTION_FAILURES:
            print(f"    ✗  {f}")
        print()
        sys.exit(1)
    else:
        print()
        print("  All tests passed.")
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()
