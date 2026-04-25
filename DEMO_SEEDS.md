# Demo seeds (for video + before/after story)

These seeds are intended to produce **repeatable** episodes for demos. Use them in `/reset?seed=...`.

## Seed A — “Fail” story (attacker reaches Auth early)
- **Seed**: `14`
- **Expected behavior**: attacker progresses to `auth_server` within ~8 steps. If the agent hesitates or fails to isolate, things escalate quickly.
- **Best use**: show an untrained agent getting overwhelmed by alerts and losing time.

## Seed B — “Deception” story (API screams, but attacker is at Auth)
- **Seed**: `108`
- **Expected behavior**: within ~8 steps, attacker is at/near `auth_server` while many high-severity alerts appear on `api_gateway` (strong decoy).
- **Best use**: show a naive agent chasing the noisy node, while a trained agent focuses on real evidence.

## Seed C — “Pressure” story (Finance blocks without citations)
- **Seed**: `23`
- **Expected behavior**: frequent high-severity alerts; attacker tends to progress toward `auth_server` by ~8 steps. Finance demands justification on isolation actions.
- **Best use**: show the debate mechanic: bad argument → block/delay; evidence + coverage → approve.

Notes:
- These are chosen via a simple deterministic search over seeds 1–220.
- If world logic changes later, rerun the search and update this file.

