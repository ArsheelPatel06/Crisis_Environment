"""
Cyber Crisis Simulator — GUI Dashboard
=======================================
Live Plotly Dash app showing model performance trends across policies,
scenarios, and training steps.

Run:
    python dashboard.py

Then open:  http://localhost:8050
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "results" / "benchmark_results.csv"
TRAINING_CSV = PROJECT_ROOT / "results" / "training_log.csv"

POLICY_COLORS = {
    "random":    "#e05252",   # red
    "heuristic": "#e09f3e",   # amber
    "trained":   "#3ec97d",   # green
}

SCENARIO_ORDER = ["easy", "medium", "hard"]
POLICY_ORDER   = ["random", "heuristic", "trained"]
TASK_LABELS    = {
    "alert_triage":         "Alert Triage",
    "stakeholder_argument": "Stakeholder Debate",
    "full_crisis_episode":  "Full Crisis Episode",
}

app = Dash(__name__, title="Cyber Crisis — Model Dashboard")

# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_benchmark() -> pd.DataFrame:
    if not CSV_PATH.exists():
        return pd.DataFrame(columns=[
            "scenario", "policy", "task_id", "seed", "step", "reward",
            "trust_Finance", "trust_Engineering", "trust_PR",
            "poisoned_count", "agent_reason", "done",
        ])
    df = pd.read_csv(CSV_PATH)
    df["task_label"] = df["task_id"].map(TASK_LABELS).fillna(df["task_id"])
    df["policy"] = pd.Categorical(df["policy"], categories=POLICY_ORDER, ordered=True)
    df["scenario"] = pd.Categorical(df["scenario"], categories=SCENARIO_ORDER, ordered=True)
    return df


def load_training() -> pd.DataFrame:
    if not TRAINING_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(TRAINING_CSV)


# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────

_CARD = {
    "backgroundColor": "#1e2130",
    "borderRadius": "10px",
    "padding": "16px",
    "marginBottom": "16px",
}

_LABEL = {"color": "#aab0c6", "fontSize": "12px", "marginBottom": "6px"}

app.layout = html.Div(
    style={"backgroundColor": "#141722", "minHeight": "100vh", "fontFamily": "Inter, sans-serif", "padding": "24px"},
    children=[
        # ── Header ────────────────────────────────────────────────────────────
        html.Div([
            html.H1(
                "Cyber Crisis Simulator — Model Dashboard",
                style={"color": "#ffffff", "margin": "0 0 4px 0", "fontSize": "22px"},
            ),
            html.P(
                "Live benchmark: 3 policies × 3 tasks × 20 seeds × 3 difficulty scenarios",
                style={"color": "#8890a8", "margin": "0", "fontSize": "13px"},
            ),
        ], style={"marginBottom": "20px"}),

        # ── Controls ──────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.P("Scenario", style=_LABEL),
                dcc.Dropdown(
                    id="scenario-filter",
                    options=[{"label": s.title(), "value": s} for s in SCENARIO_ORDER] + [{"label": "All", "value": "all"}],
                    value="all",
                    clearable=False,
                    style={"backgroundColor": "#252a3d", "color": "#fff", "border": "none"},
                ),
            ], style={"flex": "1", "marginRight": "12px"}),
            html.Div([
                html.P("Task", style=_LABEL),
                dcc.Dropdown(
                    id="task-filter",
                    options=[{"label": v, "value": k} for k, v in TASK_LABELS.items()] + [{"label": "All Tasks", "value": "all"}],
                    value="all",
                    clearable=False,
                    style={"backgroundColor": "#252a3d", "color": "#fff", "border": "none"},
                ),
            ], style={"flex": "1", "marginRight": "12px"}),
            html.Div([
                html.P("Policies", style=_LABEL),
                dcc.Checklist(
                    id="policy-filter",
                    options=[{"label": f"  {p.title()}", "value": p} for p in POLICY_ORDER],
                    value=POLICY_ORDER,
                    inline=True,
                    labelStyle={"color": "#ccd0e0", "marginRight": "14px", "fontSize": "13px"},
                ),
            ], style={"flex": "2", "paddingTop": "4px"}),
            html.Div([
                html.Button(
                    "Reload Data",
                    id="reload-btn",
                    n_clicks=0,
                    style={
                        "backgroundColor": "#3a7bd5", "color": "#fff",
                        "border": "none", "borderRadius": "6px",
                        "padding": "8px 18px", "cursor": "pointer",
                        "marginTop": "16px", "fontSize": "13px",
                    },
                ),
            ], style={"flex": "0 0 auto"}),
        ], style={"display": "flex", "alignItems": "flex-start", "marginBottom": "20px", **_CARD}),

        # ── Row 1: Policy bar + Reward trend ──────────────────────────────────
        html.Div([
            html.Div([
                html.P("Policy Comparison — Mean Reward", style={**_LABEL, "fontSize": "14px", "color": "#e0e4f0"}),
                dcc.Graph(id="policy-bar", style={"height": "300px"}, config={"displayModeBar": False}),
            ], style={**_CARD, "flex": "1", "marginRight": "12px"}),

            html.Div([
                html.P("Reward per Step — How Quickly Each Policy Improves", style={**_LABEL, "fontSize": "14px", "color": "#e0e4f0"}),
                dcc.Graph(id="reward-trend", style={"height": "300px"}, config={"displayModeBar": False}),
            ], style={**_CARD, "flex": "2"}),
        ], style={"display": "flex"}),

        # ── Row 2: Trust decay + Poisoning timeline ────────────────────────────
        html.Div([
            html.Div([
                html.P("Stakeholder Trust Decay Over Steps", style={**_LABEL, "fontSize": "14px", "color": "#e0e4f0"}),
                dcc.Graph(id="trust-lines", style={"height": "300px"}, config={"displayModeBar": False}),
            ], style={**_CARD, "flex": "2", "marginRight": "12px"}),

            html.Div([
                html.P("Poisoning Events per Step", style={**_LABEL, "fontSize": "14px", "color": "#e0e4f0"}),
                dcc.Graph(id="poison-bar", style={"height": "300px"}, config={"displayModeBar": False}),
            ], style={**_CARD, "flex": "1"}),
        ], style={"display": "flex"}),

        # ── Row 3: Steps-to-contain + Training curves ──────────────────────────
        html.Div([
            html.Div([
                html.P("Episode Length Distribution — Fewer Steps = Faster Containment", style={**_LABEL, "fontSize": "14px", "color": "#e0e4f0"}),
                dcc.Graph(id="steps-hist", style={"height": "300px"}, config={"displayModeBar": False}),
            ], style={**_CARD, "flex": "1", "marginRight": "12px"}),

            html.Div([
                html.P("GRPO Training Curves — Reward & Grad Norm over 60 Steps", style={**_LABEL, "fontSize": "14px", "color": "#e0e4f0"}),
                dcc.Graph(id="training-curves", style={"height": "300px"}, config={"displayModeBar": False}),
            ], style={**_CARD, "flex": "2"}),
        ], style={"display": "flex"}),

        # ── Scoreboard table ───────────────────────────────────────────────────
        html.Div([
            html.P("Scoreboard — Mean Reward by Scenario × Policy × Task", style={**_LABEL, "fontSize": "14px", "color": "#e0e4f0"}),
            html.Div(id="scoreboard-table"),
        ], style=_CARD),

        dcc.Store(id="data-store"),
    ],
)


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

def _filter_df(
    df: pd.DataFrame,
    scenario: str,
    task: str,
    policies: list[str],
) -> pd.DataFrame:
    if scenario != "all":
        df = df[df["scenario"] == scenario]
    if task != "all":
        df = df[df["task_id"] == task]
    if policies:
        df = df[df["policy"].isin(policies)]
    return df


def _dark_layout(title: str = "") -> dict:
    return dict(
        paper_bgcolor="#1e2130",
        plot_bgcolor="#141722",
        font=dict(color="#ccd0e0", size=11),
        title=dict(text=title, font=dict(size=13, color="#e0e4f0")),
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )


@app.callback(
    Output("policy-bar", "figure"),
    Output("reward-trend", "figure"),
    Output("trust-lines", "figure"),
    Output("poison-bar", "figure"),
    Output("steps-hist", "figure"),
    Output("training-curves", "figure"),
    Output("scoreboard-table", "children"),
    Input("scenario-filter", "value"),
    Input("task-filter", "value"),
    Input("policy-filter", "value"),
    Input("reload-btn", "n_clicks"),
)
def update_all(scenario, task, policies, _n):
    df = load_benchmark()
    training_df = load_training()

    # ── Panel 1: Policy bar ────────────────────────────────────────────────
    fdf = _filter_df(df, scenario, task, policies)
    if fdf.empty:
        bar_fig = go.Figure().update_layout(**_dark_layout("No data"))
    else:
        agg = (
            fdf.groupby(["policy", "scenario"], observed=True)["reward"]
            .mean()
            .reset_index()
            .sort_values(["scenario", "policy"])
        )
        bar_fig = px.bar(
            agg, x="policy", y="reward", color="scenario",
            barmode="group",
            color_discrete_sequence=["#3ec97d", "#e09f3e", "#e05252"],
            labels={"reward": "Mean Reward", "policy": "Policy", "scenario": "Scenario"},
            category_orders={"policy": POLICY_ORDER, "scenario": SCENARIO_ORDER},
        )
        bar_fig.update_layout(**_dark_layout())
        bar_fig.update_yaxes(range=[0, 1], gridcolor="#252a3d")
        bar_fig.update_xaxes(gridcolor="#252a3d")
        # Add baseline line
        bar_fig.add_hline(y=0.41, line_dash="dash", line_color="#e05252",
                          annotation_text="Random baseline 0.41", annotation_font_size=10)

    # ── Panel 2: Reward trend ──────────────────────────────────────────────
    if fdf.empty:
        trend_fig = go.Figure().update_layout(**_dark_layout("No data"))
    else:
        trend = (
            fdf.groupby(["policy", "step"], observed=True)["reward"]
            .mean()
            .reset_index()
        )
        trend_fig = px.line(
            trend, x="step", y="reward", color="policy",
            color_discrete_map=POLICY_COLORS,
            labels={"reward": "Mean Reward", "step": "Step", "policy": "Policy"},
            category_orders={"policy": POLICY_ORDER},
        )
        trend_fig.update_traces(line_width=2)
        trend_fig.update_layout(**_dark_layout())
        trend_fig.update_yaxes(range=[0, 1.05], gridcolor="#252a3d")
        trend_fig.update_xaxes(gridcolor="#252a3d")
        trend_fig.add_hline(y=0.41, line_dash="dot", line_color="#e05252",
                            annotation_text="Random baseline", annotation_font_size=10)

    # ── Panel 3: Trust score decay ─────────────────────────────────────────
    trust_df = _filter_df(df[df["task_id"] == "full_crisis_episode"], scenario, "full_crisis_episode", policies)
    if trust_df.empty:
        trust_fig = go.Figure().update_layout(**_dark_layout("No data"))
    else:
        trust_melt = trust_df.groupby(["policy", "step"], observed=True)[
            ["trust_Finance", "trust_Engineering", "trust_PR"]
        ].mean().reset_index().melt(
            id_vars=["policy", "step"],
            value_vars=["trust_Finance", "trust_Engineering", "trust_PR"],
            var_name="stakeholder", value_name="trust",
        )
        trust_melt["stakeholder"] = trust_melt["stakeholder"].str.replace("trust_", "")
        trust_fig = px.line(
            trust_melt, x="step", y="trust",
            color="stakeholder", line_dash="policy",
            labels={"trust": "Trust Score", "step": "Step", "stakeholder": "Stakeholder"},
            color_discrete_sequence=["#3a7bd5", "#3ec97d", "#e09f3e"],
        )
        trust_fig.update_traces(line_width=2)
        trust_fig.update_layout(**_dark_layout())
        trust_fig.update_yaxes(range=[0, 1.05], gridcolor="#252a3d")
        trust_fig.update_xaxes(gridcolor="#252a3d")
        trust_fig.add_hline(y=0.7, line_dash="dot", line_color="#555a70",
                            annotation_text="Initial trust 0.7", annotation_font_size=10)

    # ── Panel 4: Poisoning events ──────────────────────────────────────────
    if fdf.empty:
        poison_fig = go.Figure().update_layout(**_dark_layout("No data"))
    else:
        poison_agg = (
            fdf.groupby(["policy", "step"], observed=True)["poisoned_count"]
            .mean()
            .reset_index()
        )
        poison_fig = px.bar(
            poison_agg, x="step", y="poisoned_count", color="policy",
            barmode="group",
            color_discrete_map=POLICY_COLORS,
            labels={"poisoned_count": "Avg Poisoned Stakeholders", "step": "Step", "policy": "Policy"},
            category_orders={"policy": POLICY_ORDER},
        )
        poison_fig.update_layout(**_dark_layout())
        poison_fig.update_yaxes(gridcolor="#252a3d")
        poison_fig.update_xaxes(gridcolor="#252a3d")

    # ── Panel 5: Steps-to-contain histogram ───────────────────────────────
    if fdf.empty:
        steps_fig = go.Figure().update_layout(**_dark_layout("No data"))
    else:
        last_step = fdf.groupby(["policy", "seed", "task_id", "scenario"], observed=True)["step"].max().reset_index()
        last_step.rename(columns={"step": "episode_length"}, inplace=True)
        steps_fig = px.histogram(
            last_step, x="episode_length", color="policy",
            barmode="overlay", opacity=0.75,
            color_discrete_map=POLICY_COLORS,
            nbins=20,
            labels={"episode_length": "Episode Length (steps)", "policy": "Policy"},
            category_orders={"policy": POLICY_ORDER},
        )
        steps_fig.update_layout(**_dark_layout())
        steps_fig.update_yaxes(gridcolor="#252a3d")
        steps_fig.update_xaxes(gridcolor="#252a3d")

    # ── Panel 6: GRPO training curves ─────────────────────────────────────
    if training_df.empty:
        train_fig = go.Figure().update_layout(**_dark_layout("No training data found"))
    else:
        train_fig = go.Figure()
        has_grad = training_df["grad_norm"].max() > 0

        train_fig.add_trace(go.Scatter(
            x=training_df["step"], y=training_df["reward"],
            name="Reward", line=dict(color="#3ec97d", width=2),
            mode="lines+markers", marker=dict(size=4),
        ))
        if has_grad:
            real_steps = training_df[training_df["grad_norm"] > 0]
            train_fig.add_trace(go.Scatter(
                x=real_steps["step"], y=real_steps["reward"],
                name="Real gradient update", mode="markers",
                marker=dict(color="#3a7bd5", size=8, symbol="star"),
            ))
        if "entropy" in training_df.columns:
            train_fig.add_trace(go.Scatter(
                x=training_df["step"], y=training_df["entropy"] / training_df["entropy"].max(),
                name="Entropy (norm.)", line=dict(color="#e09f3e", width=1.5, dash="dot"),
                mode="lines",
            ))
        train_fig.add_hline(y=0.41, line_dash="dash", line_color="#e05252",
                            annotation_text="Random baseline 0.41", annotation_font_size=10)
        train_fig.update_layout(
            **_dark_layout("GRPO Training (60 steps, 20 seeds, num_gen=4)"),
            yaxis_title="Reward / Entropy (norm.)",
            xaxis_title="Training Step",
        )
        train_fig.update_yaxes(range=[0, 1.05], gridcolor="#252a3d")
        train_fig.update_xaxes(gridcolor="#252a3d")

    # ── Scoreboard table ───────────────────────────────────────────────────
    if df.empty:
        table = html.P("No data — run benchmark_runner.py first.", style={"color": "#8890a8"})
    else:
        scoreboard = (
            df.groupby(["scenario", "policy", "task_label"], observed=True)["reward"]
            .mean()
            .reset_index()
            .sort_values(["scenario", "policy", "task_label"])
        )
        scoreboard["reward"] = scoreboard["reward"].map(lambda x: f"{x:.4f}")

        header_style = {
            "backgroundColor": "#252a3d", "color": "#aab0c6",
            "padding": "8px 12px", "textAlign": "left", "fontSize": "12px",
        }
        cell_style = {
            "padding": "7px 12px", "color": "#ccd0e0",
            "fontSize": "12px", "borderBottom": "1px solid #252a3d",
        }

        def policy_color_style(p: str) -> dict:
            return {**cell_style, "color": POLICY_COLORS.get(p, "#ccd0e0"), "fontWeight": "600"}

        rows_html = []
        prev_scenario = None
        for _, row in scoreboard.iterrows():
            bg = "#1a1e2e" if row["scenario"] != prev_scenario else "#1e2130"
            prev_scenario = row["scenario"]
            rows_html.append(html.Tr([
                html.Td(row["scenario"].title(), style={**cell_style, "backgroundColor": bg}),
                html.Td(row["policy"].title(), style={**policy_color_style(row["policy"]), "backgroundColor": bg}),
                html.Td(row["task_label"], style={**cell_style, "backgroundColor": bg}),
                html.Td(row["reward"], style={**cell_style, "backgroundColor": bg, "fontFamily": "monospace"}),
            ]))

        table = html.Table([
            html.Thead(html.Tr([
                html.Th("Scenario", style=header_style),
                html.Th("Policy", style=header_style),
                html.Th("Task", style=header_style),
                html.Th("Mean Reward", style=header_style),
            ])),
            html.Tbody(rows_html),
        ], style={"width": "100%", "borderCollapse": "collapse"})

    return bar_fig, trend_fig, trust_fig, poison_fig, steps_fig, train_fig, table


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import webbrowser
    import threading

    port = 8050

    if not CSV_PATH.exists():
        print("No benchmark data found. Running benchmark first...")
        from benchmark_runner import run_benchmark, write_csv, SCENARIOS, TASK_IDS, SEEDS
        rows = run_benchmark(
            scenarios=list(SCENARIOS.keys()),
            policies=["random", "heuristic", "trained"],
            task_ids=TASK_IDS,
            seeds=SEEDS[:10],
            verbose=True,
        )
        write_csv(rows, CSV_PATH)

    print(f"\n  Dashboard running at  http://localhost:{port}")
    print("  Press Ctrl+C to stop.\n")

    def _open_browser():
        import time
        time.sleep(1.2)
        webbrowser.open(f"http://localhost:{port}")

    threading.Thread(target=_open_browser, daemon=True).start()
    app.run(debug=False, port=port, host="0.0.0.0")
