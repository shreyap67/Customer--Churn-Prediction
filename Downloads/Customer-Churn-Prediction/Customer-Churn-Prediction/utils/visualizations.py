"""
Premium Plotly visualization components for the Churn Intelligence Platform.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional

# ── Design tokens ────────────────────────────────────────────────────────────
BG_DARK   = "rgba(10, 14, 26, 0)"
GRID_CLR  = "rgba(255,255,255,0.06)"
FONT_CLR  = "#E2E8F0"
ACCENT1   = "#4F8EF7"   # electric blue
ACCENT2   = "#A855F7"   # violet
ACCENT3   = "#06B6D4"   # cyan
ACCENT4   = "#F97316"   # amber
DANGER    = "#EF4444"
SUCCESS   = "#22C55E"

LAYOUT_BASE = dict(
    paper_bgcolor=BG_DARK,
    plot_bgcolor=BG_DARK,
    font=dict(family="DM Sans, sans-serif", color=FONT_CLR, size=13),
    margin=dict(l=24, r=24, t=40, b=24),
    xaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
    yaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
    legend=dict(
        bgcolor="rgba(15,20,40,0.7)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1,
    ),
)


def apply_base(fig):
    fig.update_layout(**LAYOUT_BASE)
    return fig


# ── 1. Churn Distribution Donut ───────────────────────────────────────────────
def churn_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["Churn"].value_counts()
    fig = go.Figure(go.Pie(
        labels=["Retained", "Churned"],
        values=[counts.get("No", 0), counts.get("Yes", 0)],
        hole=0.62,
        marker=dict(colors=[SUCCESS, DANGER],
                    line=dict(color="rgba(10,14,26,1)", width=3)),
        textfont=dict(size=14),
        hovertemplate="%{label}: %{value} customers (%{percent})<extra></extra>",
    ))
    churn_rate = counts.get("Yes", 0) / len(df) * 100
    fig.add_annotation(
        text=f"<b>{churn_rate:.1f}%</b><br><span style='font-size:11px'>Churn Rate</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=18, color=FONT_CLR)
    )
    fig.update_layout(**LAYOUT_BASE, title=dict(text="Churn Distribution", x=0.5))
    return fig


# ── 2. Monthly Charges Distribution ──────────────────────────────────────────
def charges_by_churn(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for label, color in [("No", SUCCESS), ("Yes", DANGER)]:
        subset = df[df["Churn"] == label]["MonthlyCharges"]
        fig.add_trace(go.Histogram(
            x=subset, name=f"{'Retained' if label=='No' else 'Churned'}",
            marker_color=color, opacity=0.72,
            xbins=dict(size=5),
            hovertemplate="Charge: $%{x}<br>Count: %{y}<extra></extra>",
        ))
    fig.update_layout(**LAYOUT_BASE, barmode="overlay",
                      title=dict(text="Monthly Charges by Churn Status", x=0.5),
                      xaxis_title="Monthly Charges ($)",
                      yaxis_title="Customer Count")
    return fig


# ── 3. Contract vs Churn Grouped Bar ─────────────────────────────────────────
def contract_churn_bar(df: pd.DataFrame) -> go.Figure:
    ct = df.groupby(["Contract", "Churn"]).size().unstack(fill_value=0).reset_index()
    fig = go.Figure()
    for col, color, label in [("No", SUCCESS, "Retained"), ("Yes", DANGER, "Churned")]:
        if col in ct.columns:
            fig.add_trace(go.Bar(
                x=ct["Contract"], y=ct[col], name=label,
                marker_color=color, opacity=0.85,
                hovertemplate="%{x}<br>%{y} customers<extra></extra>",
            ))
    fig.update_layout(**LAYOUT_BASE, barmode="group",
                      title=dict(text="Churn by Contract Type", x=0.5))
    return fig


# ── 4. Tenure Churn Heatmap / Line ───────────────────────────────────────────
def tenure_churn_line(df: pd.DataFrame) -> go.Figure:
    df2 = df.copy()
    df2["TenureBin"] = pd.cut(df2["tenure"], bins=range(0, 75, 6), right=False)
    agg = df2.groupby("TenureBin").agg(
        total=("Churn", "count"),
        churned=("Churn", lambda x: (x == "Yes").sum())
    ).reset_index()
    agg["rate"] = agg["churned"] / agg["total"] * 100
    agg["label"] = agg["TenureBin"].astype(str)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg["label"], y=agg["rate"],
        mode="lines+markers",
        line=dict(color=ACCENT1, width=3),
        marker=dict(size=8, color=ACCENT1, line=dict(width=2, color="white")),
        fill="tozeroy",
        fillcolor="rgba(79,142,247,0.12)",
        name="Churn Rate %",
        hovertemplate="Tenure: %{x}<br>Churn Rate: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(**LAYOUT_BASE,
                      title=dict(text="Churn Rate by Customer Tenure", x=0.5),
                      xaxis_title="Tenure (months)", yaxis_title="Churn Rate (%)")
    return fig


# ── 5. Feature Importance Bar ─────────────────────────────────────────────────
def feature_importance_chart(feature_importance: Dict[str, float]) -> go.Figure:
    items = sorted(feature_importance.items(), key=lambda x: x[1])[-15:]
    names, values = zip(*items)

    colors = [
        f"rgba(79,142,247,{0.4 + 0.6*(v/max(values))})"
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=list(values), y=list(names),
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(**LAYOUT_BASE,
                      title=dict(text="Feature Importance (Top 15)", x=0.5),
                      xaxis_title="Importance Score",
                      height=480)
    return fig


# ── 6. ROC Curve Comparison ───────────────────────────────────────────────────
def roc_comparison(results: Dict) -> go.Figure:
    colors = [ACCENT1, ACCENT2, ACCENT3, ACCENT4]
    fig = go.Figure()

    for (name, metrics), color in zip(results.items(), colors):
        if "fpr" in metrics:
            fig.add_trace(go.Scatter(
                x=metrics["fpr"], y=metrics["tpr"],
                name=f"{name} (AUC={metrics['roc_auc']:.1f}%)",
                mode="lines", line=dict(color=color, width=2.5),
                hovertemplate="FPR: %{x:.2f}<br>TPR: %{y:.2f}<extra></extra>",
            ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name="Random Baseline",
        mode="lines", line=dict(color="gray", dash="dash", width=1.5),
    ))
    fig.update_layout(**LAYOUT_BASE,
                      title=dict(text="ROC Curve Comparison", x=0.5),
                      xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate",
                      height=420)
    return fig


# ── 7. Confusion Matrix Heatmap ───────────────────────────────────────────────
def confusion_heatmap(cm: List[List[int]], model_name: str) -> go.Figure:
    z = np.array(cm)
    labels = ["Retained", "Churned"]
    total = z.sum()
    text = [[f"{v}<br>({v/total*100:.1f}%)" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        text=text, texttemplate="%{text}",
        colorscale=[[0, "rgba(22,33,66,1)"], [1, "rgba(79,142,247,0.9)"]],
        showscale=False,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(**LAYOUT_BASE,
                      title=dict(text=f"Confusion Matrix — {model_name}", x=0.5),
                      xaxis_title="Predicted",
                      yaxis_title="Actual",
                      height=340)
    return fig


# ── 8. Model Metrics Radar ────────────────────────────────────────────────────
def metrics_radar(results: Dict) -> go.Figure:
    categories = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    colors = [ACCENT1, ACCENT2, ACCENT3, ACCENT4]
    fig = go.Figure()

    for (name, metrics), color in zip(results.items(), colors):
        vals = [
            metrics.get("accuracy", 0),
            metrics.get("precision", 0),
            metrics.get("recall", 0),
            metrics.get("f1", 0),
            metrics.get("roc_auc", 0),
        ]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill="toself", fillcolor="rgba(79,142,247,0.10)",
            line=dict(color=color, width=2),
            name=name,
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        polar=dict(
            bgcolor=BG_DARK,
            radialaxis=dict(visible=True, range=[60, 100], gridcolor=GRID_CLR, color=FONT_CLR),
            angularaxis=dict(gridcolor=GRID_CLR, color=FONT_CLR),
        ),
        title=dict(text="Model Performance Radar", x=0.5),
        height=440,
    )
    return fig


# ── 9. Payment Method Pie ─────────────────────────────────────────────────────
def payment_churn_pie(df: pd.DataFrame) -> go.Figure:
    churned = df[df["Churn"] == "Yes"]["PaymentMethod"].value_counts()
    fig = go.Figure(go.Pie(
        labels=churned.index.tolist(),
        values=churned.values.tolist(),
        hole=0.5,
        marker=dict(colors=[ACCENT1, ACCENT2, ACCENT3, ACCENT4]),
        hovertemplate="%{label}: %{value} churned<extra></extra>",
    ))
    fig.update_layout(**LAYOUT_BASE, title=dict(text="Churned Customers by Payment Method", x=0.5))
    return fig


# ── 10. Risk Score Gauge ──────────────────────────────────────────────────────
def risk_gauge(probability: float) -> go.Figure:
    pct = probability * 100
    if pct < 30:
        color, label = SUCCESS, "LOW RISK"
    elif pct < 65:
        color, label = ACCENT4, "MEDIUM RISK"
    else:
        color, label = DANGER, "HIGH RISK"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number=dict(suffix="%", font=dict(size=36, color=color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color=FONT_CLR)),
            bar=dict(color=color, thickness=0.3),
            bgcolor="rgba(255,255,255,0.05)",
            bordercolor="rgba(255,255,255,0.1)",
            steps=[
                dict(range=[0, 30], color="rgba(34,197,94,0.12)"),
                dict(range=[30, 65], color="rgba(249,115,22,0.12)"),
                dict(range=[65, 100], color="rgba(239,68,68,0.12)"),
            ],
            threshold=dict(line=dict(color=color, width=4), thickness=0.75, value=pct),
        ),
        title=dict(text=f"<b>{label}</b>", font=dict(size=16, color=color)),
    ))
    fig.update_layout(**LAYOUT_BASE, height=280)
    return fig


# ── 11. Bulk Churn Score Distribution ────────────────────────────────────────
def bulk_score_histogram(probabilities: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=probabilities * 100,
        nbinsx=30,
        marker=dict(
            color=probabilities * 100,
            colorscale=[[0, SUCCESS], [0.5, ACCENT4], [1, DANGER]],
            showscale=True,
            colorbar=dict(title="Risk %", tickfont=dict(color=FONT_CLR)),
        ),
        hovertemplate="Risk Score: %{x:.0f}%<br>Customers: %{y}<extra></extra>",
    ))
    fig.update_layout(**LAYOUT_BASE,
                      title=dict(text="Customer Risk Score Distribution", x=0.5),
                      xaxis_title="Churn Probability (%)",
                      yaxis_title="Number of Customers")
    return fig
