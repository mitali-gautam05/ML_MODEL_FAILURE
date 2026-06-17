import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

API = "https://mitali1234-when-ml-fails-api.hf.space"

st.set_page_config(
    page_title="When ML Fails",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── THEME-AWARE CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
/* Danger card — works in both modes */
.danger-card {
    background: rgba(229, 62, 62, 0.1);
    border-left: 4px solid #e53e3e;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin-bottom: .5rem;
}
.danger-card h2 { margin: 0; color: #e53e3e; }
.danger-card p  { margin: 0; color: var(--text-color); opacity: 0.7; }

/* Success card */
.success-card {
    background: rgba(56, 161, 105, 0.1);
    border-left: 4px solid #38a169;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin-bottom: .5rem;
}
.success-card h2 { margin: 0; color: #38a169; }
.success-card p  { margin: 0; color: var(--text-color); opacity: 0.7; }

/* Finding box — replaces hardcoded dark bg */
.finding-box {
    background: rgba(108, 99, 255, 0.08);
    border: 1px solid rgba(108, 99, 255, 0.35);
    border-radius: 8px;
    padding: 1rem;
    margin: .5rem 0;
    color: var(--text-color);
    font-size: 0.92rem;
    line-height: 1.6;
}
.finding-box b {
    color: #7c6fff;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,0.15);
}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("When ML Fails")
    st.caption("A controlled study of ML failure modes on imbalanced data")
    st.divider()
    page = st.radio(
        "Navigate",
        ["The Accuracy Lie", "Model Comparison", "Failure Mode Explorer"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("Dataset: Credit Card Fraud")
    st.caption("284,807 transactions · 0.17% fraud")
    st.caption("7 models compared")

# ── PLOTLY THEME HELPER ────────────────────────────────────────────────
def plotly_layout(fig, **kwargs):
    """Apply theme-neutral layout to any plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", size=13),
        **kwargs
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.15)", zerolinecolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zerolinecolor="rgba(128,128,128,0.2)")
    return fig

# ── API HELPER ─────────────────────────────────────────────────────────
def fetch(endpoint):
    try:
        r = requests.get(f"{API}{endpoint}", timeout=5)
        r.raise_for_status()
        return r.json()
    except:
        st.error("API not reachable. Run: `uvicorn backend.main:app --reload`")
        return None

# ── COLORS ─────────────────────────────────────────────────────────────
C_RED    = "#e53e3e"
C_GREEN  = "#38a169"
C_PURPLE = "#6c63ff"
C_AMBER  = "#f6ad55"

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — The Accuracy Lie
# ══════════════════════════════════════════════════════════════════════
if page == "The Accuracy Lie":
    st.title("The Accuracy Lie")
    st.markdown(
        "A model that **never detects fraud** scores **99.83% accuracy**. "
        "Here's why that's dangerous."
    )
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dummy Classifier")
        st.markdown("*Strategy: always predict 'Not Fraud'*")
        st.markdown(
            '<div class="success-card">'
            '<h2>99.83%</h2>'
            '<p>Accuracy — looks amazing</p>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="danger-card">'
            '<h2>0%</h2>'
            '<p>Fraud Recall — caught ZERO fraud</p>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="danger-card">'
            '<h2>~50%</h2>'
            '<p>ROC-AUC — random guessing level</p>'
            '</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.subheader("Why this happens")
        fig = go.Figure(go.Pie(
            labels=["Not Fraud (99.83%)", "Fraud (0.17%)"],
            values=[284315, 492],
            hole=0.55,
            marker_colors=[C_PURPLE, C_RED],
            textinfo="label+percent",
            textfont=dict(size=12),
        ))
        fig.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(
                text="1 : 577<br>ratio",
                x=0.5, y=0.5,
                font_size=14,
                showarrow=False
            )]
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div class="finding-box">'
            'The model learns that predicting <b>"Not Fraud" every single time</b> gives '
            '99.83% accuracy — it never needs to learn what fraud looks like.'
            '</div>',
            unsafe_allow_html=True
        )

    st.divider()
    st.subheader("Real Model vs Dummy — Same Data")

    data = fetch("/experiments/results")
    if data and "experiment_1" in data:
        m = data["experiment_1"]["metrics"]
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
            "Dummy": [0.9983, 0.0, 0.0, 0.0, 0.5],
            "Logistic Regression": [
                m.get("accuracy", 0), m.get("precision", 0),
                m.get("recall", 0), m.get("f1", 0), m.get("roc_auc", 0)
            ],
        })
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="Dummy",
            x=metrics_df["Metric"], y=metrics_df["Dummy"],
            marker_color=C_RED, opacity=0.85
        ))
        fig2.add_trace(go.Bar(
            name="Logistic Regression",
            x=metrics_df["Metric"], y=metrics_df["Logistic Regression"],
            marker_color=C_PURPLE, opacity=0.85
        ))
        plotly_layout(fig2,
            barmode="group",
            yaxis=dict(range=[0, 1.1], title="Score"),
            legend=dict(orientation="h", y=1.1),
            margin=dict(t=20, b=20),
            height=320
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Accuracy barely changes — but Recall and F1 tell a completely different story.")
    else:
        st.info("Place colab_results.json in data/ to see the comparison chart.")

# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — Model Comparison
# ══════════════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.markdown(
        "7 models compared — with and without SMOTE. "
        "Complexity does not guarantee better fraud detection."
    )
    st.divider()

    data = fetch("/compare")
    if not data:
        st.stop()

    rows = []
    for model_name, result in data.items():
        if "error" in result:
            continue
        for variant, metrics in result.items():
            rows.append({
                "Model": model_name,
                "Variant": "With SMOTE" if variant == "with_smote" else "No Fix",
                **metrics
            })
    df = pd.DataFrame(rows)

    c1, c2 = st.columns([2, 1])
    with c1:
        metric = st.selectbox(
            "Compare by metric",
            ["f1", "recall", "precision", "accuracy", "roc_auc"],
            format_func=lambda x: {
                "f1": "F1 Score",
                "recall": "Recall (fraud caught)",
                "precision": "Precision",
                "accuracy": "Accuracy",
                "roc_auc": "ROC-AUC"
            }[x]
        )
    with c2:
        vf = st.selectbox("Show", ["Both", "No Fix", "With SMOTE"])

    filtered = df if vf == "Both" else df[df["Variant"] == vf]

    fig = px.bar(
        filtered.sort_values(metric, ascending=True),
        x=metric, y="Model",
        color="Variant",
        barmode="group",
        orientation="h",
        color_discrete_map={"No Fix": C_RED, "With SMOTE": C_GREEN},
        labels={metric: metric.upper().replace("_", "-"), "Model": ""},
        height=420
    )
    plotly_layout(fig,
        xaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=10, r=10, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    if metric == "recall":
        st.markdown(
            '<div class="finding-box">'
            '<b>What you\'re seeing:</b> Without imbalance handling, even XGBoost can have '
            'poor recall. SMOTE forces the model to learn fraud patterns by creating '
            'synthetic fraud samples.'
            '</div>',
            unsafe_allow_html=True
        )
    elif metric == "accuracy":
        st.markdown(
            '<div class="finding-box">'
            '<b>Notice:</b> Accuracy looks high for almost every model — this is the trap. '
            'Always check Recall and F1 for imbalanced data.'
            '</div>',
            unsafe_allow_html=True
        )

    with st.expander("See full numbers"):
        show = filtered[["Model", "Variant", "f1", "recall", "precision", "accuracy", "roc_auc"]].rename(
            columns={"f1": "F1", "recall": "Recall", "precision": "Precision",
                     "accuracy": "Accuracy", "roc_auc": "ROC-AUC"}
        )
        st.dataframe(
            show.style
                .highlight_max(subset=["F1", "Recall", "Precision", "ROC-AUC"], color="rgba(56,161,105,0.25)")
                .format(precision=4),
            use_container_width=True,
            hide_index=True
        )

# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — Failure Mode Explorer
# ══════════════════════════════════════════════════════════════════════
elif page == "Failure Mode Explorer":
    st.title("Failure Mode Explorer")
    st.markdown(
        "Four controlled experiments showing exactly *how* and *why* "
        "standard ML practices break on imbalanced data."
    )
    st.divider()

    exp_data = fetch("/experiments")
    colab    = fetch("/experiments/results")
    if not exp_data:
        st.stop()

    for exp in exp_data["experiments"]:
        with st.expander(
            f"Experiment {exp['id']} — {exp['title']}",
            expanded=(exp['id'] == 1)
        ):
            st.markdown(f"**Finding:** {exp['finding']}")
            st.markdown(f"*Key metric: `{exp['key_metric']}`*")

            if colab:
                if exp["id"] == 1 and "experiment_1" in colab:
                    m = colab["experiment_1"]["metrics"]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Accuracy",  f"{m.get('accuracy', 0):.2%}")
                    c2.metric("Precision", f"{m.get('precision', 0):.2%}")
                    c3.metric("Recall",    f"{m.get('recall', 0):.2%}")
                    c4.metric("F1 Score",  f"{m.get('f1', 0):.2%}")

                elif exp["id"] == 3 and "experiment_3" in colab:
                    e3_df = pd.DataFrame(colab["experiment_3"]).T.reset_index()
                    e3_df.columns = ["Model", "F1", "Recall", "Precision", "ROC-AUC", "Train F1"]
                    fig3 = go.Figure()
                    fig3.add_trace(go.Bar(
                        name="Train F1", x=e3_df["Model"], y=e3_df["Train F1"],
                        marker_color=C_PURPLE, opacity=0.6
                    ))
                    fig3.add_trace(go.Bar(
                        name="Test F1", x=e3_df["Model"], y=e3_df["F1"],
                        marker_color=C_RED, opacity=0.9
                    ))
                    plotly_layout(fig3,
                        barmode="group",
                        yaxis=dict(range=[0, 1], title="F1"),
                        height=300,
                        margin=dict(t=10, b=10),
                        legend=dict(orientation="h", y=1.1)
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                    st.caption("Gap between Train and Test F1 = overfitting. Larger gap = more overfit.")

                elif exp["id"] == 4 and "experiment_4" in colab:
                    e4_df = pd.DataFrame(colab["experiment_4"]).T.reset_index()
                    e4_df.columns = ["Strategy", "F1", "Recall", "Precision"]
                    fig4 = px.bar(
                        e4_df, x="Strategy",
                        y=["F1", "Recall", "Precision"],
                        barmode="group",
                        color_discrete_sequence=[C_PURPLE, C_GREEN, C_RED],
                        height=300
                    )
                    plotly_layout(fig4,
                        margin=dict(t=10, b=10),
                        legend=dict(orientation="h", y=1.1),
                        yaxis=dict(range=[0, 1])
                    )
                    st.plotly_chart(fig4, use_container_width=True)
                    st.caption("No single fix wins on all metrics. The best choice depends on business context.")

    st.divider()
    st.subheader("Key Takeaway")
    st.markdown(
        '<div class="finding-box">'
        '<b>ML success depends more on how we evaluate and reason about models '
        'than on which algorithm we use.</b><br><br>'
        'The best model here isn\'t the most complex one — it\'s the one evaluated '
        'with the right metric, the right validation strategy, and the right imbalance '
        'handling for the business context.'
        '</div>',
        unsafe_allow_html=True
    )