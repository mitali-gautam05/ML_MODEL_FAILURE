import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

API = "http://localhost:8000"

st.set_page_config(
    page_title="When ML Fails",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.danger-card{background:#fff5f5;border-left:4px solid #e53e3e;border-radius:10px;padding:1rem 1.5rem;margin-bottom:.5rem}
.success-card{background:#f0fff4;border-left:4px solid #38a169;border-radius:10px;padding:1rem 1.5rem;margin-bottom:.5rem}
.finding-box{background:#fffbeb;border:1px solid #f6ad55;border-radius:8px;padding:1rem;margin:.5rem 0}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("When ML Fails")
    st.caption("A controlled study of ML failure modes on imbalanced data")
    st.divider()
    page = st.radio("Navigate", ["The Accuracy Lie","Model Comparison","Failure Mode Explorer"], label_visibility="collapsed")
    st.divider()
    st.caption("Dataset: Credit Card Fraud")
    st.caption("284,807 transactions · 0.17% fraud")
    st.caption("7 models compared")

def fetch(endpoint):
    try:
        r = requests.get(f"{API}{endpoint}", timeout=5)
        r.raise_for_status()
        return r.json()
    except:
        st.error("API not reachable. Run: `uvicorn backend.main:app --reload`")
        return None

# ── PAGE 1 ────────────────────────────────────────────────────────────
if page == "The Accuracy Lie":
    st.title("The Accuracy Lie")
    st.markdown("A model that **never detects fraud** scores **99.83% accuracy**. Here's why that's dangerous.")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dummy Classifier")
        st.markdown("*Strategy: always predict 'Not Fraud'*")
        st.markdown('<div class="success-card"><h2 style="margin:0;color:#38a169">99.83%</h2><p style="margin:0;color:#666">Accuracy — looks amazing</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="danger-card"><h2 style="margin:0;color:#e53e3e">0%</h2><p style="margin:0;color:#666">Fraud Recall — caught ZERO fraud</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="danger-card"><h2 style="margin:0;color:#e53e3e">~50%</h2><p style="margin:0;color:#666">ROC-AUC — random guessing level</p></div>', unsafe_allow_html=True)

    with col2:
        st.subheader("Why this happens")
        fig = go.Figure(go.Pie(
            labels=["Not Fraud (99.83%)", "Fraud (0.17%)"],
            values=[284315, 492],
            hole=0.55,
            marker_colors=["#6c63ff","#e53e3e"],
            textinfo="label+percent",
        ))
        fig.update_layout(showlegend=False, margin=dict(t=10,b=10,l=10,r=10), height=280,
            annotations=[dict(text="1 : 577<br>ratio", x=0.5, y=0.5, font_size=14, showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="finding-box">The model learns that predicting "Not Fraud" <b>every single time</b> gives 99.83% accuracy — it never needs to learn what fraud looks like.</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("Real Model vs Dummy — Same Data")
    data = fetch("/experiments/results")
    if data and "experiment_1" in data:
        m = data["experiment_1"]["metrics"]
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy","Precision","Recall","F1","ROC-AUC"],
            "Dummy":  [0.9983, 0.0, 0.0, 0.0, 0.5],
            "Logistic Regression": [m.get("accuracy",0), m.get("precision",0), m.get("recall",0), m.get("f1",0), m.get("roc_auc",0)],
        })
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Dummy", x=metrics_df["Metric"], y=metrics_df["Dummy"], marker_color="#e53e3e", opacity=0.8))
        fig2.add_trace(go.Bar(name="Logistic Regression", x=metrics_df["Metric"], y=metrics_df["Logistic Regression"], marker_color="#6c63ff", opacity=0.8))
        fig2.update_layout(barmode="group", yaxis=dict(range=[0,1.1],title="Score"),
            legend=dict(orientation="h",y=1.1), margin=dict(t=20,b=20), height=320)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Accuracy barely changes — but Recall and F1 tell a completely different story.")
    else:
        st.info("Place colab_results.json in data/ to see the comparison chart.")

# ── PAGE 2 ────────────────────────────────────────────────────────────
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.markdown("7 models compared — with and without SMOTE. Complexity does not guarantee better fraud detection.")
    st.divider()

    data = fetch("/compare")
    if not data: st.stop()

    rows = []
    for model_name, result in data.items():
        if "error" in result: continue
        for variant, metrics in result.items():
            rows.append({"Model": model_name, "Variant": "With SMOTE" if variant=="with_smote" else "No Fix", **metrics})
    df = pd.DataFrame(rows)

    c1, c2 = st.columns([2,1])
    with c1:
        metric = st.selectbox("Compare by metric",
            ["f1","recall","precision","accuracy","roc_auc"],
            format_func=lambda x: {"f1":"F1 Score","recall":"Recall (fraud caught)","precision":"Precision","accuracy":"Accuracy","roc_auc":"ROC-AUC"}[x])
    with c2:
        vf = st.selectbox("Show", ["Both","No Fix","With SMOTE"])

    filtered = df if vf=="Both" else df[df["Variant"]==vf]
    fig = px.bar(filtered.sort_values(metric, ascending=True), x=metric, y="Model", color="Variant",
        barmode="group", orientation="h",
        color_discrete_map={"No Fix":"#e53e3e","With SMOTE":"#38a169"},
        labels={metric: metric.upper().replace("_","-"), "Model":""}, height=420)
    fig.update_layout(xaxis=dict(range=[0,1.05]), legend=dict(orientation="h",y=1.05), margin=dict(l=10,r=10,t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)

    if metric == "recall":
        st.markdown('<div class="finding-box"><b>What you\'re seeing:</b> Without imbalance handling, even XGBoost can have poor recall. SMOTE forces the model to learn fraud patterns by creating synthetic fraud samples.</div>', unsafe_allow_html=True)
    elif metric == "accuracy":
        st.markdown('<div class="finding-box"><b>Notice:</b> Accuracy looks high for almost every model — this is the trap. Always check Recall and F1 for imbalanced data.</div>', unsafe_allow_html=True)

    with st.expander("See full numbers"):
        show = filtered[["Model","Variant","f1","recall","precision","accuracy","roc_auc"]].rename(
            columns={"f1":"F1","recall":"Recall","precision":"Precision","accuracy":"Accuracy","roc_auc":"ROC-AUC"})
        st.dataframe(show.style.highlight_max(subset=["F1","Recall","Precision","ROC-AUC"], color="#c6f6d5").format(precision=4),
            use_container_width=True, hide_index=True)

# ── PAGE 3 ────────────────────────────────────────────────────────────
elif page == "Failure Mode Explorer":
    st.title("Failure Mode Explorer")
    st.markdown("Four controlled experiments showing exactly *how* and *why* standard ML practices break on imbalanced data.")
    st.divider()

    exp_data = fetch("/experiments")
    colab    = fetch("/experiments/results")
    if not exp_data: st.stop()

    for exp in exp_data["experiments"]:
        with st.expander(f"Experiment {exp['id']} — {exp['title']}", expanded=(exp['id']==1)):
            st.markdown(f"**Finding:** {exp['finding']}")
            st.markdown(f"*Key metric: `{exp['key_metric']}`*")

            if colab:
                if exp["id"]==1 and "experiment_1" in colab:
                    m = colab["experiment_1"]["metrics"]
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("Accuracy",  f"{m.get('accuracy',0):.2%}")
                    c2.metric("Precision", f"{m.get('precision',0):.2%}")
                    c3.metric("Recall",    f"{m.get('recall',0):.2%}")
                    c4.metric("F1 Score",  f"{m.get('f1',0):.2%}")

                elif exp["id"]==3 and "experiment_3" in colab:
                    e3_df = pd.DataFrame(colab["experiment_3"]).T.reset_index()
                    e3_df.columns = ["Model","F1","Recall","Precision","ROC-AUC","Train F1"]
                    fig3 = go.Figure()
                    fig3.add_trace(go.Bar(name="Train F1", x=e3_df["Model"], y=e3_df["Train F1"], marker_color="#6c63ff", opacity=0.6))
                    fig3.add_trace(go.Bar(name="Test F1",  x=e3_df["Model"], y=e3_df["F1"],       marker_color="#e53e3e", opacity=0.9))
                    fig3.update_layout(barmode="group", yaxis=dict(range=[0,1],title="F1"), height=300,
                        margin=dict(t=10,b=10), legend=dict(orientation="h",y=1.1))
                    st.plotly_chart(fig3, use_container_width=True)
                    st.caption("Gap between Train and Test F1 = overfitting. Larger gap = more overfit.")

                elif exp["id"]==4 and "experiment_4" in colab:
                    e4_df = pd.DataFrame(colab["experiment_4"]).T.reset_index()
                    e4_df.columns = ["Strategy","F1","Recall","Precision"]
                    fig4 = px.bar(e4_df, x="Strategy", y=["F1","Recall","Precision"], barmode="group",
                        color_discrete_sequence=["#6c63ff","#38a169","#e53e3e"], height=300)
                    fig4.update_layout(margin=dict(t=10,b=10), legend=dict(orientation="h",y=1.1), yaxis=dict(range=[0,1]))
                    st.plotly_chart(fig4, use_container_width=True)
                    st.caption("No single fix wins on all metrics. The best choice depends on business context.")

    st.divider()
    st.subheader("Key Takeaway")
    st.markdown('<div class="finding-box"><b>ML success depends more on how we evaluate and reason about models than on which algorithm we use.</b><br><br>The best model here isn\'t the most complex one — it\'s the one evaluated with the right metric, the right validation strategy, and the right imbalance handling for the business context.</div>', unsafe_allow_html=True)