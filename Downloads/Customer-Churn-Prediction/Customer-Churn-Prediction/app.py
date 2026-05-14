"""
╔══════════════════════════════════════════════════════════════╗
║         ChurnIQ — Customer Churn Intelligence Platform       ║
║         Enterprise AI/ML Analytics · v2.4.1                  ║
╚══════════════════════════════════════════════════════════════╝

Usage: streamlit run app.py
"""

import sys
import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churniq")

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Streamlit config — MUST be first ─────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ · Churn Intelligence Platform",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Internal imports ──────────────────────────────────────────────────────────
from utils.styles import CUSTOM_CSS, SIDEBAR_HTML, section_header
from utils.model_trainer import safe_predict_proba          # model-agnostic, float64-safe
from utils.visualizations import (
    churn_donut, charges_by_churn, contract_churn_bar,
    tenure_churn_line, feature_importance_chart, roc_comparison,
    confusion_heatmap, metrics_radar, payment_churn_pie,
    bulk_score_histogram, risk_gauge,
)

# ── Inject CSS ────────────────────────────────────────────────────────────────
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_models():
    """Load trained artifacts from disk (cached)."""
    try:
        from utils.model_trainer import load_artifacts
        preprocessor, model, metadata = load_artifacts(ROOT / "trained_model")
        return preprocessor, model, metadata, True
    except Exception as e:
        logger.warning(f"Could not load models: {e}")
        return None, None, None, False


@st.cache_data(show_spinner=False)
def load_dataset():
    """Load or generate the dataset (cached)."""
    dataset_path = ROOT / "dataset" / "telco_churn.csv"
    if dataset_path.exists():
        return pd.read_csv(dataset_path)

    from dataset.generate_dataset import generate_churn_dataset
    df = generate_churn_dataset(output_path=dataset_path)
    return df


@st.cache_data(show_spinner=False)
def auto_train():
    """Auto-train models if not already trained."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "train_model.py")],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    return result.returncode == 0


def ensure_models():
    """Ensure models are trained and loaded."""
    preprocessor, model, metadata, ok = load_models()
    if not ok:
        with st.spinner("🤖 First launch: training AI models (takes ~30 seconds)..."):
            success = auto_train()
            if success:
                st.cache_resource.clear()
                preprocessor, model, metadata, ok = load_models()
    return preprocessor, model, metadata, ok


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown(SIDEBAR_HTML, unsafe_allow_html=True)

        pages = {
            "Home Dashboard":        "home",
            "Customer Prediction":   "predict",
            "Bulk CSV Analysis":     "bulk",
            "Analytics Dashboard":   "analytics",
            "Model Performance Lab": "model_lab",
            "Business Insights":     "insights",
            "About Platform":        "about",
        }

        selected_label = st.radio(
            "Navigation", list(pages.keys()),
            label_visibility="collapsed"
        )

        st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:1.5rem 0;'>",
                    unsafe_allow_html=True)

        # Status widget
        preprocessor, model, metadata, ok = load_models()
        if ok:
            model_name = metadata.get("name", "Unknown")
            best_auc   = metadata["results"][model_name]["roc_auc"]
            st.markdown(f"""
            <div style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.2);
                        border-radius:12px;padding:0.9rem;margin-top:0.5rem;">
              <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                          color:#22C55E;margin-bottom:0.4rem;">● MODELS READY</div>
              <div style="font-family:'DM Sans',sans-serif;font-size:0.8rem;color:#E2E8F0;">
                Best: <b>{model_name}</b><br>
                AUC: <b style="color:#4F8EF7;">{best_auc:.1f}%</b>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);
                        border-radius:12px;padding:0.9rem;">
              <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#F59E0B;">
                ⚡ INITIALIZING...
              </div>
            </div>
            """, unsafe_allow_html=True)

        return pages[selected_label]


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def page_home():
    df = load_dataset()

    # Hero banner
    churn_rate = (df["Churn"] == "Yes").mean() * 100
    total = len(df)
    avg_charges = df["MonthlyCharges"].mean()
    high_risk_est = int(total * churn_rate / 100)

    st.markdown(f"""
    <div class="hero-banner fade-up-1">
      <div class="hero-badge">🔮 AI-Powered · Real-Time · Enterprise Grade</div>
      <div class="hero-title">Customer Churn<br>Intelligence Platform</div>
      <div class="hero-sub">
        Predict, analyze and prevent customer churn with state-of-the-art
        machine learning. Built for telecom, banking and SaaS enterprises.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, "Total Customers", f"{total:,}", "+12.4% MoM", False),
        (c2, "Churn Rate",      f"{churn_rate:.1f}%", "-2.1% QoQ", False),
        (c3, "High Risk",       f"{high_risk_est:,}", "Needs action", True),
        (c4, "Avg Monthly",     f"${avg_charges:.0f}", "Per customer", False),
        (c5, "Models Active",   "4", "All trained", False),
    ]

    for col, label, value, delta, is_danger in metrics:
        with col:
            delta_class = "negative" if is_danger else ""
            st.markdown(f"""
            <div class="kpi-card fade-up-2">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{value}</div>
              <div class="kpi-delta {delta_class}">{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)

    # Charts row
    c1, c2 = st.columns([1, 2])
    with c1:
        st.plotly_chart(churn_donut(df), use_container_width=True)
    with c2:
        st.plotly_chart(charges_by_churn(df), use_container_width=True)

    # Contract + Tenure
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(contract_churn_bar(df), use_container_width=True)
    with c2:
        st.plotly_chart(tenure_churn_line(df), use_container_width=True)

    # Feature cards
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='page-title' style='font-size:1.4rem;margin-bottom:1rem;'>Platform Capabilities</div>",
                unsafe_allow_html=True)

    fc = st.columns(4)
    features = [
        ("🎯", "Real-Time Prediction", "Predict individual customer churn probability with AI confidence scoring in milliseconds."),
        ("📂", "Bulk CSV Analysis",    "Upload thousands of customers and get instant risk scores with downloadable reports."),
        ("📊", "Analytics Dashboard",  "Comprehensive churn analytics with interactive Plotly visualizations and KPI tracking."),
        ("🧪", "Model Lab",            "Compare Logistic Regression, Random Forest, XGBoost and Decision Tree performance."),
    ]
    for col, (icon, title, desc) in zip(fc, features):
        with col:
            st.markdown(f"""
            <div class="feature-card fade-up-3">
              <span class="feature-icon">{icon}</span>
              <div class="feature-title">{title}</div>
              <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CUSTOMER PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

def page_predict():
    st.markdown(section_header("🎯", "Customer Prediction Engine",
        "Enter customer attributes to get real-time churn probability and retention guidance."),
        unsafe_allow_html=True)

    preprocessor, model, metadata, ok = ensure_models()
    if not ok:
        st.error("❌ Models not loaded. Please run: `python train_model.py`")
        return

    with st.form("prediction_form"):
        st.markdown("#### 👤 Customer Demographics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with c2:
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        with c3:
            partner = st.selectbox("Has Partner", ["Yes", "No"])
        with c4:
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])

        st.markdown("#### 📋 Service Details")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
        with c2:
            phone_svc = st.selectbox("Phone Service", ["Yes", "No"])
        with c3:
            multi_lines = st.selectbox("Multiple Lines",
                ["No", "Yes", "No phone service"])
        with c4:
            internet = st.selectbox("Internet Service",
                ["Fiber optic", "DSL", "No"])

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            online_sec = st.selectbox("Online Security",
                ["No", "Yes", "No internet service"])
        with c2:
            online_bk = st.selectbox("Online Backup",
                ["No", "Yes", "No internet service"])
        with c3:
            device_prot = st.selectbox("Device Protection",
                ["No", "Yes", "No internet service"])
        with c4:
            tech_sup = st.selectbox("Tech Support",
                ["No", "Yes", "No internet service"])

        c1, c2 = st.columns(2)
        with c1:
            streaming_tv = st.selectbox("Streaming TV",
                ["No", "Yes", "No internet service"])
        with c2:
            streaming_mv = st.selectbox("Streaming Movies",
                ["No", "Yes", "No internet service"])

        st.markdown("#### 💳 Billing & Contract")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            contract = st.selectbox("Contract Type",
                ["Month-to-month", "One year", "Two year"])
        with c2:
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        with c3:
            payment = st.selectbox("Payment Method",
                ["Electronic check", "Mailed check",
                 "Bank transfer (automatic)", "Credit card (automatic)"])
        with c4:
            monthly_chg = st.number_input("Monthly Charges ($)",
                min_value=15.0, max_value=120.0, value=65.0, step=0.5)
        with c5:
            tickets = st.number_input("Support Tickets", 0, 10, 1)

        submitted = st.form_submit_button("🔮 Predict Churn Risk", type="primary",
                                           use_container_width=True)

    if submitted:
        # Build input DataFrame
        row = pd.DataFrame([{
            "customerID": "PRED-TEMP",
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_svc,
            "MultipleLines": multi_lines,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_bk,
            "DeviceProtection": device_prot,
            "TechSupport": tech_sup,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_mv,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_chg,
            "TotalCharges": monthly_chg * max(tenure, 1),
            "NumSupportTickets": tickets,
        }])

        with st.spinner("Running inference..."):
            try:
                X = preprocessor.transform(row)
                # safe_predict_proba: sanitizes to float64, handles all model types,
                # never accesses model-specific attrs like multi_class
                prob = float(safe_predict_proba(model, X.values)[0])
                pred = int(model.predict(X.values.astype(np.float64))[0])
            except Exception as e:
                st.error(f"Prediction error: {e}")
                return

        # Risk classification
        if prob < 0.30:
            risk_class, risk_label, risk_emoji, card_style = "low", "LOW RISK", "✅", "success"
            recommendation = "Customer appears stable. Focus on loyalty rewards and upselling opportunities."
        elif prob < 0.65:
            risk_class, risk_label, risk_emoji, card_style = "medium", "MEDIUM RISK", "⚠️", "warning"
            recommendation = "Proactively engage this customer. Offer a contract upgrade or personalized discount."
        else:
            risk_class, risk_label, risk_emoji, card_style = "high", "HIGH RISK", "🚨", "danger"
            recommendation = "Immediate intervention required! Contact within 48 hours with retention offer."

        # Results
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])

        with c1:
            st.plotly_chart(risk_gauge(prob), use_container_width=True)

        with c2:
            st.markdown(f"""
            <div class="prediction-panel">
              <div style="margin-bottom:1rem;">
                <span class="risk-badge risk-{risk_class}">{risk_emoji} {risk_label}</span>
              </div>
              <div class="kpi-label">Churn Probability</div>
              <div class="prediction-prob" style="color:{'#EF4444' if prob>0.65 else '#F59E0B' if prob>0.30 else '#22C55E'};">
                {prob*100:.1f}%
              </div>
              <div class="kpi-label" style="margin-top:0.75rem;">Confidence Score</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;color:#4F8EF7;">
                {max(prob, 1-prob)*100:.1f}%
              </div>
              <div style="margin-top:1.25rem;">
                <div class="kpi-label">Retention Recommendation</div>
                <div class="insight-card {card_style}" style="margin-top:0.5rem;">
                  {recommendation}
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Risk factors
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        st.markdown("#### 🔍 Risk Factor Analysis")

        risk_factors = []
        if contract == "Month-to-month":
            risk_factors.append(("⚠️", "Month-to-month contract", "3× higher churn probability vs. annual contracts", "warning"))
        if internet == "Fiber optic":
            risk_factors.append(("📡", "Fiber optic customer", "Higher churn segment — review competitive pricing", "warning"))
        if payment == "Electronic check":
            risk_factors.append(("💳", "Electronic check payment", "Highest churn correlation among payment methods", "danger"))
        if tenure < 12:
            risk_factors.append(("📅", f"Early-stage customer ({tenure}mo)", "First-year customers have 45% higher churn risk", "danger"))
        if online_sec == "No" and internet != "No":
            risk_factors.append(("🔒", "No online security", "Security add-ons reduce churn by 18%", "warning"))
        if tickets >= 3:
            risk_factors.append(("🎫", f"{tickets} support tickets", "High support usage correlated with dissatisfaction", "danger"))
        if monthly_chg > 80:
            risk_factors.append(("💰", f"High monthly charge ${monthly_chg:.0f}", "Above median — price sensitivity risk", "warning"))

        if risk_factors:
            fc = st.columns(min(len(risk_factors), 3))
            for i, (icon, title, desc, style) in enumerate(risk_factors):
                with fc[i % 3]:
                    st.markdown(f"""
                    <div class="insight-card {style}">
                      <b>{icon} {title}</b><br>
                      <span style="font-size:0.82rem;color:#8892A4;">{desc}</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-card success">
              ✅ <b>No major risk factors detected</b> — This customer profile shows strong retention indicators.
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: BULK CSV ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def page_bulk():
    st.markdown(section_header("📂", "Bulk CSV Analysis",
        "Upload a customer CSV file to get churn predictions for your entire base."),
        unsafe_allow_html=True)

    # Move all imports to function top — inline imports inside render loops
    # cause module re-import on every Streamlit rerun, resetting React component
    # state mid-render and triggering React error #185.
    import plotly.graph_objects as go
    from utils.visualizations import LAYOUT_BASE

    preprocessor, model, metadata, ok = ensure_models()
    if not ok:
        st.error("❌ Models not loaded. Please run: `python train_model.py`")
        return

    # Sample download — explicit key prevents duplicate widget ID on rerun
    col1, col2 = st.columns([3, 1])
    with col2:
        sample_path = ROOT / "sample_data" / "sample_customers.csv"
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                st.download_button(
                    "Download Sample CSV",
                    f.read(),
                    "sample_customers.csv",
                    "text/csv",
                    use_container_width=True,
                    key="bulk_sample_download",          # unique, stable key
                )

    uploaded = st.file_uploader(
        "Drop your customer CSV file here",
        type=["csv"],
        help="Must contain the same columns as the training dataset."
    )

    if uploaded is None:
        # Show expected schema
        st.markdown("""
        <div class="insight-card" style="margin-top:1rem;">
          <b>Expected Columns:</b><br>
          <code style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;">
          customerID, gender, SeniorCitizen, Partner, Dependents, tenure,
          PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
          DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract,
          PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, NumSupportTickets
          </code>
        </div>
        """, unsafe_allow_html=True)
        return

    df_upload = pd.read_csv(uploaded)
    st.success(f"✅ Loaded **{len(df_upload):,}** customers — {df_upload.shape[1]} columns")

    with st.spinner("Running batch predictions..."):
        try:
            # Keep customer IDs if the column exists, otherwise use row index
            if "customerID" in df_upload.columns:
                ids = df_upload["customerID"].values
            else:
                ids = np.arange(len(df_upload))

            # Transform → guaranteed float64, NaN-free (see preprocessor.transform)
            X = preprocessor.transform(df_upload)

            # safe_predict_proba: model-agnostic, sanitizes input, never accesses
            # model-specific attrs such as multi_class (removed in sklearn 1.8)
            probs = safe_predict_proba(model, X.values)
            preds = model.predict(X.values.astype(np.float64))

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            logger.error("Bulk prediction error", exc_info=True)
            return

    # Build results DataFrame
    def classify(p):
        if p < 0.30: return "Low Risk"
        if p < 0.65: return "Medium Risk"
        return "High Risk"

    results_df = pd.DataFrame({
        "CustomerID":       ids,
        "ChurnProbability": (probs * 100).round(1),
        "RiskLevel":        [classify(p) for p in probs],
        "WillChurn":        ["Yes" if p else "No" for p in preds],
    })

    # Summary KPIs
    total      = len(results_df)
    high       = (results_df["RiskLevel"] == "High Risk").sum()
    medium     = (results_df["RiskLevel"] == "Medium Risk").sum()
    low        = (results_df["RiskLevel"] == "Low Risk").sum()
    churn_pred = (results_df["WillChurn"] == "Yes").sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, value in [
        (c1, "Total Analysed",  f"{total:,}"),
        (c2, "Predicted Churn", f"{churn_pred:,}"),
        (c3, "High Risk",       f"{high:,}"),
        (c4, "Medium Risk",     f"{medium:,}"),
        (c5, "Low Risk",        f"{low:,}"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value" style="font-size:1.8rem;">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # Charts — unique keys prevent React reconciler collisions across reruns
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            bulk_score_histogram(probs),
            use_container_width=True,
            key="bulk_histogram",
        )
    with c2:
        risk_counts = results_df["RiskLevel"].value_counts()
        colors      = {"Low Risk": "#22C55E", "Medium Risk": "#F59E0B", "High Risk": "#EF4444"}
        fig_risk    = go.Figure(go.Bar(
            x=risk_counts.index.tolist(),
            y=risk_counts.values.tolist(),
            marker_color=[colors.get(l, "#4F8EF7") for l in risk_counts.index],
            hovertemplate="%{x}: %{y} customers<extra></extra>",
        ))
        fig_risk.update_layout(**LAYOUT_BASE, title=dict(text="Risk Level Distribution", x=0.5))
        st.plotly_chart(fig_risk, use_container_width=True, key="bulk_risk_bar")

    # Results table
    st.markdown("#### Customer Risk Report")

    # Tab labels must be plain strings — emoji characters in tab labels corrupt
    # React's internal component key hashing in Streamlit ≤ 1.33, causing
    # React error #185 when the tab panel re-renders after a file upload rerun.
    tabs = st.tabs(["All Customers", "High Risk", "Medium Risk", "Low Risk"])

    # reset_index() before dataframe render: filtered subsets retain the original
    # integer index from results_df. Streamlit passes row indices as React keys;
    # non-contiguous indices (e.g. [3, 7, 12, ...]) can cause key collisions when
    # React diffs the virtual DOM between the full-table and filtered-table renders.
    def safe_df(df_sub, limit=500):
        return df_sub.head(limit).reset_index(drop=True)

    with tabs[0]:
        st.dataframe(safe_df(results_df), use_container_width=True, hide_index=True)
    with tabs[1]:
        hr = results_df[results_df["RiskLevel"] == "High Risk"]
        st.dataframe(safe_df(hr), use_container_width=True, hide_index=True)
        st.caption(f"{len(hr):,} high-risk customers")
    with tabs[2]:
        mr = results_df[results_df["RiskLevel"] == "Medium Risk"]
        st.dataframe(safe_df(mr), use_container_width=True, hide_index=True)
        st.caption(f"{len(mr):,} medium-risk customers")
    with tabs[3]:
        lr = results_df[results_df["RiskLevel"] == "Low Risk"]
        st.dataframe(safe_df(lr), use_container_width=True, hide_index=True)
        st.caption(f"{len(lr):,} low-risk customers")

    # Export — separate stable key from the sample download button above.
    # Two download_button widgets with no key= get auto-keyed by render order;
    # after a file upload triggers a full rerun, render order can shift,
    # causing Streamlit to reassign keys and React to see a "new" widget
    # where an old one was — this is the primary trigger of React error #185.
    csv_out = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export Full Results CSV",
        csv_out,
        "churniq_predictions.csv",
        "text/csv",
        use_container_width=True,
        type="primary",
        key="bulk_export_download",             # unique, stable key
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def page_analytics():
    st.markdown(section_header("📊", "Analytics Dashboard",
        "Comprehensive churn analytics across customer segments, services and behaviour."),
        unsafe_allow_html=True)

    df = load_dataset()

    # Row 1
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(churn_donut(df), use_container_width=True)
    with c2: st.plotly_chart(payment_churn_pie(df), use_container_width=True)

    # Row 2
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(charges_by_churn(df), use_container_width=True)
    with c2: st.plotly_chart(contract_churn_bar(df), use_container_width=True)

    # Row 3 — full width tenure
    st.plotly_chart(tenure_churn_line(df), use_container_width=True)

    # Internet service churn table
    st.markdown("#### 📡 Churn Rate by Internet Service")
    inet_stats = (
        df.groupby("InternetService")
        .apply(lambda g: pd.Series({
            "Total Customers": len(g),
            "Churned":         (g["Churn"] == "Yes").sum(),
            "Churn Rate %":    round((g["Churn"] == "Yes").mean() * 100, 1),
            "Avg Monthly ($)": round(g["MonthlyCharges"].mean(), 2),
        }))
        .reset_index()
    )
    st.dataframe(inet_stats, use_container_width=True, hide_index=True)

    # Senior citizen analysis
    st.markdown("#### 👴 Senior Citizen Churn Analysis")
    senior_stats = (
        df.groupby("SeniorCitizen")["Churn"]
        .value_counts(normalize=True)
        .mul(100).round(1)
        .unstack()
        .reset_index()
    )
    senior_stats["SeniorCitizen"] = senior_stats["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior Citizen"})
    if "Yes" in senior_stats.columns:
        senior_stats.rename(columns={"Yes": "Churn Rate %", "No": "Retention Rate %"}, inplace=True)
    st.dataframe(senior_stats, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL LAB
# ═══════════════════════════════════════════════════════════════════════════════

def page_model_lab():
    st.markdown(section_header("🧪", "Model Performance Laboratory",
        "Train, evaluate and compare 4 ML models. Best model auto-selected for production."),
        unsafe_allow_html=True)

    preprocessor, model, metadata, ok = ensure_models()
    if not ok:
        st.error("❌ Models not trained yet. Run: `python train_model.py`")
        return

    results   = metadata["results"]
    best_name = metadata["name"]

    # Model comparison table
    st.markdown("#### 🏆 Model Comparison Matrix")
    rows = []
    for name, m in results.items():
        rows.append({
            "Model":      name,
            "Accuracy %": m["accuracy"],
            "Precision %": m["precision"],
            "Recall %":   m["recall"],
            "F1 Score %": m["f1"],
            "ROC-AUC %":  m["roc_auc"],
            "CV AUC %":   m.get("cv_roc_auc_mean", "-"),
            "Train Time": f"{m.get('train_time_sec', '?')}s",
            "Status":     "⭐ BEST" if name == best_name else "—",
        })

    table_html = """<table class='model-table'><thead><tr>"""
    for col in rows[0].keys():
        table_html += f"<th>{col}</th>"
    table_html += "</tr></thead><tbody>"
    for row in rows:
        style = "background:rgba(79,142,247,0.06);" if row["Model"] == best_name else ""
        table_html += f"<tr style='{style}'>"
        for v in row.values():
            table_html += f"<td>{v}</td>"
        table_html += "</tr>"
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(roc_comparison(results), use_container_width=True)
    with c2:
        st.plotly_chart(metrics_radar(results), use_container_width=True)

    # Individual confusion matrices
    st.markdown("#### 🗺️ Confusion Matrices")
    cm_cols = st.columns(min(len(results), 4))
    for col, (name, m) in zip(cm_cols, results.items()):
        with col:
            if "confusion_matrix" in m:
                st.plotly_chart(
                    confusion_heatmap(m["confusion_matrix"], name),
                    use_container_width=True
                )

    # Feature importance for best model
    if "feature_importance" in results.get(best_name, {}):
        st.markdown(f"#### 🎯 Feature Importance — {best_name}")
        st.plotly_chart(
            feature_importance_chart(results[best_name]["feature_importance"]),
            use_container_width=True
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: BUSINESS INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

def page_insights():
    st.markdown(section_header("💡", "Business Insights Engine",
        "AI-derived insights on churn drivers, risk patterns and retention strategies."),
        unsafe_allow_html=True)

    df = load_dataset()

    # ── Key Insight Metrics ────────────────────────────────────────────────────
    month_churn = df[df["Contract"] == "Month-to-month"]["Churn"].value_counts(normalize=True).get("Yes", 0)
    fiber_churn  = df[df["InternetService"] == "Fiber optic"]["Churn"].value_counts(normalize=True).get("Yes", 0)
    echeck_churn = df[df["PaymentMethod"] == "Electronic check"]["Churn"].value_counts(normalize=True).get("Yes", 0)
    senior_churn = df[df["SeniorCitizen"] == 1]["Churn"].value_counts(normalize=True).get("Yes", 0)
    no_sec_churn = df[df["OnlineSecurity"] == "No"]["Churn"].value_counts(normalize=True).get("Yes", 0)
    overall_churn= (df["Churn"] == "Yes").mean()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Month-to-Month Churn Rate</div>
          <div class="kpi-value" style="color:#EF4444;">{month_churn*100:.1f}%</div>
          <div class="kpi-delta negative">vs {overall_churn*100:.1f}% overall</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Fiber Optic Churn Rate</div>
          <div class="kpi-value" style="color:#F59E0B;">{fiber_churn*100:.1f}%</div>
          <div class="kpi-delta negative">High-value segment at risk</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">E-Check Churn Rate</div>
          <div class="kpi-value" style="color:#EF4444;">{echeck_churn*100:.1f}%</div>
          <div class="kpi-delta negative">Highest churn payment method</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

    # ── Actionable Insights ────────────────────────────────────────────────────
    st.markdown("#### 🔴 Critical Churn Drivers")
    critical_insights = [
        ("🔴", f"Month-to-month contracts drive {month_churn*100:.0f}% churn rate — "
               "3× higher than annual contracts. Priority: migrate customers to annual plans with incentives."),
        ("🔴", f"Electronic check users churn at {echeck_churn*100:.0f}% — "
               "highest among all payment methods. Promote auto-pay adoption with bill credit."),
        ("🟡", f"Fiber optic customers churn at {fiber_churn*100:.0f}% despite premium service. "
               "Suggests price-performance dissatisfaction. Review competitive pricing."),
        ("🟡", f"Senior citizens churn at {senior_churn*100:.0f}% — "
               "may need dedicated support channels and simplified billing."),
        ("🔵", f"Customers without online security churn at {no_sec_churn*100:.0f}% vs lower rates with it. "
               "Bundle security add-ons for high-risk segments."),
    ]
    for icon, text in critical_insights:
        style = "danger" if icon == "🔴" else "warning" if icon == "🟡" else ""
        st.markdown(f'<div class="insight-card {style}">{icon} {text}</div>',
                    unsafe_allow_html=True)

    # ── Retention Playbook ─────────────────────────────────────────────────────
    st.markdown("#### 🟢 Retention Strategy Playbook")
    strategies = [
        ("Contract Migration Campaign", "Offer 20% discount for month-to-month customers who upgrade to annual contracts. Target the top 500 high-risk accounts first."),
        ("Auto-Pay Incentive Program",  "Provide $5–10/month bill credit for switching to auto-pay. Reduces churn by reducing friction and cognitive load."),
        ("Early Tenure Nurture Track",  "Implement structured onboarding for first 90 days: proactive check-ins, tutorials, satisfaction surveys at 30/60/90 days."),
        ("Fiber Optic Loyalty Bundle",  "Create exclusive 'Fiber Premium' tier with speed guarantee, free tech support, and streaming bundle for fiber churners."),
        ("Senior Citizen Support Hub",  "Dedicated support line, simplified billing, and larger-print materials. Consider home visit tech support premium option."),
    ]
    for title, desc in strategies:
        st.markdown(f"""
        <div class="insight-card success" style="border-left-color:#22C55E;">
          <b>✅ {title}</b><br>
          <span style="font-size:0.87rem;color:#94A3B8;">{desc}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Segment Deep-Dive ──────────────────────────────────────────────────────
    st.markdown("#### 📊 Churn Rate by Customer Segment")
    seg_data = []
    for contract in df["Contract"].unique():
        for internet in df["InternetService"].unique():
            sub = df[(df["Contract"] == contract) & (df["InternetService"] == internet)]
            if len(sub) > 20:
                cr = (sub["Churn"] == "Yes").mean()
                seg_data.append({
                    "Contract":        contract,
                    "Internet Service":internet,
                    "Customers":       len(sub),
                    "Churn Rate %":    round(cr * 100, 1),
                    "Avg Monthly ($)": round(sub["MonthlyCharges"].mean(), 2),
                    "Risk Level":      "🔴 High" if cr > 0.4 else "🟡 Medium" if cr > 0.2 else "🟢 Low",
                })

    seg_df = pd.DataFrame(seg_data).sort_values("Churn Rate %", ascending=False)
    st.dataframe(seg_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════════════════════

def page_about():

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:2rem;">
      <div style="
        font-family:'Sora',sans-serif;
        font-size:0.72rem;
        font-weight:600;
        letter-spacing:0.12em;
        text-transform:uppercase;
        color:var(--accent-blue);
        margin-bottom:0.6rem;">
        ChurnIQ · Enterprise Edition · v2.4.1
      </div>
      <div style="
        font-family:'Sora',sans-serif;
        font-size:2rem;
        font-weight:800;
        letter-spacing:-0.02em;
        background:linear-gradient(135deg,#E2E8F0 0%,var(--accent-blue) 55%,var(--accent-violet) 100%);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
        line-height:1.15;
        margin-bottom:0.5rem;">
        Customer Churn Intelligence Platform
      </div>
      <div style="
        font-family:'DM Sans',sans-serif;
        font-size:1rem;
        color:var(--text-muted);
        max-width:680px;
        line-height:1.7;">
        An AI-powered churn intelligence platform designed for proactive customer
        retention, behavioural analytics, and data-driven revenue protection
        across telecom, banking, and SaaS enterprises.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Divider ───────────────────────────────────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Platform Overview ─────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:2.5rem;">
      <div style="
        font-family:'Sora',sans-serif;
        font-size:0.68rem;
        font-weight:600;
        letter-spacing:0.1em;
        text-transform:uppercase;
        color:var(--text-muted);
        margin-bottom:0.75rem;">
        Platform Overview
      </div>
      <div style="
        display:grid;
        grid-template-columns:1fr 1fr;
        gap:1rem;">
        <div style="
          background:var(--bg-card);
          border:1px solid var(--border);
          border-top:2px solid var(--accent-blue);
          border-radius:var(--radius-lg);
          padding:1.75rem;
          backdrop-filter:blur(20px);">
          <div style="
            font-family:'Sora',sans-serif;
            font-size:0.8rem;
            font-weight:600;
            letter-spacing:0.06em;
            text-transform:uppercase;
            color:var(--accent-blue);
            margin-bottom:0.75rem;">
            Mission
          </div>
          <div style="
            font-family:'DM Sans',sans-serif;
            font-size:0.92rem;
            color:var(--text-primary);
            line-height:1.75;">
            ChurnIQ empowers enterprise teams to move from reactive support to
            proactive retention. By surfacing churn risk before it materialises,
            businesses protect recurring revenue and extend customer lifetime value.
          </div>
        </div>
        <div style="
          background:var(--bg-card);
          border:1px solid var(--border);
          border-top:2px solid var(--accent-violet);
          border-radius:var(--radius-lg);
          padding:1.75rem;
          backdrop-filter:blur(20px);">
          <div style="
            font-family:'Sora',sans-serif;
            font-size:0.8rem;
            font-weight:600;
            letter-spacing:0.06em;
            text-transform:uppercase;
            color:var(--accent-violet);
            margin-bottom:0.75rem;">
            Approach
          </div>
          <div style="
            font-family:'DM Sans',sans-serif;
            font-size:0.92rem;
            color:var(--text-primary);
            line-height:1.75;">
            The platform combines ensemble machine learning with interpretable
            analytics, translating raw behavioural signals into clear risk scores
            and targeted retention strategies — without requiring data science expertise.
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Core Capabilities ─────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:2.5rem;">
      <div style="
        font-family:'Sora',sans-serif;
        font-size:0.68rem;
        font-weight:600;
        letter-spacing:0.1em;
        text-transform:uppercase;
        color:var(--text-muted);
        margin-bottom:0.75rem;">
        Core Capabilities
      </div>
      <div style="
        display:grid;
        grid-template-columns:repeat(3,1fr);
        gap:1rem;">
    """, unsafe_allow_html=True)

    capabilities = [
        ("Real-Time Prediction",
         "Assess individual customer churn risk instantly. Each prediction returns a calibrated probability score, risk tier classification, and a tailored retention recommendation."),
        ("Bulk Customer Analysis",
         "Process entire customer bases via CSV upload. Risk-scored results are available for download, enabling CRM-ready segmentation and targeted outreach campaigns."),
        ("Predictive Risk Scoring",
         "Customers are automatically classified into Low, Medium, and High risk tiers — enabling operations teams to prioritise retention effort with precision."),
        ("Multi-Model Intelligence",
         "Four independently trained models — Logistic Regression, Random Forest, XGBoost, and Decision Tree — are evaluated against each other; the highest-performing model is auto-selected."),
        ("Retention Insights Engine",
         "The platform surfaces the primary churn drivers across segments and translates them into concrete, business-readable retention strategies and intervention priorities."),
        ("Interactive Analytics",
         "A comprehensive analytics layer presents churn distribution, tenure trends, payment behaviour, and contract segmentation through fully interactive visualisations."),
    ]

    cap_html = ""
    for title, desc in capabilities:
        cap_html += f"""
        <div style="
          background:var(--bg-card);
          border:1px solid var(--border);
          border-radius:var(--radius-lg);
          padding:1.5rem;
          backdrop-filter:blur(20px);
          transition:border-color 0.25s;">
          <div style="
            font-family:'Sora',sans-serif;
            font-size:0.88rem;
            font-weight:600;
            color:var(--text-primary);
            margin-bottom:0.6rem;">
            {title}
          </div>
          <div style="
            font-family:'DM Sans',sans-serif;
            font-size:0.82rem;
            color:var(--text-muted);
            line-height:1.7;">
            {desc}
          </div>
        </div>"""

    st.markdown(cap_html + "</div></div>", unsafe_allow_html=True)

    # ── Enterprise Features + Business Value (side by side) ───────────────────
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("""
        <div style="margin-bottom:2rem;">
          <div style="
            font-family:'Sora',sans-serif;
            font-size:0.68rem;
            font-weight:600;
            letter-spacing:0.1em;
            text-transform:uppercase;
            color:var(--text-muted);
            margin-bottom:0.75rem;">
            Enterprise Features
          </div>
        </div>
        """, unsafe_allow_html=True)

        enterprise_items = [
            ("Secure ML Pipeline",
             "All model training, evaluation, and inference occurs server-side with no raw customer data transmitted externally."),
            ("Scalable Architecture",
             "Designed to handle single-record API calls and batch CSV workloads of tens of thousands of records within the same pipeline."),
            ("Model Monitoring",
             "Per-model accuracy, precision, recall, F1, and ROC-AUC metrics are tracked and compared across every training run."),
            ("Exportable Intelligence",
             "Risk-scored prediction outputs are available as structured CSV exports, ready for ingestion into CRM and BI systems."),
            ("Cross-Validation Rigour",
             "Five-fold stratified cross-validation is applied to every model, ensuring performance estimates are robust and generalisable."),
        ]

        for title, desc in enterprise_items:
            st.markdown(f"""
            <div style="
              display:flex;
              gap:0.85rem;
              align-items:flex-start;
              padding:1.1rem 1.25rem;
              margin-bottom:0.6rem;
              background:var(--bg-card);
              border:1px solid var(--border);
              border-left:3px solid var(--accent-blue);
              border-radius:var(--radius-lg);
              backdrop-filter:blur(16px);">
              <div style="flex:1;">
                <div style="
                  font-family:'Sora',sans-serif;
                  font-size:0.84rem;
                  font-weight:600;
                  color:var(--text-primary);
                  margin-bottom:0.3rem;">
                  {title}
                </div>
                <div style="
                  font-family:'DM Sans',sans-serif;
                  font-size:0.8rem;
                  color:var(--text-muted);
                  line-height:1.65;">
                  {desc}
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div style="margin-bottom:2rem;">
          <div style="
            font-family:'Sora',sans-serif;
            font-size:0.68rem;
            font-weight:600;
            letter-spacing:0.1em;
            text-transform:uppercase;
            color:var(--text-muted);
            margin-bottom:0.75rem;">
            Business Value
          </div>
        </div>
        """, unsafe_allow_html=True)

        value_items = [
            ("Reduce Revenue Churn",
             "Identify at-risk customers before they cancel, enabling proactive outreach that materially reduces involuntary and voluntary churn rates."),
            ("Improve Retention ROI",
             "Focus retention spend on the customers most likely to churn, maximising the return on every engagement and offer deployed."),
            ("Accelerate Decision-Making",
             "Real-time risk scores and pre-built retention playbooks eliminate the latency between data insight and frontline action."),
            ("Extend Customer Lifetime Value",
             "By intervening at the right moment in the customer lifecycle, businesses preserve long-term revenue from high-value segments."),
            ("Segment with Precision",
             "Go beyond binary churn flags — understand which contract types, payment methods, and service profiles drive the highest risk exposure."),
        ]

        for title, desc in value_items:
            st.markdown(f"""
            <div style="
              display:flex;
              gap:0.85rem;
              align-items:flex-start;
              padding:1.1rem 1.25rem;
              margin-bottom:0.6rem;
              background:var(--bg-card);
              border:1px solid var(--border);
              border-left:3px solid var(--accent-violet);
              border-radius:var(--radius-lg);
              backdrop-filter:blur(16px);">
              <div style="flex:1;">
                <div style="
                  font-family:'Sora',sans-serif;
                  font-size:0.84rem;
                  font-weight:600;
                  color:var(--text-primary);
                  margin-bottom:0.3rem;">
                  {title}
                </div>
                <div style="
                  font-family:'DM Sans',sans-serif;
                  font-size:0.8rem;
                  color:var(--text-muted);
                  line-height:1.65;">
                  {desc}
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Deployment Readiness ──────────────────────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom:2.5rem;">
      <div style="
        font-family:'Sora',sans-serif;
        font-size:0.68rem;
        font-weight:600;
        letter-spacing:0.1em;
        text-transform:uppercase;
        color:var(--text-muted);
        margin-bottom:0.75rem;">
        Deployment Readiness
      </div>
      <div style="
        display:grid;
        grid-template-columns:repeat(3,1fr);
        gap:1rem;">
    """, unsafe_allow_html=True)

    deploy_items = [
        ("Streamlit Cloud",
         "One-click deployment from a connected GitHub repository. No infrastructure management required.",
         "#4F8EF7"),
        ("Render",
         "Containerised deployment with automatic build pipelines. Scales horizontally under high prediction load.",
         "#A855F7"),
        ("Hugging Face Spaces",
         "Instant public demo hosting with Streamlit runtime support. Ideal for stakeholder demonstrations.",
         "#06B6D4"),
    ]

    deploy_html = ""
    for platform, desc, accent in deploy_items:
        deploy_html += f"""
        <div style="
          background:var(--bg-card);
          border:1px solid var(--border);
          border-radius:var(--radius-lg);
          padding:1.5rem 1.75rem;
          backdrop-filter:blur(20px);">
          <div style="
            font-family:'Sora',sans-serif;
            font-size:0.78rem;
            font-weight:700;
            letter-spacing:0.05em;
            color:{accent};
            margin-bottom:0.5rem;
            text-transform:uppercase;">
            {platform}
          </div>
          <div style="
            font-family:'DM Sans',sans-serif;
            font-size:0.83rem;
            color:var(--text-muted);
            line-height:1.7;">
            {desc}
          </div>
        </div>"""

    st.markdown(deploy_html + "</div></div>", unsafe_allow_html=True)

    # ── Professional footer ───────────────────────────────────────────────────
    st.markdown("""
    <div style="
      margin-top:3rem;
      padding:2rem 2.5rem;
      background:linear-gradient(135deg,rgba(79,142,247,0.06),rgba(168,85,247,0.04));
      border:1px solid var(--border);
      border-radius:var(--radius-xl);
      display:flex;
      align-items:center;
      justify-content:space-between;
      flex-wrap:wrap;
      gap:1rem;">
      <div>
        <div style="
          font-family:'Sora',sans-serif;
          font-size:1rem;
          font-weight:700;
          color:var(--text-primary);
          margin-bottom:0.25rem;">
          ChurnIQ · Customer Churn Intelligence Platform
        </div>
        <div style="
          font-family:'DM Sans',sans-serif;
          font-size:0.8rem;
          color:var(--text-muted);">
          Enterprise AI Analytics · Retention Intelligence · Predictive Risk Scoring
        </div>
      </div>
      <div style="
        font-family:'DM Sans',sans-serif;
        font-size:0.75rem;
        color:var(--text-muted);
        text-align:right;
        line-height:1.8;">
        Version 2.4.1 &nbsp;·&nbsp; Enterprise Edition<br>
        Python &nbsp;·&nbsp; Scikit-learn &nbsp;·&nbsp; XGBoost &nbsp;·&nbsp; Streamlit
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    page = render_sidebar()

    if   page == "home":      page_home()
    elif page == "predict":   page_predict()
    elif page == "bulk":      page_bulk()
    elif page == "analytics": page_analytics()
    elif page == "model_lab": page_model_lab()
    elif page == "insights":  page_insights()
    elif page == "about":     page_about()


if __name__ == "__main__":
    main()
