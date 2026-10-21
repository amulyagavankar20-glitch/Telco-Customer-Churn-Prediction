import os
import requests
import streamlit as st

st.set_page_config(page_title="ChurnIQ", page_icon="📊", layout="wide")
API_URL = os.getenv("CHURN_API_URL", "http://127.0.0.1:8000").rstrip("/")
api_health = {}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:#0b1020;color:#edf2ff}
[data-testid="stSidebar"]{background:#10182b;border-right:1px solid #293750}
.hero{padding:1.6rem;border-radius:22px;background:linear-gradient(115deg,#182b4a,#243d65 55%,#29335c);border:1px solid #344969;margin-bottom:1rem}
.eyebrow{color:#8ce6d2;font-size:.78rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase}
.hero h1{font-family:Manrope,sans-serif;color:#f4f7ff;font-size:2.2rem;margin:.4rem 0}
.hero p,.muted{color:#b6c4dc}
.section{font-family:Manrope,sans-serif;font-size:1.2rem;font-weight:800;margin:.5rem 0}
.metric,.result{background:#131d31;border:1px solid #293750;border-radius:16px;padding:1rem}
.metric{min-height:95px}.metric-label{color:#9eacc5;font-size:.8rem}
.metric-value{font-family:Manrope,sans-serif;color:#f4f7ff;font-size:1.7rem;font-weight:800}
div[data-testid="stForm"]{background:#111a2c;border:1px solid #293750;border-radius:16px;padding:1.2rem}
div[data-testid="stFormSubmitButton"] button{border-radius:10px;min-height:2.8rem;font-weight:700}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ◈ ChurnIQ")
    st.caption("CUSTOMER INTELLIGENCE")
    page = st.radio("Workspace", ["Churn prediction", "Model information"])
    st.divider()
    st.markdown("**API status**")
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=3)
        health_response.raise_for_status()
        health = health_response.json()
        api_health = health
        st.success("Backend connected")
        st.caption(f"Model: {health.get('model', 'Loaded')}")
        st.caption(f"Threshold: {health.get('decision_threshold', health.get('threshold', .541)):.3f}")
    except requests.RequestException:
        st.error("Backend unavailable")
        st.caption("Start FastAPI or set CHURN_API_URL.")

st.markdown("""
<div class="hero"><div class="eyebrow">Predictive customer analytics</div>
<h1>Understand churn before it happens.</h1>
<p>Score customer risk with your trained machine-learning pipeline and prioritize retention outreach.</p></div>
""", unsafe_allow_html=True)

if page == "Model information":
    st.markdown('<div class="section">Model overview</div>', unsafe_allow_html=True)
    st.markdown('<p class="muted">The backend serves the saved preprocessing pipeline and Logistic Regression model.</p>', unsafe_allow_html=True)
    metrics = api_health.get("test_metrics", {})
    cols = st.columns(4)
    metric_values = [
        ("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}"),
        ("PR-AUC", f"{metrics.get('pr_auc', 0):.4f}"),
        ("Recall", f"{metrics.get('recall', 0):.2%}"),
        ("F1 score", f"{metrics.get('f1', 0):.4f}"),
    ]
    for col, (name, val) in zip(cols, metric_values):
        col.markdown(f'<div class="metric"><div class="metric-label">{name}</div><div class="metric-value">{val}</div></div>', unsafe_allow_html=True)
    if metrics:
        st.info("Evaluation results come from the saved model metadata. Performance may vary on new customer populations.")
    else:
        st.warning("Model metrics are unavailable because the backend did not return them.")
else:
    left, right = st.columns([1.25,.75], gap="large")
    with left:
        st.markdown('<div class="section">Customer profile</div><p class="muted">Enter account, service, and billing details.</p>', unsafe_allow_html=True)
        with st.form("churn_form"):
            st.markdown("**Account & relationship**")
            a,b,c=st.columns(3)
            gender=a.selectbox("Gender",["Female","Male"])
            senior=b.selectbox("Senior citizen",[0,1],format_func=lambda x:"Yes" if x else "No")
            partner=c.selectbox("Partner",["Yes","No"])
            a,b,c=st.columns(3)
            dependents=a.selectbox("Dependents",["No","Yes"])
            tenure=b.number_input("Tenure (months)",0,72,12)
            contract=c.selectbox("Contract",["Month-to-month","One year","Two year"])
            st.markdown("**Services**")
            a,b=st.columns(2)
            phone=a.selectbox("Phone service",["Yes","No"])
            multiple=b.selectbox("Multiple lines",["No phone service","No","Yes"])
            internet=a.selectbox("Internet service",["Fiber optic","DSL","No"])
            security=b.selectbox("Online security",["No","Yes","No internet service"])
            backup=a.selectbox("Online backup",["No","Yes","No internet service"])
            device=b.selectbox("Device protection",["No","Yes","No internet service"])
            support=a.selectbox("Tech support",["No","Yes","No internet service"])
            tv=b.selectbox("Streaming TV",["No","Yes","No internet service"])
            movies=a.selectbox("Streaming movies",["No","Yes","No internet service"])
            st.markdown("**Billing & payment**")
            a,b,c=st.columns(3)
            paperless=a.selectbox("Paperless billing",["Yes","No"])
            payment=b.selectbox("Payment method",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
            monthly=c.number_input("Monthly charges ($)",0.0,1000.0,70.0,1.0)
            total=st.number_input("Total charges ($)",0.0,100000.0,float(monthly*tenure),10.0)
            submitted=st.form_submit_button("✦  Assess churn risk",type="primary",use_container_width=True)
    with right:
        st.markdown('<div class="section">Assessment</div><p class="muted">Prediction results appear here.</p>',unsafe_allow_html=True)
        if submitted:
            payload={"gender":gender,"SeniorCitizen":int(senior),"Partner":partner,"Dependents":dependents,
             "tenure":float(tenure),"PhoneService":phone,"MultipleLines":multiple,"InternetService":internet,
             "OnlineSecurity":security,"OnlineBackup":backup,"DeviceProtection":device,"TechSupport":support,
             "StreamingTV":tv,"StreamingMovies":movies,"Contract":contract,"PaperlessBilling":paperless,
             "PaymentMethod":payment,"MonthlyCharges":float(monthly),"TotalCharges":str(total)}
            try:
                with st.spinner("Scoring customer..."):
                    response=requests.post(f"{API_URL}/predict",json=payload,timeout=15)
                response.raise_for_status()
                result=response.json()
                probability=float(result["churn_probability"])
                if result["predicted_churn"]:
                    st.error("Elevated churn risk")
                    st.markdown("### :red[" + f"{probability:.1%}" + "]")
                    st.write("Consider a retention conversation, service review, or plan-fit check.")
                else:
                    st.success("Lower predicted churn risk")
                    st.markdown("### :green[" + f"{probability:.1%}" + "]")
                    st.write("Continue standard engagement and monitor changes.")
                st.progress(min(max(probability,0),1))
                st.caption(f"Decision threshold: {result['decision_threshold']:.3f}")
            except requests.HTTPError:
                try: detail=response.json().get("detail","Check the submitted fields.")
                except ValueError: detail=response.text
                st.error(f"Prediction rejected: {detail}")
            except requests.ConnectionError:
                st.error(f"Cannot reach API at {API_URL}. Start FastAPI and retry.")
            except requests.Timeout:
                st.error("Prediction service timed out. Please retry.")
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
        else:
            st.markdown('<div class="result"><div class="metric-label">READY TO ANALYZE</div><h3>Customer churn score</h3><p class="muted">Complete the form and select Assess churn risk to request a live prediction.</p></div>',unsafe_allow_html=True)
        st.divider()
        st.markdown("**How to use this score**")
        st.markdown('<p class="muted">Use the probability to prioritize human review. It is not a guarantee that a customer will churn.</p>',unsafe_allow_html=True)

st.divider()
st.caption("ChurnIQ · Streamlit + FastAPI · Requires a reachable prediction backend.")
