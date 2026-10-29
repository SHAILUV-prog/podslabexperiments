"""Customer Churn Prediction - Streamlit app.

The churn result comes ONLY from the saved scikit-learn pipeline
(preprocessing + trained classifier). Nothing is retrained here.
"""
import json

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "model/churn_model.pkl"
METRICS_PATH = "model/metrics.json"

st.set_page_config(page_title="Customer Churn Prediction", page_icon="🔮", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


try:
    model = load_model()
    info = load_metrics()
except FileNotFoundError:
    st.error("Trained model not found. Run `python train_model.py` first.")
    st.stop()

FEATURES = info["feature_columns"]  # same column order used in training

# ---------- default values & demo customers ----------
DEFAULTS = dict(customer_id="CUST100001", age=30, gender="Male", plan="Standard",
                tenure=12, charges=399.0, watch=10.0, login=5, failures=0,
                complaints=0, auto_renew="Yes")
HIGH_RISK = dict(customer_id="CUST-HIGH-01", age=24, gender="Male", plan="Basic",
                 tenure=4, charges=199.0, watch=3.0, login=2, failures=2,
                 complaints=1, auto_renew="No")
LOW_RISK = dict(customer_id="CUST-LOW-01", age=41, gender="Female", plan="Premium",
                tenure=48, charges=649.0, watch=22.0, login=12, failures=0,
                complaints=0, auto_renew="Yes")

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def load_example(example: dict):
    for k, v in example.items():
        st.session_state[k] = v
    st.session_state.pop("result", None)


# ---------- header ----------
st.title("Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn based on subscription and behavioral data.")
st.caption("Demo use case: a Netflix-inspired subscription streaming service. "
           "Trained on a synthetic dataset; not affiliated with or used by Netflix.")

# ---------- demo buttons ----------
b1, b2, _ = st.columns([1, 1, 1])
b1.button("🧪 Load High-Risk Example", on_click=load_example, args=(HIGH_RISK,))
b2.button("🧪 Load Low-Risk Example", on_click=load_example, args=(LOW_RISK,))

# ---------- inputs ----------
st.subheader("Customer Information")
st.text_input("Customer ID", key="customer_id")

c1, c2 = st.columns(2)
with c1:
    st.number_input("Age", min_value=18, max_value=100, step=1, key="age")
    st.selectbox("Subscription Plan", ["Basic", "Standard", "Premium"], key="plan")
    st.number_input("Monthly Charges (₹)", min_value=0.0, max_value=5000.0, step=10.0, key="charges")
    st.number_input("Login Frequency per Week", min_value=0, max_value=50, step=1, key="login")
    st.number_input("Customer Complaints", min_value=0, max_value=50, step=1, key="complaints")
with c2:
    st.selectbox("Gender", ["Male", "Female", "Other"], key="gender")
    st.number_input("Tenure (months)", min_value=0, max_value=120, step=1, key="tenure")
    st.number_input("Watch Hours per Week", min_value=0.0, max_value=168.0, step=0.5, key="watch")
    st.number_input("Payment Failures", min_value=0, max_value=50, step=1, key="failures")
    st.selectbox("Auto Renewal", ["Yes", "No"], key="auto_renew")

# ---------- predict ----------
if st.button("🔮 PREDICT CHURN", type="primary", use_container_width=True):
    s = st.session_state
    if not s.customer_id.strip():
        st.warning("Please enter a Customer ID.")
    elif s.charges <= 0:
        st.warning("Monthly Charges must be greater than 0.")
    else:
        input_data = pd.DataFrame([{
            "Age": s.age, "TenureMonths": s.tenure, "MonthlyCharges": s.charges,
            "WatchHoursPerWeek": s.watch, "LoginFrequencyPerWeek": s.login,
            "PaymentFailures": s.failures, "CustomerComplaints": s.complaints,
            "Gender": s.gender, "SubscriptionPlan": s.plan, "AutoRenewal": s.auto_renew,
        }])[FEATURES]

        prediction = model.predict(input_data)[0]                 # 1 = churn
        probability = float(model.predict_proba(input_data)[0][1])  # P(churn)
        st.session_state["result"] = dict(
            prediction=int(prediction), probability=probability,
            snapshot=input_data.iloc[0].to_dict(), cid=s.customer_id.strip())

# ---------- result ----------
res = st.session_state.get("result")
if res:
    p = res["probability"]
    risk = "LOW" if p < 0.30 else ("MEDIUM" if p <= 0.60 else "HIGH")
    churn = res["prediction"] == 1

    st.divider()
    st.markdown("<h3 style='text-align:center'>CUSTOMER CHURN RESULT</h3>", unsafe_allow_html=True)
    st.caption(f"Customer ID: {res['cid']}")

    box = st.error if churn else st.success
    box(f"### {'🔴 LIKELY TO CHURN' if churn else '🟢 LIKELY TO STAY'}\n\n"
        f"**Prediction:** {'YES' if churn else 'NO'}\n\n"
        f"**Churn Probability:** {p * 100:.1f}%\n\n"
        f"**Risk Level:** {risk}")

    st.write("**Churn Probability**")
    st.progress(min(max(p, 0.0), 1.0), text=f"{p * 100:.1f}%")

    # Explanatory observations only - NOT how the prediction is made
    d = res["snapshot"]
    notes = []
    if d["WatchHoursPerWeek"] <= 4:
        notes.append("Low weekly engagement may indicate reduced platform usage.")
    if d["PaymentFailures"] >= 2:
        notes.append("Multiple payment failures may indicate a payment-related churn risk.")
    if d["CustomerComplaints"] >= 2:
        notes.append("Multiple complaints may indicate customer dissatisfaction.")
    if d["AutoRenewal"] == "No":
        notes.append("Disabled auto-renewal may increase the possibility of subscription cancellation.")
    if d["TenureMonths"] <= 6:
        notes.append("The customer has a relatively short subscription history.")

    st.subheader("Why might this customer be at risk?")
    st.caption("Potential contributing factors — simple observations on the entered values. "
               "They are not the exact reasons behind the model's prediction.")
    if notes:
        for n in notes:
            st.markdown(f"- {n}")
    else:
        st.write("No common risk indicators were observed in the entered values.")

# ---------- model info ----------
st.divider()
st.subheader("Model Information")
m = info["metrics"]
st.markdown(
    f"**Model Used:** {info['selected_model']}  \n"
    f"**Accuracy:** {m['accuracy'] * 100:.1f}%  \n"
    f"**Precision:** {m['precision'] * 100:.1f}%  \n"
    f"**Recall:** {m['recall'] * 100:.1f}%  \n"
    f"**F1 Score:** {m['f1_score'] * 100:.1f}%  \n"
    f"**ROC-AUC:** {m['roc_auc'] * 100:.1f}%"
)
with st.expander("Compare all trained models"):
    st.dataframe((pd.DataFrame(info["all_models"]).T * 100).round(1).rename(
        columns={"accuracy": "Accuracy %", "precision": "Precision %", "recall": "Recall %",
                 "f1_score": "F1 %", "roc_auc": "ROC-AUC %"}))
