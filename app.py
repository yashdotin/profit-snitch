import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="ShopKart Profit Predictor",
    page_icon="📦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container {
        padding-top: 3.5rem;
        padding-bottom: 3rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 480px;
        margin-left: auto;
        margin-right: auto;
    }

    .app-title {
        font-size: 1.6rem;
        font-weight: 700;
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 0.1rem;
        line-height: 1.4;
    }
    .app-subtitle {
        font-size: 0.9rem;
        text-align: center;
        color: #888;
        margin-bottom: 1.5rem;
    }

    .section-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #4a4a4a;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-top: 1.2rem;
        margin-bottom: 0.4rem;
    }

    div[data-baseweb="select"] > div, .stNumberInput input, .stSlider {
        font-size: 1rem;
    }
    .stButton > button {
        width: 100%;
        height: 3rem;
        font-size: 1.05rem;
        font-weight: 600;
        border-radius: 10px;
        margin-top: 1.2rem;
    }

    .result-card {
        padding: 1.2rem;
        border-radius: 14px;
        text-align: center;
        margin-top: 1.2rem;
    }
    .result-high {
        background: green;
        border: 1px solid #4caf50;
    }
    .result-low {
        background: red;
        border: 1px solid #ff9800;
    }
    .result-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .result-conf {
        font-size: 0.95rem;
        color: #444;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    columns = joblib.load('columns.pkl')
    return model, scaler, columns

model, scaler, columns = load_artifacts()

CITIES = ["Bengaluru", "Chennai", "Delhi", "Hyderabad", "Jaipur", "Lucknow", "Mumbai", "Pune"]
CATEGORIES = ["Beauty", "Electronics", "Fashion", "Furniture", "Grocery", "Sports"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

st.markdown('<div class="app-title">📦 ShopKart Profit Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Fill in order details to predict profit category</div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Customer</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    customer_age = st.number_input("Age", min_value=18, max_value=65, value=30)
with c2:
    gender = st.selectbox("Gender", ["Male", "Female"])

st.markdown('<div class="section-label">Order</div>', unsafe_allow_html=True)
city = st.selectbox("City", CITIES)
category = st.selectbox("Category", CATEGORIES)

c3, c4 = st.columns(2)
with c3:
    qty = st.number_input("Quantity", min_value=1, max_value=5, value=2)
with c4:
    unit_price = st.number_input("Unit Price (Rs)", min_value=0, max_value=50000, value=10000, step=500)

c5, c6 = st.columns(2)
with c5:
    discount = st.number_input("Discount (%)", min_value=0, max_value=100, value=10)
with c6:
    sales = st.number_input("Sales (Rs)", min_value=0.0, value=20000.0, step=500.0)

st.markdown('<div class="section-label">Shipping & Rating</div>', unsafe_allow_html=True)
c7, c8 = st.columns(2)
with c7:
    shipping = st.number_input("Shipping Cost (Rs)", min_value=0.0, max_value=600.0, value=200.0, step=10.0)
with c8:
    delivery = st.number_input("Delivery Days", min_value=0, max_value=30, value=5)

rating = st.slider("Customer Rating", 1, 5, 3)

c9, c10 = st.columns(2)
with c9:
    order_month_label = st.selectbox("Order Month", MONTHS)
with c10:
    order_weekday_label = st.selectbox("Order Weekday", WEEKDAYS)

order_month = MONTHS.index(order_month_label) + 1
order_weekday = WEEKDAYS.index(order_weekday_label)

if st.button("Predict Profit Category"):
    row = {col: 0 for col in columns}

    row['Customer_Age'] = customer_age
    row['Gender'] = 0 if gender == "Male" else 1
    row['Qty'] = qty
    row['Unit Price'] = unit_price
    row['Discount'] = discount
    row['Shipping'] = shipping
    row['Delivery'] = delivery
    row['Sales'] = sales
    row['Rating'] = rating
    row['Order_Month'] = order_month
    row['Order_Weekday'] = order_weekday

    city_col = f"City_{city}"
    if city_col in row:
        row[city_col] = 1

    category_col = f"Category_{category}"
    if category_col in row:
        row[category_col] = 1

    input_df = pd.DataFrame([row])[columns]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    if prediction == 1:
        st.markdown(f"""
        <div class="result-card result-high">
            <div class="result-title">High-Profit Order</div>
            <div class="result-conf">Confidence: {proba[1]*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card result-low">
            <div class="result-title">Low-Profit Order</div>
            <div class="result-conf">Confidence: {proba[0]*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)