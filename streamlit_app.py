import os
import io
import base64
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Gemini SDK detection
GENAI_NEW = False
try:
    from google import genai as genai_new
    GENAI_NEW = True
except Exception:
    GENAI_NEW = False

# pyttsx3 TTS
try:
    import pyttsx3
    TTS_OK = True
except Exception:
    TTS_OK = False

# PDF exporter
try:
    from fpdf import FPDF
    PDF_OK = True
except Exception:
    PDF_OK = False

# -----------------------------
# Creative Aircraft UI (style)
# -----------------------------
st.set_page_config(page_title="AeroRunway", layout="wide")

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    /* App background - aviation gradient */
    .stApp {
        background: linear-gradient(180deg, #001326 0%, #003060 90%);
        color: #EAF4FF;
    }

    /* Glassmorphic cards */
    .air-card {
        background: rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 18px;
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 6px 30px rgba(0,0,0,0.45);
        backdrop-filter: blur(8px);
        margin-bottom: 18px;
    }

    /* Neon header */
    .air-header {
        font-size: 34px;
        font-weight: 700;
        color: #7cc9ff;
        text-shadow: 0px 0px 8px #2bbcff;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }

    /* Subtitles */
    .air-sub {
        font-size: 18px;
        font-weight: 600;
        color: #bdddff;
        margin-bottom: 10px;
    }

    /* Runway divider */
    .runway-line {
        height: 5px;
        background: repeating-linear-gradient(
            90deg,
            #FFFFFF 0px,
            #FFFFFF 12px,
            transparent 12px,
            transparent 24px
        );
        margin-top: 12px;
        margin-bottom: 20px;
        opacity: 0.45;
        border-radius: 2px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg,#0c84ff,#2bbcff);
        border-radius: 10px;
        color: white;
        padding: 8px 18px;
        border: none;
        font-weight: 600;
        box-shadow: 0 6px 18px rgba(43, 171, 255, 0.12);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
    }

    /* Metric styles tweaks */
    [data-testid="stMetricLabel"] > div { font-size: 14px; color: #d7eaff; }
    [data-testid="stMetricValue"] > div { font-size: 28px; font-weight:700; color:#69c5ff; }

    /* Input fields */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.12);
        color: #e6f7ff;
        border-radius: 10px;
        padding: 8px;
    }

    /* Small text */
    .small { font-size:12px; color:#cbd5e1; }

    </style>
    """,
    unsafe_allow_html=True,
)

# Developer-provided uploaded doc path (kept as a variable, not shown in UI)
uploaded_project_doc_path = "/mnt/data/Prototype to Production.pdf"

# -----------------------------
# Synthetic Data (same as before)
# -----------------------------
@st.cache_data(ttl=3600)
def generate_runway_data(days=7, step_minutes=10, seed=42):
    np.random.seed(seed)
    rows = []
    start = datetime.now().replace(hour=0, minute=0)
    total = (24*60//step_minutes) * days
    for i in range(total):
        ts = start + timedelta(minutes=i*step_minutes)
        hour = ts.hour
        base_arr = 5 + 8*np.exp(-(hour-9)**2/18) + 8*np.exp(-(hour-18)**2/18)
        base_dep = 4 + 6*np.exp(-(hour-8)**2/18) + 7*np.exp(-(hour-19)**2/18)
        arr = max(0, int(np.random.poisson(base_arr)))
        dep = max(0, int(np.random.poisson(base_dep)))
        traffic = arr + dep
        weather = np.random.choice([0,1,2], p=[0.85,0.13,0.02])
        vis = 10 - 2*weather + np.random.normal(0,0.4)
        taxi = 15 + traffic*0.4 + weather*4 + np.random.normal(0,1.5)
        util = min(1.0, traffic/22 + weather*0.08)
        rows.append({
            "timestamp": ts,
            "hour": hour,
            "arrivals": arr,
            "departures": dep,
            "traffic": traffic,
            "weather": int(weather),
            "visibility_km": float(round(vis,2)),
            "avg_taxi_time_min": float(round(taxi,1)),
            "runway_utilization": float(round(util,3))
        })
    return pd.DataFrame(rows)

# Session load
if "df" not in st.session_state:
    st.session_state.df = generate_runway_data()

# -----------------------------
# Tools / Agents (same logic)
# -----------------------------
def cleaning_agent(df):
    d = df.copy()
    d["visibility_km"] = d["visibility_km"].ffill().bfill()
    d["taxi_smooth"] = d["avg_taxi_time_min"].rolling(3, min_periods=1).mean()
    d["is_peak_hour"] = d["hour"].isin([7,8,9,17,18,19]).astype(int)
    return d

def prediction_agent(df):
    d = df.copy()
    d["lag1"] = d["traffic"].shift(1).bfill()
    d["lag2"] = d["traffic"].shift(2).bfill()
    feats = ["lag1","lag2","hour","weather","visibility_km","is_peak_hour"]
    X = d[feats].values
    y = d["traffic"].values
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    model = RandomForestRegressor(n_estimators=80, random_state=42)
    model.fit(Xs,y)
    d["pred_traffic"] = np.round(model.predict(Xs)).astype(int)
    return d, model, sc, feats

def optimizer_agent(df, top_k=5):
    r = df.tail(48).copy()
    r["score"] = (1-r["runway_utilization"]) * (r["visibility_km"]/(r["visibility_km"].max()+1e-6))
    best = r.sort_values("score", ascending=False).head(top_k)
    return best[["timestamp","traffic","runway_utilization","visibility_km","score"]]

# -----------------------------
# Gemini Insight (clean)
# -----------------------------
def gemini_insight(prompt_text, api_key):
    if not api_key:
        return local_insight()

    if GENAI_NEW:
        try:
            client = genai_new.Client(api_key=api_key)
            resp = client.models.generate_content(
                model="models/gemini-1.5-flash",
                contents=[{"role":"user","parts":[{"text":prompt_text}]}]
            )
            if resp and resp.candidates:
                parts = resp.candidates[0].content.parts
                if parts:
                    return parts[0].text
            return local_insight()
        except Exception:
            return local_insight()

    return local_insight()

def local_insight():
    d = cleaning_agent(st.session_state.df)
    top = d.groupby("hour").traffic.mean().sort_values(ascending=False).head(5)
    count = int((d.runway_utilization > 0.7).sum())

    # Nicely formatted insight
    text = "Local Insight:\n\nTop Busy Hours:\n"
    for h, v in top.items():
        text += f" • Hour {int(h)} → {v:.2f}\n"
    text += f"\nHigh Utilization Events: {count}\n\n"
    text += "Recommendation:\nShift non-urgent flights to low-utilization time windows."
    return text

# -----------------------------
# Text-to-speech (pyttsx3)
# -----------------------------
def text_to_speech_save(text: str, filename: str = "insight.wav"):
    if not TTS_OK:
        return False, "pyttsx3 not installed."
    try:
        engine = pyttsx3.init()
        try:
            engine.setProperty("rate", 160)
        except Exception:
            pass
        engine.save_to_file(text, filename)
        engine.runAndWait()
        return True, filename
    except Exception as e:
        return False, str(e)

def speak_text(text):
    if not TTS_OK:
        return
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate",165)
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass

# -----------------------------
# Navigation & pages
# -----------------------------
page = st.sidebar.radio("Navigate", ["Dashboard","EDA","Predict & Optimize","Insights","Export PDF"])

# -----------------------------
# Dashboard (creative UI)
# -----------------------------
if page == "Dashboard":
    st.markdown("<div class='air-header'>🛫 Runway Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='runway-line'></div>", unsafe_allow_html=True)

    st.markdown("<div class='air-card'>", unsafe_allow_html=True)
    df = cleaning_agent(st.session_state.df)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Avg Traffic/slot", f"{df.traffic.mean():.1f}")
    c2.metric("Max Traffic", int(df.traffic.max()))
    c3.metric("Avg Taxi Time (min)", f"{df.avg_taxi_time_min.mean():.1f}")
    c4.metric("High-util events (>0.7)", int((df.runway_utilization>0.7).sum()))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='air-sub'>📡 Traffic Heatmap (hour × day)</div>", unsafe_allow_html=True)
    tmp = df.copy()
    tmp["day"] = tmp["timestamp"].dt.date
    heat = tmp.pivot_table(values="traffic", index=tmp.timestamp.dt.hour, columns="day")
    st.plotly_chart(px.imshow(heat, color_continuous_scale="Blues"), use_container_width=True)

# -----------------------------
# EDA
# -----------------------------
elif page == "EDA":
    st.markdown("<div class='air-header'>📈 Exploratory Data Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='runway-line'></div>", unsafe_allow_html=True)
    st.markdown("<div class='air-card'>", unsafe_allow_html=True)

    df = cleaning_agent(st.session_state.df)
    n = st.slider("Last N slots", 30, 500, 120)
    sub = df.tail(n)
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(sub.timestamp, sub.traffic, label="Traffic")
    ax.plot(sub.timestamp, sub.avg_taxi_time_min, label="Taxi")
    ax.legend()
    st.pyplot(fig)

    st.markdown("### Correlation")
    st.dataframe(df[["traffic","avg_taxi_time_min","visibility_km","runway_utilization"]].corr())

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Predict & Optimize
# -----------------------------
elif page == "Predict & Optimize":
    st.markdown("<div class='air-header'>🤖 Prediction & Optimization</div>", unsafe_allow_html=True)
    st.markdown("<div class='runway-line'></div>", unsafe_allow_html=True)
    st.markdown("<div class='air-card'>", unsafe_allow_html=True)

    if st.button("Run Prediction Agent"):
        clean = cleaning_agent(st.session_state.df)
        dp, model, sc, feats = prediction_agent(clean)
        st.session_state.dp = dp
        st.session_state.model = model
        st.session_state.scaler = sc
        st.session_state.feats = feats
        st.success("Prediction completed.")

    if st.session_state.get("dp") is not None:
        dp = st.session_state.dp
        sub = dp.tail(120)
        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(sub.timestamp, sub.traffic, label="Actual")
        ax.plot(sub.timestamp, sub.pred_traffic, label="Predicted")
        ax.legend()
        st.pyplot(fig)

        mae = mean_absolute_error(dp.traffic, dp.pred_traffic)
        rmse = mean_squared_error(dp.traffic, dp.pred_traffic)**0.5
        st.write(f"**MAE: {mae:.2f}**  |  **RMSE: {rmse:.2f}**")

        if st.button("Run Optimizer"):
            rec = optimizer_agent(dp)
            st.dataframe(rec)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Insights (auto voice + text)
# -----------------------------
elif page == "Insights":
    st.markdown("<div class='air-header'>🎧 Insights</div>", unsafe_allow_html=True)
    st.markdown("<div class='runway-line'></div>", unsafe_allow_html=True)
    st.markdown("<div class='air-card'>", unsafe_allow_html=True)

    # secure key loading: show text box only if secret not set (keeps your previous UX)
    api_key = ""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        api_key = ""

    if not api_key:
        api_key = st.text_input("Gemini API Key (optional)", type="password")

    df = cleaning_agent(st.session_state.df)
    top_hours = df.groupby("hour").traffic.mean().sort_values(ascending=False).head(5)
    high_count = int((df.runway_utilization > 0.7).sum())

    prompt_text = (
        "You are an airport operations analyst. Provide a concise analysis and 3 recommendations.\n\n"
        f"Top busy hours:\n{top_hours.to_string(index=False)}\n\n"
        f"High-util events count: {high_count}\n\n"
        "Give severity rating, top 3 schedule optimization steps, and one-line executive summary."
    )

    if st.button("Generate Insight"):
        insight = gemini_insight(prompt_text, api_key)
        st.subheader("Insight")
        st.write(insight)

        # Auto voice (silent success)
        speak_text(insight)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Export PDF
# -----------------------------
elif page=="Export PDF":
    st.markdown("<div class='air-header'>📄 Export PDF</div>", unsafe_allow_html=True)
    st.markdown("<div class='runway-line'></div>", unsafe_allow_html=True)

    if not PDF_OK:
        st.error("Install fpdf: pip install fpdf")
    else:
        if st.button("Generate PDF"):

            df = cleaning_agent(st.session_state.df)

            # Create PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial","B",14)

            # ASCII-only title
            pdf.cell(0,10,"AeroRunway - Traffic Report", ln=True)

            pdf.set_font("Arial", size=11)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pdf.cell(0,6,"Generated: " + timestamp, ln=True)
            pdf.ln(4)

            # Top busy hours
            pdf.set_font("Arial","B",12)
            pdf.cell(0,6,"Top Busy Hours:", ln=True)

            pdf.set_font("Arial", size=10)
            top = df.groupby("hour").traffic.mean().sort_values(ascending=False).head(5)

            # SAFE ASCII LOOP
            for h,v in top.items():
                line = f"Hour {int(h)} - {v:.1f}"
                pdf.cell(0,5,line, ln=True)

            pdf.ln(6)

            # Plot image (safe)
            fig, ax = plt.subplots(figsize=(6,3))
            agg = df.groupby("hour").traffic.mean()
            ax.bar(agg.index, agg.values)
            ax.set_xlabel("Hour")
            ax.set_ylabel("Traffic")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)

            tmp = "plot.png"
            with open(tmp,"wb") as f:
                f.write(buf.getvalue())

            pdf.image(tmp, x=10, w=180)
            os.remove(tmp)

            outname = "AeroRunway_Report.pdf"
            pdf.output(outname)

            with open(outname,"rb") as f:
                b64 = base64.b64encode(f.read()).decode()

            st.markdown(
                f'<a href="data:application/pdf;base64,{b64}" download="{outname}">Download PDF</a>',
                unsafe_allow_html=True
            )

