# SkySense AI  
### Runway Traffic Optimization Agent (Enterprise Agents Track)

SkySense AI is an intelligent multi-agent system designed to analyze runway conditions, predict congestion, optimize schedules, and generate real-time operational insights for airport traffic controllers.  
Built with Streamlit, ML models, and Gemini-powered intelligence, SkySense AI demonstrates practical enterprise automation for aviation operations.

---

## ✈️ Problem Statement
Airports struggle with runway congestion, inefficient scheduling, and delays caused by sudden changes in traffic, weather, and operational workload.  
Manual monitoring makes it difficult to:

- Detect peak congestion hours  
- Predict incoming runway traffic  
- Optimize available slots  
- Generate actionable insights quickly  

This leads to operational delays, higher fuel burn, and reduced runway efficiency.

---

## 💡 Solution — SkySense AI  
SkySense AI acts as a **Runway Traffic Optimization Agent** that:

### ✔ Cleans and analyzes live-like runway data  
### ✔ Predicts traffic using ML (Random Forest)  
### ✔ Identifies best operational windows  
### ✔ Generates AI insights (optional Gemini integration)  
### ✔ Provides voice-based explanations  
### ✔ Exports professional PDF reports  
### ✔ Uses a creative aviation-themed UI for real-world usability  

It reduces manual workload and allows faster, more accurate airport decision-making.

---

## 🧠 Architecture Overview

SkySense AI follows a **multi-agent architecture** with:

### **1️⃣ Cleaning Agent**
- Preprocesses and smooths data  
- Handles visibility, taxi time anomalies  
- Detects peak hours automatically  

### **2️⃣ Prediction Agent**
- Builds Random Forest model  
- Predicts near-future traffic  
- Computes performance metrics (MAE, RMSE)

### **3️⃣ Optimization Agent**
- Identifies best 5 timeslots  
- Scores slots using utilization + visibility  
- Helps schedule non-urgent flights  

### **4️⃣ Insight Agent (Gemini optional)**
- Summarizes congestion severity  
- Recommends actions  
- Auto-generates voice briefing  

All agents work sequentially and collectively → a multi-agent workflow.

---

## 🛰 Tools & AI Features Used

| Feature | Used in Project |
|--------|----------------|
| Multi-Agent System | ✔ Cleaning, Prediction, Optimization, Insight Agents |
| LLM Agent | ✔ Gemini Insight Agent |
| Tools | ✔ Custom Tools (Cleaner, Predictor, Optimizer) |
| Session Memory | ✔ Streamlit session state for model, data, and prediction memory |
| Observability | ✔ Metrics (MAE, RMSE), charts, logs removed for cleaner UI |
| Context Engineering | ✔ Data summarization for Gemini prompts |
| Export Tools | ✔ PDF Report generator |
| Voice System | ✔ pyttsx3 TTS |

You satisfy **more than 3 mandatory features**, making your submission strong.

---

## 🚀 Project Flow

1. **Synthetic runway dataset generated**
2. **Cleaning Agent** prepares data  
3. **Prediction Agent** trains ML & predicts  
4. **Optimizer Agent** recommends ideal windows  
5. **Insight Agent** generates AI summary  
6. **Voice engine** speaks insights  
7. **User exports a PDF report**

---

## 🖥️ Tech Stack

- **Python**
- **Streamlit UI**
- **Scikit-Learn**
- **NumPy / Pandas**
- **Plotly / Matplotlib**
- **Gemini LLM (optional)**
- **pyttsx3 (Voice)**  
- **FPDF (PDF exporter)**

---

## 📸 Screenshots
(Add images here after uploading on GitHub)
