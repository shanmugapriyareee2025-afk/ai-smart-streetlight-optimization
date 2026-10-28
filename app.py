import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Smart Streetlight Optimizer", page_icon="💡", layout="wide")
st.title("💡 AI-Based Smart Streetlight Energy Optimization")
st.caption("Educational prototype using simulated LDR/PIR sensor data. Not connected to real streetlights.")

@st.cache_data
def make_dataset(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    hour = rng.integers(0, 24, n)
    ambient = np.clip(70 + 450*np.sin((hour-6)*np.pi/12), 0, 600) + rng.normal(0, 35, n)
    ambient = np.clip(ambient, 0, 600)
    motion = rng.integers(0, 2, n)
    # Labels: OFF in daylight, HIGH with motion in darkness, DIM in darkness without motion.
    mode = []
    for light, movement in zip(ambient, motion):
        if light >= 250:
            mode.append("OFF")
        elif movement == 1:
            mode.append("HIGH")
        else:
            mode.append("DIM")
    return pd.DataFrame({"hour": hour, "ambient_light_lux": ambient.round(1),
                         "motion_detected": motion, "lighting_mode": mode})

df = make_dataset()
X = df[["hour", "ambient_light_lux", "motion_detected"]]
y = df["lighting_mode"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

st.subheader("Live demo — change the sensor values")
c1, c2, c3 = st.columns(3)
with c1:
    hour = st.slider("Hour of day", 0, 23, 20)
with c2:
    lux = st.slider("Ambient light (lux)", 0, 600, 60)
with c3:
    motion = st.selectbox("Motion detected?", ["No", "Yes"])
motion_value = 1 if motion == "Yes" else 0
sample = pd.DataFrame([{"hour": hour, "ambient_light_lux": lux, "motion_detected": motion_value}])
mode = model.predict(sample)[0]
brightness = {"OFF": 0, "DIM": 35, "HIGH": 100}[mode]
power_w = 0 if mode == "OFF" else (12 if mode == "DIM" else 30)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Predicted lighting mode", mode)
m2.metric("LED brightness", f"{brightness}%")
m3.metric("Estimated power", f"{power_w} W")
m4.metric("Test accuracy", f"{accuracy*100:.1f}%")

if mode == "OFF":
    st.info("Daylight detected: keep the LED switched OFF.")
elif mode == "DIM":
    st.warning("Dark conditions with no motion: use DIM lighting.")
else:
    st.success("Motion detected in dark conditions: use HIGH lighting.")

st.subheader("Dataset overview")
st.dataframe(df.head(12), use_container_width=True)
st.write("Lighting-mode counts")
counts = df["lighting_mode"].value_counts()
st.bar_chart(counts)

st.subheader("Model evaluation")
st.write(f"Accuracy on held-out test data: **{accuracy:.3f}**")
st.text("Classification report:\n" + classification_report(y_test, pred, zero_division=0))
fig, ax = plt.subplots()
cm = confusion_matrix(y_test, pred, labels=["OFF", "DIM", "HIGH"])
ax.imshow(cm)
ax.set_xticks(range(3), ["OFF", "DIM", "HIGH"])
ax.set_yticks(range(3), ["OFF", "DIM", "HIGH"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center")
st.pyplot(fig)

st.subheader("Estimated energy comparison")
st.write("Assumption for demonstration: 12-hour night, one 30 W LED; baseline runs at full power for all 12 hours.")
night_hours = 12
baseline_kwh = 30 * night_hours / 1000
# Simulated smart policy: 25% of time HIGH, 75% DIM at 12 W
smart_kwh = (30 * 0.25 * night_hours + 12 * 0.75 * night_hours) / 1000
savings = (baseline_kwh - smart_kwh) / baseline_kwh * 100
e1, e2, e3 = st.columns(3)
e1.metric("Baseline energy / night", f"{baseline_kwh:.3f} kWh")
e2.metric("Smart energy / night", f"{smart_kwh:.3f} kWh")
e3.metric("Estimated energy saved", f"{savings:.1f}%")
st.caption("These energy values are assumption-based estimates, not measured field results.")
