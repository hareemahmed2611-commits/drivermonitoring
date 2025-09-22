import streamlit as st
import cv2
from ultralytics import YOLO
import time
from streamlit_lottie import st_lottie
import json
import base64

# ------------------ Load YOLO Model ------------------
model = YOLO("best.pt")

st.set_page_config(page_title="Driver Monitoring Dashboard")
st.title("🚗 Real-time Driver Monitoring System")

# ------------------ Load Lottie Animation ------------------
with open("driving.json", "r") as f:
    lottie_animation = json.load(f)

st_lottie(lottie_animation, speed=1, height=500, width=800)

# ------------------ Project Description ------------------
st.markdown("""
### 📝 Project Overview
This Real-time Driver Monitoring System improves **road safety** by detecting risky behaviors:
- **Drowsiness detection**: Monitors if the driver closes eyes for extended periods.
- **Seatbelt monitoring**: Alerts if the driver is not wearing a seatbelt.
- **Smoking detection**: Detects if the driver is smoking while driving.
- **Phone usage detection**: Alerts if the driver is using a phone.

**Benefits:**
- Reduces accidents caused by distracted or drowsy driving.
- Provides real-time alerts to prevent risky driving behaviors.
- Can be deployed in vehicles for fleet safety management.
""")

# ------------------ Webcam Feed ------------------
st.markdown("### 🎥 Live Camera Feed")
FRAME_WINDOW = st.image([])

# ------------------ Alert Placeholders ------------------
alert_display = st.empty()

# ------------------ Session State ------------------
if 'eyes_timer' not in st.session_state: st.session_state.eyes_timer = 0
if 'seatbelt_timer' not in st.session_state: st.session_state.seatbelt_timer = 0
if 'cig_timer' not in st.session_state: st.session_state.cig_timer = 0
if 'phone_timer' not in st.session_state: st.session_state.phone_timer = 0
if 'prev_time' not in st.session_state: st.session_state.prev_time = time.time()
if 'monitoring' not in st.session_state: st.session_state.monitoring = False
if 'alert_played' not in st.session_state:
    st.session_state.alert_played = {'eyes': False, 'seatbelt': False, 'cig': False, 'phone': False}

# ------------------ Thresholds ------------------
CLOSED_EYE_THRESHOLD = 0.1
EYE_ALERT_SEC = 5
SEATBELT_ALERT_SEC = 5
CIG_ALERT_SEC = 3
PHONE_ALERT_SEC = 3
PHONE_THRESHOLD = 0.1

# ------------------ Audio Alert ------------------
def play_alert_sound():
    try:
        audio_file = "emergency-alarm-69780.mp3"
        audio_html = f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{base64.b64encode(open(audio_file, "rb").read()).decode()}" type="audio/mp3">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    except:
        pass

# ------------------ Start/Stop Monitoring Button ------------------
def toggle_monitoring():
    st.session_state.monitoring = not st.session_state.monitoring

left, mid_c, mid, mid_l, right = st.columns(5)
with mid:
    btn_label = "Stop Monitoring" if st.session_state.monitoring else "Start Monitoring"
    st.button(btn_label, on_click=toggle_monitoring)

    # ------------------ Run Detection ------------------
    if st.session_state.monitoring:
        cap = cv2.VideoCapture(0)

        while st.session_state.monitoring:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to grab frame")
                break

            # Increase camera feed size by resizing frame
            frame = cv2.resize(frame, (960, 540))  # adjust as needed
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb, verbose=False)[0]

            labels = results.names
            boxes = results.boxes

            # ------------------ Detection Scores ------------------
            scores = {
                "eyes_timer": 0.0,
                "seatbelt_timer": 0.0,
                "cig_timer": 0.0,
                "phone_timer": 0.0
            }

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = labels[cls_id].lower()
                if "closed eyes" in name:
                    scores["eyes_timer"] = max(scores["eyes_timer"], conf)
                elif "seatbelt" in name:
                    scores["seatbelt_timer"] = max(scores["seatbelt_timer"], conf)
                elif "cigarette" in name:
                    scores["cig_timer"] = max(scores["cig_timer"], conf)
                elif "phone" in name:
                    scores["phone_timer"] = max(scores["phone_timer"], conf)

            # ------------------ Update Timers ------------------
            current_time = time.time()
            dt = current_time - st.session_state.prev_time
            st.session_state.prev_time = current_time

            st.session_state.eyes_timer = st.session_state.eyes_timer + dt if scores["eyes_timer"] > CLOSED_EYE_THRESHOLD else 0
            st.session_state.seatbelt_timer = st.session_state.seatbelt_timer + dt if scores["seatbelt_timer"] < 0.5 else 0
            st.session_state.cig_timer = st.session_state.cig_timer + dt if scores["cig_timer"] > 0.5 else 0
            st.session_state.phone_timer = st.session_state.phone_timer + dt if scores["phone_timer"] > PHONE_THRESHOLD else 0

            # ------------------ Display Prominent Alerts ------------------
            alert_texts = []
            if st.session_state.eyes_timer > EYE_ALERT_SEC:
                alert_texts.append("😴 Drowsy Driver Detected!")
                if not st.session_state.alert_played['eyes']:
                    play_alert_sound()
                    st.session_state.alert_played['eyes'] = True
            else:
                st.session_state.alert_played['eyes'] = False

            if st.session_state.seatbelt_timer > SEATBELT_ALERT_SEC:
                alert_texts.append("⚠️ Seatbelt Not Detected!")
                if not st.session_state.alert_played['seatbelt']:
                    play_alert_sound()
                    st.session_state.alert_played['seatbelt'] = True
            else:
                st.session_state.alert_played['seatbelt'] = False

            if st.session_state.cig_timer > CIG_ALERT_SEC:
                alert_texts.append("🚬 Smoking Detected!")
                if not st.session_state.alert_played['cig']:
                    play_alert_sound()
                    st.session_state.alert_played['cig'] = True
            else:
                st.session_state.alert_played['cig'] = False

            if st.session_state.phone_timer > PHONE_ALERT_SEC:
                alert_texts.append("📱 Phone Usage Detected!")
                if not st.session_state.alert_played['phone']:
                    play_alert_sound()
                    st.session_state.alert_played['phone'] = True
            else:
                st.session_state.alert_played['phone'] = False

            # Center-align alert text
            if alert_texts:
                alert_display.markdown(
                    "<div style='text-align:center; font-size:32px; font-weight:bold; color:red;'>"
                    + "<br>".join(alert_texts) + "</div>", unsafe_allow_html=True
                )
            else:
                alert_display.markdown(
                    "<div style='text-align:center; font-size:24px; font-weight:bold; color:green;'>✅ All Normal</div>",
                    unsafe_allow_html=True
                )

            # ------------------ Annotate and Display Frame ------------------
            annotated_frame = results.plot()
            FRAME_WINDOW.image(annotated_frame)

        cap.release()
