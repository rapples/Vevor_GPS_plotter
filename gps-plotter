import streamlit as st
import cv2
from PIL import Image
import pytesseract
import re
import folium
from folium.plugins import MarkerCluster
import tempfile
import os
import webbrowser
from math import radians, sin, cos, sqrt, atan2

# important! include the max upload 4GB in your streamlit config.toml file.
#
#    mkdir -p ~/.streamlit
#    nano ~/.streamlit/config.toml
#    [server]
#     maxUploadSize = 4096


# Setup Streamlit

st.title("📍 GPS Video Map Processor")
uploaded_files = st.file_uploader("Upload video file(s)", type=["mp4"], accept_multiple_files=True)

interval = st.number_input("Sampling interval (seconds)", min_value=1, max_value=10, value=2)
run = st.button("▶️ Process Videos")

# Regex to extract GPS coordinates from OCR text
full_pattern = re.compile(
    r'N?\s*([0-9]{2})[°:\s]+([0-9]{2})[\'′:\s]+([0-9.oO]+)["”]?\s*[Nn]?[,\s]*W\s*([0-9]{2,3})[°:\s]+([0-9]{2})[\'′:\s]+([0-9.oO]+)["”]?',
    re.IGNORECASE
)

def dms_to_decimal(deg, min_, sec):
    return round(float(deg) + float(min_) / 60 + float(sec) / 3600, 6)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

if run and uploaded_files:
    gps_results = []
    cumulative_time = 0
    last_valid_point = None

    progress_text = st.empty()
    progress_bar = st.progress(0)

    for i, uploaded_file in enumerate(uploaded_files):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
                st.warning(f"⚠️ Unable to read FPS for {uploaded_file.name}, skipping.")
                continue
        duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
	#fps = cap.get(cv2.CAP_PROP_FPS)
        #duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
        total_steps = len(uploaded_files) * (duration // interval + 1)
        step = 0

        for t in range(0, duration, interval):
            step += 1
            progress_text.text(f"Processing {uploaded_file.name}, time {t}s")
            progress_bar.progress(min(step / total_steps, 1.0))

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ret, frame = cap.read()
            if not ret:
                continue

            h, w, _ = frame.shape
            cropped = frame[int(h * 0.70):, int(w * 0.65):]
            pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            text = pytesseract.image_to_string(pil_img)
            text = text.replace("o", "0").replace("O", "0")

            match = full_pattern.search(text)
            if match:
                try:
                    lat = dms_to_decimal(*match.groups()[0:3])
                    lon = dms_to_decimal(*match.groups()[3:6]) * -1
                    timestamp = cumulative_time + t

                    if not (0 <= lat <= 90 and -180 <= lon <= 0):
                        continue

                    if last_valid_point:
                        _, prev_lat, prev_lon = last_valid_point
                        dist = haversine(prev_lat, prev_lon, lat, lon)
                        dt = timestamp - last_valid_point[0]
                        if dt > 0 and (dist / dt) > 10:
                            continue

                    gps_results.append((timestamp, lat, lon))
                    last_valid_point = (timestamp, lat, lon)

                except ValueError:
                    continue

        cap.release()
        cumulative_time += duration

    # Velocity calculations
    velocities = []
    for i in range(1, len(gps_results)):
        t1, lat1, lon1 = gps_results[i - 1]
        t2, lat2, lon2 = gps_results[i]
        dt = t2 - t1
        dist = haversine(lat1, lon1, lat2, lon2)
        mps = dist / dt if dt else 0
        velocities.append(mps * 2.23694)
    velocities = [0] + velocities

    avg_velocities = []
    for i in range(len(gps_results)):
        t0 = gps_results[i][0]
        j = i
        while j > 0 and t0 - gps_results[j][0] <= 15:
            j -= 1
        count = i - j
        avg = sum(velocities[j+1:i+1]) / count if count else 0
        avg_velocities.append(avg)

    if gps_results:
        m = folium.Map(location=(gps_results[0][1], gps_results[0][2]), zoom_start=15)
        marker_cluster = MarkerCluster().add_to(m)

        for i in range(1, len(gps_results)):
            _, lat1, lon1 = gps_results[i - 1]
            _, lat2, lon2 = gps_results[i]
            speed = velocities[i]

            if speed < 10:
                color = "green"
            elif speed < 20:
                color = "yellow"
            else:
                color = "red"

            folium.PolyLine([(lat1, lon1), (lat2, lon2)], color=color, weight=4).add_to(m)

        for i, (t, lat, lon) in enumerate(gps_results):
            folium.Marker(
                location=(lat, lon),
                popup=(f"Time: {t}s\nLat: {lat}\nLon: {lon}\nSpeed: {velocities[i]:.2f} mph\nAvg (15s): {avg_velocities[i]:.2f} mph"),
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(marker_cluster)

        html_path = os.path.join(tempfile.gettempdir(), "streamlit_gps_map.html")
        m.save(html_path)
        st.success("✅ Map saved successfully.")
        st.markdown(f"[Click here to open the map](file://{html_path})", unsafe_allow_html=True)
        webbrowser.open_new_tab(f"file://{html_path}")
    else:
        st.error("❌ No valid GPS data found.")
