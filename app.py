import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

st.set_page_config(page_title="Damage, Cost & Fraud", page_icon="🚗", layout="wide")
st.title("Vehicle Damage Detection, Cost Estimation & Fraud Checks 🚗🛠️🔍")

# -------------------------
# Sidebar controls
# -------------------------
model_option = st.sidebar.selectbox(
    "Select Model",
    ("best.pt", "best_new.pt", "best_git.pt"),
    index=0
)
conf_thres = st.sidebar.slider("Confidence Threshold", 0.05, 0.9, 0.25, 0.05)
show_table = st.sidebar.checkbox("Show detections table", value=True)

# -------------------------
# Load YOLO model (cached)
# -------------------------
@st.cache_resource
def load_model(model_name: str):
    return YOLO(model_name)

model = load_model(model_option)

# -------------------------
# Cost map per detected class
# -------------------------
REPAIR_COSTS = {
    "Broken light": 200,
    "Broken mirror": 150,
    "Cracked bumper": 500,
    "Dent": 300,
    "Scratch": 100,
    "Shattered glass": 400,
    "Tire puncture": 120,
}

# -------------------------
# Fraud rules
# -------------------------
def fraud_check(damages, total_cost):
    flags = []

    # --- Global rules ---
    if total_cost > 2000:
        flags.append("⚠️ Unusually high overall repair cost detected.")

    if "Shattered glass" in damages and "Dent" not in damages:
        flags.append("⚠️ Shattered glass without dents may be suspicious.")

    if len(damages) == 0 and total_cost > 0:
        flags.append("⚠️ Cost estimated but no damages detected.")

    counts = {}
    for d in damages:
        counts[d] = counts.get(d, 0) + 1
    for part, c in counts.items():
        if c >= 5:
            flags.append(f"⚠️ Many '{part}' detections ({c}); please verify image/claim context.")

    # --- Class-specific rules ---
    if "Broken light" in damages and total_cost > 150:
        flags.append("⚠️ Broken light repair cost unusually high.")
    if "Broken mirror" in damages and total_cost > 800:
        flags.append("⚠️ Broken mirror repair cost unusually high.")
    if "Cracked bumper" in damages and total_cost > 4000:
        flags.append("⚠️ Cracked bumper repair cost unusually high.")
    if "Dent" in damages and total_cost > 250:
        flags.append("⚠️ Dent repair cost unusually high.")
    if "Scratch" in damages and total_cost > 1500:
        flags.append("⚠️ Scratch repair cost unusually high.")
    if "Shattered glass" in damages and total_cost > 3000:
        flags.append("⚠️ Shattered glass repair cost unusually high.")
    if "Tire puncture" in damages and total_cost > 600:
        flags.append("⚠️ Tire puncture repair cost unusually high.")

    # Combination rule
    if "Scratch" in damages and len(damages) == 1 and total_cost > 2000:
        flags.append("⚠️ Single scratch with very high cost.")

    return flags

# -------------------------
# Image uploader
# -------------------------
uploaded_file = st.file_uploader("Upload a vehicle image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Normalize orientation & convert
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        st.image(image, caption="Uploaded Vehicle Image", use_column_width=True)

        # Run YOLO
        image_np = np.array(image)
        results = model(image_np, conf=conf_thres)

        # Show detections
        st.subheader("Detection Output")
        for r in results:
            plotted_bgr = r.plot()
            plotted_rgb = plotted_bgr[:, :, ::-1]
            st.image(plotted_rgb, caption=f"Model: {model_option}", use_column_width=True)

        # Collect detections
        detections, detected_parts, total_cost = [], [], 0
        for r in results:
            names_map = r.names
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0]) if hasattr(box, "conf") else None
                cls_name = names_map.get(cls_id, str(cls_id))

                detected_parts.append(cls_name)
                total_cost += REPAIR_COSTS.get(cls_name, 0)

                xyxy = box.xyxy[0].tolist()
                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 3) if conf is not None else None,
                    "x1": round(xyxy[0], 1),
                    "y1": round(xyxy[1], 1),
                    "x2": round(xyxy[2], 1),
                    "y2": round(xyxy[3], 1),
                })

        # Show detections table
        if show_table and len(detections) > 0:
            st.subheader("Detections")
            st.dataframe(detections, use_container_width=True)

        # Cost estimation
        st.subheader("Estimated Repair Cost 💰")
        if detected_parts:
            with st.expander("Detected damage parts (counted per detection)"):
                st.write(detected_parts)
            st.success(f"Estimated Repair Cost: **${total_cost}**")
        else:
            st.info("No visible damages detected.")

        # Fraud checks
        st.subheader("Fraud Detection 🔍")
        flags = fraud_check(detected_parts, total_cost)
        if flags:
            for f in flags:
                st.error(f)
        else:
            st.success("✅ No fraud detected based on current rules.")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
else:
    st.info("Please upload an image to get started.")
