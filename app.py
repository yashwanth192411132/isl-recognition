"""Module 5: Accessible ISL recognition website (Streamlit)."""
import os, json, string, numpy as np, cv2, streamlit as st
import tensorflow as tf
from hand_detector import get_detector, detect_robust
from PIL import Image

st.set_page_config(page_title="ISL Recognition", page_icon="🤟", layout="centered")

# Find the model whether the files are in a folder called isl_work or beside app.py
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = None
for d in [os.environ.get("ISL_BASE"), os.path.join(HERE, "isl_work"), HERE, "isl_work", "."]:
    if d and os.path.exists(os.path.join(d, "isl_cnn.keras")):
        BASE = d
        break
if BASE is None:
    st.error("Model file isl_cnn.keras was not found in the repository. "
             "Upload isl_cnn.keras and cnn_labels.json to GitHub (same place as app.py).")
    st.write("Files the app can see:", sorted(os.listdir(HERE)))
    st.stop()

@st.cache_resource
def load():
    m = tf.keras.models.load_model(f"{BASE}/isl_cnn.keras")
    labels = json.load(open(f"{BASE}/cnn_labels.json"))
    hands = get_detector(num_hands=2, conf=0.2)
    return m, labels, hands
model, labels, hands = load()

def crop_hand(rgb):
    h, w = rgb.shape[:2]
    found = detect_robust(hands, rgb)
    if not found: return None
    xs = [p.x * w for H in found for p in H]
    ys = [p.y * h for H in found for p in H]
    side = max(max(xs) - min(xs), max(ys) - min(ys)) * 1.25
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    return rgb[int(max(cy - side/2, 0)):int(min(cy + side/2, h)), int(max(cx - side/2, 0)):int(min(cx + side/2, w))]

page = st.sidebar.radio("Pages", ["Recognize", "How to sign", "Supported signs", "Model performance", "About"])

if page == "Recognize":
    st.title("🤟 Indian Sign Language Recognition")
    st.caption("Tip: open the 'How to sign' page and copy the example pose exactly, with both hands fully in view.")
    thr = st.sidebar.slider("Confidence threshold", 0.3, 0.95, 0.6)
    src = st.radio("Input", ["Webcam", "Upload image"], horizontal=True)
    f = st.camera_input("Show a hand sign") if src == "Webcam" else st.file_uploader("Upload", type=["jpg", "jpeg", "png"])
    if f:
        rgb = np.array(Image.open(f).convert("RGB"))
        roi = crop_hand(rgb)
        if roi is None:
            st.error("No hand detected. Use a plain background and good lighting.")
        else:
            x = cv2.resize(roi, (224, 224)).astype("float32")
            p = model.predict(x[None], verbose=0)[0]
            top = p.argsort()[::-1][:3]
            c1, c2 = st.columns(2)
            c1.image(roi, caption="Detected hand", width=200)
            if p[top[0]] < thr:
                c2.warning(f"Uncertain ({p[top[0]]*100:.0f}%). Try again.")
            else:
                c2.success(f"Sign: **{labels[top[0]]}**  ({p[top[0]]*100:.1f}%)")
            st.write("Top 3 predictions")
            for i in top: st.progress(float(p[i]), text=f"{labels[i]}: {p[i]*100:.1f}%")

elif page == "How to sign":
    st.title("How to sign each letter")
    st.write("These example poses come from the training dataset. Copy them as closely as you can, "
             "keep both hands fully in the frame, and use a plain background.")
    ref = next((os.path.join(d, "isl_reference.jpg") for d in [HERE, BASE]
                if os.path.exists(os.path.join(d, "isl_reference.jpg"))), None)
    if ref is None:
        st.info("Reference sheet (isl_reference.jpg) not found in the repository.")
    else:
        sheet = Image.open(ref)
        T = sheet.width // 6
        L = st.selectbox("Pick a letter", [c for c in string.ascii_uppercase if c in labels])
        r, c = divmod(string.ascii_uppercase.index(L), 6)
        st.image(sheet.crop((c * T, r * T, (c + 1) * T, (r + 1) * T)), caption=f"Letter {L}", width=360)
        st.subheader("All letters")
        st.image(sheet)
        st.caption("H, J and Y are not in the dataset, so they are not supported.")

elif page == "Supported signs":
    st.title("Supported signs")
    st.write(f"{len(labels)} static signs:"); st.write(", ".join(labels))
    st.info("Only static signs are supported, not moving gestures or full sentences.")

elif page == "Model performance":
    st.title("Model performance")
    for f, t in [("cm_cnn.png", "CNN confusion matrix"), ("cm_landmark.png", "Landmark model confusion matrix"),
                 ("training_curve.png", "Training curve")]:
        if os.path.exists(f"{BASE}/{f}"): st.image(f"{BASE}/{f}", caption=t)

else:
    st.title("About")
    st.write("Capstone: An Accessible Vision-Based Indian Sign Language Alphabet Recognition Framework "
             "Using Hand Region Classification. Pipeline: MediaPipe hand ROI, then MobileNetV2 classifier.")
