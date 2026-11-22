"""ISL recognition website. Matches the hand pose in a photo against YOUR reference pictures
(isl_reference.jpg and any ref_<LETTER>*.jpg files). No Kaggle model or data is used."""
import os, re, string
import numpy as np, cv2, streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from hand_detector import get_detector, detect, detect_robust, _pad

st.set_page_config(page_title="ISL Recognition", page_icon="🤟", layout="centered")
HERE = os.path.dirname(os.path.abspath(__file__))
SHEET = os.path.join(HERE, "isl_reference.jpg")
COLS, TIPS = 6, [4, 8, 12, 16, 20]
LETTERS = list(string.ascii_uppercase)


# ---------------- pose features (scale / position independent) ----------------
def hand_feats(P):
    """P: (21,2) landmark pixels of one hand -> 64 numbers."""
    rel = P - P[0]
    rel = rel / (np.linalg.norm(rel, axis=1).max() + 1e-6)
    tip_wrist = [np.linalg.norm(rel[t]) for t in TIPS]
    tip_tip = [np.linalg.norm(rel[a] - rel[b]) for i, a in enumerate(TIPS) for b in TIPS[i + 1:]]
    angles = []
    for a, b, c in [(1, 2, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20), (5, 9, 13), (9, 13, 17)]:
        v1, v2 = rel[a] - rel[b], rel[c] - rel[b]
        cos = v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angles.append(np.arccos(np.clip(cos, -1, 1)))
    return np.concatenate([rel.ravel(), tip_wrist, tip_tip, angles])


def pose_features(hands):
    """hands: list of 1-2 arrays (21,2). Returns one fixed-length vector."""
    hands = sorted(hands, key=lambda P: P[0, 0])[:2]
    n = len(hands)
    per = [hand_feats(P) for P in hands] + [np.zeros(64)] * (2 - n)
    inter = np.zeros(27)
    if n == 2:
        allp = np.vstack(hands)
        scale = np.linalg.norm(allp - allp.mean(0), axis=1).max() + 1e-6
        inter[:2] = (hands[1][0] - hands[0][0]) / scale
        inter[2:] = [np.linalg.norm(hands[0][a] - hands[1][b]) / scale for a in TIPS for b in TIPS]
    return np.concatenate(per + [inter, [n]])


def mirror(hands):
    return [H * np.array([-1.0, 1.0]) for H in hands]


def augment(hands, rng, n=40):
    """Make n slightly rotated / stretched / jittered copies (hand far, close or tilted)."""
    allp = np.vstack(hands)
    c = allp.mean(0)
    ext = (allp.max(0) - allp.min(0)).max()
    out = []
    for _ in range(n):
        t = np.deg2rad(rng.uniform(-30, 30))
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        S = rng.uniform(0.85, 1.15, 2)
        out.append(pose_features([((H - c) @ R.T) * S + c + rng.normal(0, 0.015 * ext, H.shape) for H in hands]))
    return out


def readings(detector, rgb):
    """Every hand reading of a picture (plain, then several border sizes). Returns a list of hand-sets.
    The detector is unsure on tight pictures, so several readings are kept and averaged later."""
    h, w = rgb.shape[:2]
    out = []
    for scale in (None, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3):
        im, x0, y0, sx, sy = (rgb, 0, 0, w, h) if scale is None else (lambda c, a, b, S: (c, a, b, S, S))(*_pad(rgb, scale))
        r = detect(detector, im)
        if r:
            out.append([np.array([[p.x * sx - x0, p.y * sy - y0] for p in H]) for H in r])
    if not out:                                              # last resort: mirrored copy / zoomed windows
        found = detect_robust(detector, rgb)
        if found:
            out.append([np.array([[p.x * w, p.y * h] for p in H]) for H in found])
    return out


def predict(clf, reads):
    """Average the match probabilities over all readings."""
    return np.mean([clf.predict_proba([pose_features(hs)])[0] for hs in reads], axis=0)


# ---------------- reference pictures -> classifier ----------------
def load_sheet():
    return Image.open(SHEET).convert("RGB") if os.path.exists(SHEET) else None


@st.cache_resource(show_spinner="Reading the reference pictures...")
def build_matcher():
    detector = get_detector(num_hands=2, conf=0.2)
    sheet = load_sheet()
    refs = []
    if sheet is not None:
        T = sheet.width // COLS
        for k, L in enumerate(LETTERS):
            r, c = divmod(k, COLS)
            refs.append((L, np.array(sheet.crop((c * T, r * T, (c + 1) * T, (r + 1) * T)))))
    for f in sorted(os.listdir(HERE)):                       # extra pictures you upload: ref_A.jpg, ref_A_2.png ...
        m = re.match(r"ref_([A-Za-z])", f)
        if m and f.lower().endswith((".jpg", ".jpeg", ".png")):
            img = ImageOps.exif_transpose(Image.open(os.path.join(HERE, f))).convert("RGB")
            refs.append((m.group(1).upper(), np.array(img)))
    rng = np.random.default_rng(0)
    X, y, counts, tried = [], [], {}, set()
    for L, img in refs:
        tried.add(L)
        reads = readings(detector, img)
        if not reads:
            continue
        counts[L] = counts.get(L, 0) + 1
        for hands in reads:
            for hs in (hands, mirror(hands)):
                X.append(pose_features(hs)); y.append(L)
                for v in augment(hs, rng, 10):
                    X.append(v); y.append(L)
    if len(set(y)) < 2:
        return detector, None, counts, sorted(tried)
    clf = make_pipeline(StandardScaler(), SVC(C=10, probability=True, random_state=0)).fit(np.array(X), y)
    return detector, clf, counts, sorted(tried - set(counts))


def labeled_sheet(sheet):
    img = sheet.copy()
    d = ImageDraw.Draw(img)
    T = img.width // COLS
    try:
        font = ImageFont.load_default(size=34)
    except TypeError:
        font = ImageFont.load_default()
    for k, L in enumerate(LETTERS):
        r, c = divmod(k, COLS)
        d.rectangle([c * T, r * T, c * T + 52, r * T + 50], fill="white")
        d.text((c * T + 10, r * T + 4), L, fill="black", font=font)
    return img


def draw_hands(rgb, hands):
    img = rgb.copy()
    rad = max(2, img.shape[1] // 120)
    for H in hands:
        for x, y in H:
            cv2.circle(img, (int(x), int(y)), rad, (255, 60, 60), -1)
    s = 480 / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s) if s < 1 else img


detector, clf, counts, failed = build_matcher()
page = st.sidebar.radio("Pages", ["Recognize", "How to sign", "Supported signs", "About"])

if page == "Recognize":
    st.title("🤟 Indian Sign Language Recognition")
    st.caption("The app compares your hand pose with the reference pictures on the 'How to sign' page. "
               "Copy a pose from there, with both hands fully in view.")
    if clf is None:
        st.error("The reference pictures could not be read (isl_reference.jpg missing or no hands found in it).")
        st.stop()
    thr = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.5)
    src = st.radio("Input", ["Webcam", "Upload image"], horizontal=True)
    f = st.camera_input("Show a hand sign") if src == "Webcam" else st.file_uploader("Upload", type=["jpg", "jpeg", "png"])
    if f:
        rgb = np.array(ImageOps.exif_transpose(Image.open(f)).convert("RGB"))   # fixes phone-rotated photos
        reads = readings(detector, rgb)
        if not reads:
            st.error("No hand detected. Use a plain background and good lighting, and keep the hands in view.")
        else:
            hands = max(reads, key=len)
            p = predict(clf, reads)
            top = p.argsort()[::-1][:3]
            c1, c2 = st.columns(2)
            c1.image(draw_hands(rgb, hands), caption=f"What the app saw ({len(hands)} hand{'s' if len(hands) > 1 else ''})", width=260)
            if p[top[0]] < thr:
                c2.warning(f"Uncertain ({p[top[0]]*100:.0f}%). Try again.")
            else:
                c2.success(f"Sign: **{clf.classes_[top[0]]}**  ({p[top[0]]*100:.1f}%)")
            st.write("Top 3 matches")
            for i in top:
                st.progress(float(p[i]), text=f"{clf.classes_[i]}: {p[i]*100:.1f}%")

elif page == "How to sign":
    st.title("How to sign each letter")
    st.write("Indian Sign Language alphabet poses. Copy them as closely as you can, keep both hands fully "
             "in the frame, and use a plain background.")
    sheet = load_sheet()
    if sheet is None:
        st.info("Reference sheet (isl_reference.jpg) not found in the repository.")
    else:
        T = sheet.width // COLS
        L = st.selectbox("Pick a letter", LETTERS)
        r, c = divmod(LETTERS.index(L), COLS)
        st.image(sheet.crop((c * T, r * T, (c + 1) * T, (r + 1) * T)), caption=f"Letter {L}", width=360)
        st.subheader("All letters")
        st.image(labeled_sheet(sheet))

elif page == "Supported signs":
    st.title("Supported signs")
    ok = sorted(counts)
    st.write(f"{len(ok)} letters can be recognised: " + ", ".join(ok))
    if failed:
        st.warning("No hand could be read in the reference picture for: " + ", ".join(failed) +
                   ". Upload clearer pictures named like ref_" + failed[0] + ".jpg to add them.")
    st.info("Only static signs are supported, not moving gestures or full sentences. "
            "To improve a letter, add more pictures of it named ref_<LETTER>.jpg, ref_<LETTER>_2.jpg ... "
            "in the GitHub repository (different distance, angle, background).")

else:
    st.title("About")
    st.write("Capstone: An Accessible Vision-Based Indian Sign Language Alphabet Recognition Framework "
             "Using Hand Region Classification.")
    st.write("Pipeline: MediaPipe finds the hands, their 21 landmark points per hand become a pose description "
             "(independent of size and position), and the pose is matched against the reference pictures. "
             "No external training dataset is used.")
