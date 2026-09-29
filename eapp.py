import os
import tempfile

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from huggingface_hub import hf_hub_download

# ============ Settings ============
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(EMOTIONS)
IMG_SIZE = 160          # InceptionResnetV1 input size
BASE_SIZE = 48          # Original size the data was prepared with (FER2013)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLORS = {  # BGR
    "angry": (0, 0, 255), "disgust": (0, 128, 0), "fear": (128, 0, 128),
    "happy": (0, 200, 0), "neutral": (200, 200, 200), "sad": (255, 0, 0),
    "surprise": (0, 165, 255),
}

HF_REPO_ID = "ShahdAmr2004/emotion-facenet"
HF_FILENAME = "emotion_model_facenet_final.pth"

st.set_page_config(page_title="Emotion Recognition", layout="centered")
st.title("😀 Facial Emotion Recognition")


# ============ Model loading (once) ============
@st.cache_resource
def get_weights_path():
    # Downloads the file once and caches it; later calls return the local path immediately
    return hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)


@st.cache_resource
def load_model(weights_path):
    # pretrained=None so we don't download the VGGFace2 weights again; we load our own weights
    model = InceptionResnetV1(pretrained=None, classify=False)
    model.logits = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, NUM_CLASSES))
    model.classify = True
    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


@st.cache_resource
def load_face_detector():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(path)


# ============ Same preprocessing as training ============
def preprocess_face(gray_face):
    face = cv2.resize(gray_face, (BASE_SIZE, BASE_SIZE))          # 48x48
    face = face.astype(np.float32) / 255.0                         # 0-1
    img = np.stack([face, face, face], axis=0)                     # (3,48,48)
    tensor = torch.from_numpy(img).float().unsqueeze(0)
    tensor = F.interpolate(tensor, size=(IMG_SIZE, IMG_SIZE),
                           mode="bilinear", align_corners=False)   # 160x160
    tensor = (tensor - 0.5) / 0.5
    return tensor.to(DEVICE)


@torch.no_grad()
def predict_probs(model, gray_face):
    out = model(preprocess_face(gray_face))
    return torch.softmax(out, dim=1)[0].cpu().numpy()


def process_frame(frame_bgr, model, detector, prev_probs, smooth):
    """Detects faces, classifies the emotion, draws the result. Returns (frame, last probs)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

    probs = None
    for (x, y, w, h) in faces:
        probs = predict_probs(model, gray[y:y + h, x:x + w])
        # Simple smoothing so the label doesn't jump between frames
        if prev_probs is not None:
            probs = smooth * prev_probs + (1 - smooth) * probs

        idx = int(np.argmax(probs))
        label = f"{EMOTIONS[idx]} ({probs[idx] * 100:.0f}%)"
        color = COLORS[EMOTIONS[idx]]
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame_bgr, label, (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    return frame_bgr, (probs if probs is not None else prev_probs)


def show_probs(placeholder, probs):
    if probs is None:
        placeholder.info("No face detected in the frame.")
        return
    placeholder.bar_chart({e: float(p) for e, p in zip(EMOTIONS, probs)})


# ============ UI ============
st.sidebar.header("Settings")
mode = st.sidebar.radio("Source", ["Camera (live)", "Upload video"])
smooth = st.sidebar.slider("Result smoothing", 0.0, 0.9, 0.5, 0.1)

with st.spinner("Downloading the model from Hugging Face..."):
    try:
        weights_path = get_weights_path()
    except Exception as e:
        st.error(f"Failed to download the model: {e}")
        st.stop()

model = load_model(weights_path)
detector = load_face_detector()

frame_box = st.empty()
probs_box = st.empty()

if mode == "Camera (live)":
    run = st.sidebar.checkbox("Start camera")
    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Unable to open the camera. Make sure it's connected and not in use by another program.")
        else:
            prev = None
            while run:
                ok, frame = cap.read()
                if not ok:
                    st.error("Problem reading a frame from the camera.")
                    break
                frame, prev = process_frame(frame, model, detector, prev, smooth)
                frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                show_probs(probs_box, prev)
            cap.release()
    else:
        st.info("Enable 'Start camera' from the sidebar.")

else:
    skip = st.sidebar.slider("Process every N-th frame", 1, 10, 2)
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        prev, i = None, 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i % skip:
                continue
            frame, prev = process_frame(frame, model, detector, prev, smooth)
            frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            show_probs(probs_box, prev)
        cap.release()
        os.unlink(tfile.name)
        st.success("Video finished.")
    else:
        st.info("Upload a video to start the analysis.")
