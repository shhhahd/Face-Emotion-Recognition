import av
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from huggingface_hub import hf_hub_download
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

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
    return hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)


@st.cache_resource
def load_model(weights_path):
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
    face = cv2.resize(gray_face, (BASE_SIZE, BASE_SIZE))
    face = face.astype(np.float32) / 255.0
    img = np.stack([face, face, face], axis=0)
    tensor = torch.from_numpy(img).float().unsqueeze(0)
    tensor = F.interpolate(tensor, size=(IMG_SIZE, IMG_SIZE),
                           mode="bilinear", align_corners=False)
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
        if prev_probs is not None:
            probs = smooth * prev_probs + (1 - smooth) * probs

        idx = int(np.argmax(probs))
        label = f"{EMOTIONS[idx]} ({probs[idx] * 100:.0f}%)"
        color = COLORS[EMOTIONS[idx]]
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame_bgr, label, (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    return frame_bgr, (probs if probs is not None else prev_probs)


# ============ Live camera processor ============
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model(get_weights_path())
        self.detector = load_face_detector()
        self.prev = None
        self.smooth = 0.5

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img, self.prev = process_frame(img, self.model, self.detector, self.prev, self.smooth)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ============ UI ============
st.sidebar.header("Settings")
smooth = st.sidebar.slider("Result smoothing", 0.0, 0.9, 0.5, 0.1)

# Download the model once up front so the camera starts without delay
with st.spinner("Downloading the model from Hugging Face..."):
    try:
        get_weights_path()
    except Exception as e:
        st.error(f"Failed to download the model: {e}")
        st.stop()

ctx = webrtc_streamer(
    key="emotion",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=EmotionProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.video_processor:
    ctx.video_processor.smooth = smooth

st.caption("Click START, then allow camera access in your browser.")
