import os
import tempfile

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

# ============ الإعدادات ============
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(EMOTIONS)
IMG_SIZE = 160          # مقاس InceptionResnetV1
BASE_SIZE = 48          # المقاس الأصلي اللي الداتا اتحضرت بيه (FER2013)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLORS = {  # BGR
    "angry": (0, 0, 255), "disgust": (0, 128, 0), "fear": (128, 0, 128),
    "happy": (0, 200, 0), "neutral": (200, 200, 200), "sad": (255, 0, 0),
    "surprise": (0, 165, 255),
}

st.set_page_config(page_title="Emotion Recognition", layout="centered")
st.title("😀 كشف تعابير الوجه")


# ============ تحميل الموديل (مرة واحدة) ============
@st.cache_resource
def load_model(weights_path):
    # pretrained=None عشان مانحملش أوزان VGGFace2 تاني، هنحمّل أوزانك إنت
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


# ============ نفس الـ preprocessing بتاع التدريب ============
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
    """يكتشف الوجوه، يصنف التعبير، يرسم النتيجة. يرجع (الفريم، آخر probs)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

    probs = None
    for (x, y, w, h) in faces:
        probs = predict_probs(model, gray[y:y + h, x:x + w])
        # تنعيم بسيط عشان اللابل ما يفضلش يتنطط بين فريم والتاني
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
        placeholder.info("مفيش وجه ظاهر في الفريم.")
        return
    placeholder.bar_chart({e: float(p) for e, p in zip(EMOTIONS, probs)})


# ============ الواجهة ============
st.sidebar.header("الإعدادات")
weights_path = st.sidebar.text_input("مسار ملف الموديل", "emotion_model_facenet_final.pth")
mode = st.sidebar.radio("المصدر", ["الكاميرا (لايف)", "رفع فيديو"])
smooth = st.sidebar.slider("تنعيم النتيجة", 0.0, 0.9, 0.5, 0.1)

if not os.path.exists(weights_path):
    st.error(f"ملف الموديل مش موجود: {weights_path}")
    st.stop()

model = load_model(weights_path)
detector = load_face_detector()

frame_box = st.empty()
probs_box = st.empty()

if mode == "الكاميرا (لايف)":
    run = st.sidebar.checkbox("تشغيل الكاميرا")
    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("مش قادر أفتح الكاميرا. تأكد إنها متوصلة ومش مستخدمة في برنامج تاني.")
        else:
            prev = None
            while run:
                ok, frame = cap.read()
                if not ok:
                    st.error("مشكلة في قراءة الفريم من الكاميرا.")
                    break
                frame, prev = process_frame(frame, model, detector, prev, smooth)
                frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                show_probs(probs_box, prev)
            cap.release()
    else:
        st.info("فعّل 'تشغيل الكاميرا' من القائمة الجانبية.")

else:
    skip = st.sidebar.slider("عالج فريم كل كام فريم", 1, 10, 2)
    video_file = st.file_uploader("ارفع فيديو", type=["mp4", "mov", "avi", "mkv"])
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
        st.success("خلص الفيديو.")
    else:
        st.info("ارفع فيديو عشان يبدأ التحليل.")