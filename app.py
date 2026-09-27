import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import streamlit as st
from facenet_pytorch import InceptionResnetV1

# 1. إعدادات الصفحة
st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="🎭",
    layout="centered"
)

# 2. تحديد الفئات والتسميات الإنجليزي والرموز التعبيرية
EMOTIONS = {
    0: "Angry 😡",
    1: "Disgust 🤢",
    2: "Fear 😨",
    3: "Happy 😄",
    4: "Neutral 😐",
    5: "Sad 😢",
    6: "Surprise 😲"
}
NUM_CLASSES = len(EMOTIONS)
IMG_SIZE = 160  # الحجم القياسي المعتمد لـ InceptionResnetV1

# 3. دالة بناء وتنسيق الموديل بنفس هيكل التدريب تماماً
def build_model():
    model = InceptionResnetV1(pretrained=None, classify=False)
    model.logits = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512, NUM_CLASSES)
    )
    model.classify = True
    return model

# 4. دالة تحميل الموديل والأوزان
@st.cache_resource
def load_emotion_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    
    model_path = "../models/emotion_model_facenet_final.pth"
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, device
    else:
        st.error(f"File '{model_path}' not found in current directory.")
        return None, device

# 5. دالة معالجة الصورة المرفوعة لتطابق كلاس FERDataset تماماً
def preprocess_image(image_pil, img_size=IMG_SIZE):
    # تحويل الصورة إلى رمادي ثم تحويلها لـ numpy array
    img_gray = np.array(image_pil.convert('L'))
    
    # التطبيق النمطي المماثل لـ FERDataset
    if img_gray.max() > 1.0:
        img = img_gray.astype(np.float32) / 255.0
    else:
        img = img_gray.astype(np.float32)

    # تحويل 1 channel إلى 3 channels (RGB)
    img_3ch = np.stack([img, img, img], axis=0)
    tensor = torch.from_numpy(img_3ch).float()
    
    # تغيير الحجم باستخدام Bilinear Interpolation
    tensor = F.interpolate(
        tensor.unsqueeze(0), 
        size=(img_size, img_size),
        mode="bilinear", 
        align_corners=False
    )
    
    # تطبيع القيم Scaling إلى نطاق [-1, 1]
    tensor = (tensor - 0.5) / 0.5
    return tensor

# ---- الواجهة والتفاعل ----
st.title("🎭 Facial Emotion Recognition")
st.write("Upload a face image to predict the emotion using InceptionResnetV1.")

model, device = load_emotion_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # قراءة الصورة وعرضها
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('Predict Emotion 🔍'):
        with st.spinner('Analyzing facial expression...'):
            # التجهيز وحساب التوقع
            input_tensor = preprocess_image(image).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]
                
                confidence, predicted_idx = torch.max(probabilities, dim=0)
                
                predicted_label = EMOTIONS[predicted_idx.item()]
                confidence_pct = confidence.item() * 100

        st.divider()
        
        # عرض النتيجة الرئيسية
        st.subheader("🎯 Result:")
        st.success(f"**Predicted Emotion:** {predicted_label}")
        st.info(f"**Confidence:** {confidence_pct:.2f}%")
        st.progress(float(confidence.item()))

        # عرض تفاصيل احتمالات باقي المشاعر
        with st.expander("Show probability breakdown for all classes"):
            for idx, emotion_name in EMOTIONS.items():
                prob = probabilities[idx].item() * 100
                st.write(f"- **{emotion_name}:** {prob:.2f}%")