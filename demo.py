import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

# --- Function Definitions ---

def dice_coef(y_true, y_pred, smooth=1):
    pred = tf.argmax(y_pred, axis=-1)[..., tf.newaxis]
    y_true = tf.cast(y_true, tf.float32)
    pred = tf.cast(pred, tf.float32)
    inter = tf.reduce_sum(y_true * pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(pred, axis=[1, 2, 3])
    return tf.reduce_mean((2. * inter + smooth) / (union + smooth))

def prepare_image(uploaded_bytes):
    img = Image.open(BytesIO(uploaded_bytes)).convert("L").resize((256, 256))
    arr = np.array(img, dtype=np.float32) / 1024.0
    return arr.reshape((1, 256, 256, 1)), img

def create_mask(pred_mask):
    mask = tf.argmax(pred_mask, axis=-1)[0]
    return Image.fromarray((mask.numpy() * 255).astype(np.uint8))

# --- Load the model once at startup ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "Main_model_2.keras"
    )

model = load_model()

# --- Streamlit UI ---
st.title("Left Ventricle Segmentation Demo")

uploaded = st.file_uploader("Upload a cardiac MRI slice", type=["png", "jpg", "jpeg"])
if uploaded:
    img_tensor, original_img = prepare_image(uploaded.getvalue())
    st.image(original_img, caption="Input Image", use_container_width=True)

    if st.button("Segment"):
        with st.spinner("Running segmentation…"):
            pred_mask = model.predict(img_tensor)
            mask_img = create_mask(pred_mask)

        st.image(mask_img, caption="Predicted Mask", use_container_width=True)
        st.download_button(
            label="Download Mask",
            data=mask_img.tobytes(),
            file_name="lv_mask.png",
            mime="image/png"
        )
