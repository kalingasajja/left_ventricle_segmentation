import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose

# --- Function Definitions ---
def dice_coef(y_true, y_pred, smooth=1):
    pred = tf.argmax(y_pred, axis=-1)[..., tf.newaxis]
    y_true = tf.cast(y_true, tf.float32)
    pred   = tf.cast(pred,   tf.float32)
    inter  = tf.reduce_sum(y_true * pred, axis=[1,2,3])
    union  = tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(pred, axis=[1,2,3])
    return tf.reduce_mean((2. * inter + smooth) / (union + smooth))

def prepare_image(uploaded_bytes):
    img = Image.open(BytesIO(uploaded_bytes)).convert("L").resize((256,256))
    arr = np.array(img, dtype=np.float32) / 1024.0
    return arr.reshape((1,256,256,1)), img

def create_mask(pred_mask):
    mask = tf.argmax(pred_mask, axis=-1)[0]
    return Image.fromarray((mask.numpy() * 255).astype(np.uint8))

# --- Load model architecture + weights ---
@st.cache_resource
def load_model():
    # Rebuild your original Sequential model
    layers = [
        Conv2D(100, 5, strides=2, padding="same",
               activation=tf.nn.relu, input_shape=(256,256,1), name="Conv1"),
        MaxPool2D(pool_size=2, strides=2, padding="same"),
        Conv2D(200, 5, strides=2, padding="same", activation=tf.nn.relu),
        MaxPool2D(pool_size=2, strides=2, padding="same"),
        Conv2D(300, 3, strides=1, padding="same", activation=tf.nn.relu),
        Conv2D(300, 3, strides=1, padding="same", activation=tf.nn.relu),
        Conv2D(2,   1, strides=1, padding="same", activation=tf.nn.relu),
        Conv2DTranspose(2, 31, strides=16, padding="same")
    ]
    model = tf.keras.models.Sequential(layers)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # dice_coef can be used as a metric if desired
        metrics=[dice_coef, 'accuracy']
    )
    # Load your trained weights
    model.load_weights("Main_model_2.h5")
    return model

model = load_model()

# --- Streamlit UI ---
st.title("Left Ventricle Segmentation Demo")

uploaded = st.file_uploader("Upload a cardiac MRI slice", type=["png","jpg","jpeg"])
if uploaded:
    img_tensor, original_img = prepare_image(uploaded.getvalue())
    st.image(original_img, caption="Input Image", use_container_width=True)

    if st.button("Segment"):
        with st.spinner("Running segmentation…"):
            pred = model.predict(img_tensor)
            mask_img = create_mask(pred)

        st.image(mask_img, caption="Predicted Mask", use_container_width=True)
        st.download_button(
            label="Download Mask",
            data=mask_img.tobytes(),
            file_name="lv_mask.png",
            mime="image/png"
        )
