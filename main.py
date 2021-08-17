import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


def convert(img, model="spirit_away"):
    imported = tf.saved_model.load(os.path.join("saved_models", model))
    f = imported.signatures["serving_default"]
    img = np.array(img.convert("RGB"))
    img = np.expand_dims(img, 0).astype(np.float32) / 127.5 - 1
    out = f(tf.constant(img))['conv2d_25']

    return ((out.numpy().squeeze() + 1) * 127.5).astype(np.uint8)


"""
# Cartoonify
"""

# Select style
style = st.selectbox('Please pick an anime you like', ['spirit_away', 'your_name'])

'You selected: ', style

# Upload file
uploaded_file = st.file_uploader("Upload Files", type=['png', 'jpeg', 'jpg'])
if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
    image = Image.open(uploaded_file)

    # 2 cols: input - output
    st.image(image, caption='Sunrise by the mountains')

    st.image(convert(image, style), caption="Output")
