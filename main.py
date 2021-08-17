import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


def convert(img_path, model="spirit_away"):
    imported = tf.saved_model.load(os.path.join("saved_models", model))
    f = imported.signatures["serving_default"]
    img = np.array(Image.open(img_path).convert("RGB"))
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
image = Image.open('2808.jpg')

# 2 cols: input - output
st.image(image, caption='Sunrise by the mountains')

st.image(convert('2808.jpg', style), caption="Output")
