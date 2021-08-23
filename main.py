import os
from datetime import time

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


def convert(img, model="spirit_away"):
    st.spinner(text='In progress...')
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

col1, col2 = st.columns(2)

original = Image.open("1.jpg")
col1.header("Original")
col1.image(original, use_column_width=True)

col2.header(f"'{style}' style")
if style == 'spirit_away':
    col2.image(Image.open("sa.jpeg"), use_column_width=True)
else:
    col2.image(Image.open("yn.jpeg"), use_column_width=True)

if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
    image = Image.open(uploaded_file)

    # 2 cols: input - output
    st.image(image, caption=f"Your input: {uploaded_file.name}")

    with st.spinner(f":cooking: `Cooking` your image :cooking:"):
        result = convert(image, style)

    st.image(result, caption=f"Your image has turned to `{style}` style.")
