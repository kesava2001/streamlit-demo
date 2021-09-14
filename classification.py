import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from model import *



dictionary = ['cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']

model = define_model()

@st.cache(allow_output_mutation=True)

def predict(image):
    model.load_weights('weights.hdf5')
    img = ImageOps.fit(image, (32, 32), Image.ANTIALIAS)
    image = np.array(img)
    img = image.reshape(1, 32, 32, 3)
    res = model.predict(img)
    label = dictionary[np.argmax(res)]
    return label

st.title("Upload a Classification Example")
st.set_option('deprecation.showfileUploaderEncoding', False)


uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    img = uploaded_file.read()
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image.", use_column_width=True)
    st.write("")
    st.write("Processing...")

    label = predict(image)
    st.success(f"{label}")
