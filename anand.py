import streamlit as st
import numpy as np
import keras.utils as image
import tensorflow as tf

model = tf.keras.models.load_model("C:/Users/NCJ-CL/Downloads/trained.h5")

st.title("Pneumonia Detection")


girish = st.file_uploader("Enter Your Image", type = 'jpg')

if girish is not None:
    img = image.load_img(girish,target_size = (300,300))
    imgs = image.img_to_array(img)
    imgs = np.expand_dims(imgs,axis = 0)
    
    
    prediction = model.predict(imgs)
    prediction =  np.argmax(prediction)
    st.image(girish, caption = "Mukesh Scan",use_column_width=False)
    st.title("The result is:")
    
    if prediction >= 0.5:
        st.title("sorry to say u r positive")
    else:
        st.title("Congo not positive")