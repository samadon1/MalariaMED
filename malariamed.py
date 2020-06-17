import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from tf_explain.core.grad_cam import GradCAM
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity

@st.cache(allow_output_mutation=True)
def load_cnn_model():
    cnn_model = load_model("C:\\Users\\Samadon\\Desktop\\model\\model.h5")

    return cnn_model

st.write('# MalariaMED')
st.write('This Deep learning framework is for the purpose of computer aided diagnosis of focal leakage, punctuate leakage, vessel leakage and papilledema, a set of retinal abnormalities that is unique to severe malaria which is common in children with cerebral malaria. ')
st.write('Select an image from the left pane and leave the rest to the neural network...')

model = load_cnn_model()

uploaded_image = st.sidebar.file_uploader("Choose a JPG file", type="jpg")
if uploaded_image:
    st.sidebar.info('Uploaded image:')
    st.sidebar.image(uploaded_image, width=240)
    grad_cam_button = st.sidebar.button('Grad CAM')
    patch_size_value = st.sidebar.slider('Patch size:', 10, 90, 20, 10)
    occlusion_sensitivity_button = st.sidebar.button('Occlusion Sensitivity')
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = cv2.resize(image, (150, 150))
    image = img_to_array(image)
    image = preprocess_input(image)
    expanded_image = np.expand_dims(image, axis=0)
    
    class_names = ['Focal Leakage', 'Normal', 'Papilledema', 'Punctuate Leakage', 'Vessel Leakage']
    classes = model.predict(expanded_image)[0]
    (focal, normal, papilledema, punctuate, vessel) = classes
    pred = list(classes)
    for i in pred:
        x =  max(pred)
        y =  pred.index(x)
    predicted_class_index = y
    classes = class_names[y]
    x = int(x*100)
    st.subheader(classes + ':  ' + str(x) + '%')
    
    
    
    if grad_cam_button:
        data = ([image], None)
        explainer = GradCAM()
        grad_cam_grid = explainer.explain(
            data, model, class_index=y, layer_name="mixed7"
        )
        st.image(grad_cam_grid, width = 200)

    if occlusion_sensitivity_button:
        data = ([image], None)
        explainer = OcclusionSensitivity()
        sensitivity_occlusion_grid = explainer.explain(data, model, y , patch_size_value)
        st.image(sensitivity_occlusion_grid, width = 200)

