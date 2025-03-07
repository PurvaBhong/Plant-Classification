import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

# Loading the Model
model = load_model('C:/Users/purva/OneDrive/Desktop/plant_project/api/plant_disease.h5')

# Name of Classes
CLASS_NAMES = [

"Corn_(maize) Common_rust_",
"Potato Early blight",
"Tomato_Bacterial_spot",


]

# Setting Title of App
st.title("Plant Disease Detection")
st.markdown("Upload an image of the plant leaf")

# Uploading the plant image
plant_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict')

# On predict button click
if submit:
    if plant_image is not None:
        # Convert the file to an opencv image
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write("Image shape:", opencv_image.shape)

        # Resizing the image
        opencv_image = cv2.resize(opencv_image, (256, 256))

        # Normalize pixel values to range 0-1
        opencv_image = opencv_image / 255.0

        # Convert image to 4 Dimensions (batch size, height, width, channels)
        opencv_image = np.expand_dims(opencv_image, axis=0)

        # Make Prediction
        Y_pred = model.predict(opencv_image)

        # Display prediction probabilities for debugging
        st.write("Prediction probabilities:", Y_pred)
        
        # Check if the prediction is correct
        predicted_class_index = np.argmax(Y_pred)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = Y_pred[0][predicted_class_index]
        
        # Display result with confidence
        st.title(f"This is a {predicted_class_name} with confidence {confidence:.2f}")
    else:
        st.error("Please upload an image.")
