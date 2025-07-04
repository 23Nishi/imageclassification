import tensorflow as tf
import keras
import streamlit as st #type:ignore
import pickle
import numpy as np
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Image Classification App",
    page_icon="🖼️",
    layout="centered"
)

# Function to load model
@st.cache_resource
def load_model():
    try:
        with open('cnn.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to preprocess image
def preprocess_image(image, target_size=(40, 40)):  # Changed target size to match model expectation
    # Resize image
    image = image.resize(target_size)
    # Convert to array and normalize
    image_array = np.array(image) / 255.0
    
    # Reshape if needed (handle grayscale vs RGB)
    if len(image_array.shape) == 2:  # Grayscale
        image_array = np.expand_dims(image_array, axis=-1)
    elif len(image_array.shape) == 3 and image_array.shape[2] == 4:  # RGBA
        image_array = image_array[:, :, :3]  # Remove alpha channel
    
    # Flatten the image if model expects flattened input
    # Comment out the next line if your model expects 4D input (batch, height, width, channels)
    # image_array = image_array.reshape(1, -1)  # Flatten to (1, height*width*channels)
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Main function
def main():
    st.title("Image Classification App")
    st.write("Upload an image and the model will classify it!")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.warning("Please ensure 'cnn.pkl' file is in the same directory as this script")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image",  use_container_width=True)
        st.write("")
        
        # Add a prediction button
        if st.button("Classify Image"):
            with st.spinner("Classifying..."):
                try:
                    # Display image shape information for debugging
                    image_array = np.array(image)
                    st.info(f"Original image shape: {image_array.shape}")
                    
                    # Preprocess the image
                    processed_image = preprocess_image(image)
                    st.info(f"Processed image shape: {processed_image.shape}")
                    
                    # Make prediction
                    prediction = model.predict(processed_image)
                    
                    # Display results
                    st.subheader("Prediction Results:")
                    
                    # Assuming the model outputs class probabilities
                    # Modify this part according to your model's output format
                    predicted_class = np.argmax(prediction, axis=1)[0]
                    confidence = np.max(prediction) * 100
                    
                    st.success(f"Predicted Class: {predicted_class}")
                    st.progress(int(confidence))
                    st.write(f"Confidence: {confidence:.2f}%")
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    st.error("Model input shape mismatch. Please check your model architecture.")
                    
                    # Print model summary if possible
                    try:
                        st.write("Model expects input shape:")
                        model_input = model.layers[0].input_shape
                        st.code(f"Input shape: {model_input}")
                    except:
                        pass

if __name__ == "__main__":
    main()
