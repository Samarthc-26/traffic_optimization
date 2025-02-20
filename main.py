import streamlit as st
from ml_model import process_image

st.title("Traffic Optimization System")

uploaded_file = st.file_uploader("Upload an image of a traffic intersection", type=["jpg", "png", "jpeg"])

if uploaded_file:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Call function from `ml_model.py`
    vehicle_count, green_time, processed_image = process_image("temp.jpg")

    # Display processed output directly
    st.image(processed_image, caption=f"Detected Vehicles: {vehicle_count} | Green Light: {green_time}s", use_column_width=True)
