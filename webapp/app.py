import streamlit as st
import numpy as np
import cv2
from io import BytesIO

def generate_photo(input_image):
    # Placeholder: Apply a basic blur as an example
    blurred_image = cv2.GaussianBlur(input_image, (15, 15), 0)
    return blurred_image

def main():
    st.title("ProfessionalMe")

    uploaded_file = st.file_uploader("Upload a selfie", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        print("Not a valid image!")
    else:
        # Read the uploaded image
        file_bytes = BytesIO(uploaded_file.read())
        input_image = cv2.imdecode(np.frombuffer(file_bytes.read(), np.uint8), -1)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        # Generate professional photo
        generated_image = generate_photo(input_image)

        # Display photo
        st.image(generated_image, caption="Generated Professional Photo", use_column_width=True)



if __name__ == "__main__":
    main()