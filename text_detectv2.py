import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os
import requests

# Configure Streamlit
st.set_page_config(layout="wide", page_title="Text Detection")
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Style for the URL input */
[data-testid="stTextInput"] input::placeholder {
    color: #888;
    font-style: italic;
}

.url-example-text {
    font-size: 0.8em;
    color: #888;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# JavaScript for placeholder behavior
st.components.v1.html("""
<script>
function setupUrlInput() {
    const input = document.querySelector("input[type='text']");
    if (input) {
        // Set placeholder text
        input.placeholder = "https://example.com/test.jpg";
        
        // Show example text below (optional)
        const container = input.closest('.stTextInput');
        if (container) {
            const exampleText = document.createElement('div');
            exampleText.className = 'url-example-text';
            exampleText.textContent = "Example: https://example.com/image.jpg";
            exampleText.style.fontSize = '0.8em';
            exampleText.style.color = '#888';
            exampleText.style.marginTop = '4px';
            container.appendChild(exampleText);
        }
    }
}

// Run after page loads
document.addEventListener('DOMContentLoaded', setupUrlInput);
// Also run after Streamlit reruns
if (window.stConnected) {
    setupUrlInput();
} else {
    window.addEventListener('stConnected', setupUrlInput);
}
</script>
""", height=0)

# Load custom trained model
@st.cache_resource
def load_custom_model():
    try:
        model_path = "best_roboflow.pt"
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {os.path.abspath(model_path)}")
            return None
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

model = load_custom_model()

# App title
st.title("Text Detection using roboflow dataset")

# Show both input options
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])


# URL input with empty initial value (placeholder will show instead)
image_url = st.text_input(
    "Or enter image URL", 
    value="",  # Start empty, placeholder will show
    key="url_input",
    label_visibility="visible",
    placeholder="https://example.com/test.jpg"  # This will be styled by our CSS
)

# Determine which source to use (priority to uploaded file)
source = None
source_type = None
if uploaded_file is not None:
    source = ("upload", uploaded_file)
    source_type = "Uploaded Image"
elif image_url and image_url.strip():
    source = ("url", image_url.strip())
    source_type = "URL Image"

if st.button("Detect Text") and source and model:
    with st.spinner('Processing...'):
        try:
            # Process either file upload or URL
            if source[0] == "upload":
                file_bytes = np.asarray(bytearray(source[1].read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                try:
                    response = requests.get(source[1], timeout=10)
                    response.raise_for_status()
                    img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        st.error("Failed to decode image from URL")
                        st.stop()
                    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    st.error(f"Failed to load image from URL: {str(e)}")
                    st.stop()
            
            # Create temp file for prediction
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                temp_path = tmp.name
                cv2.imwrite(temp_path, img)
            
            # Run detection
            results = model.predict(
                source=temp_path,
                conf=0.5,
                imgsz=640,
                save=False
            )
            
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
            
            # Process and display results
            if len(results) > 0:
                res_plotted = results[0].plot()
                detected_img = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"Original Image ({source_type})")
                    st.image(original_img, use_column_width=True)
                
                with col2:
                    st.subheader("Detected Text")
                    st.image(detected_img, use_column_width=True)
                
                # Show detection details
                with st.expander("Detection Details"):
                    st.write(f"Source: {source_type}")
                    if source[0] == "url":
                        st.write(f"URL: {source[1]}")
                    st.write(f"Detections: {len(results[0].boxes)}")
                    st.write("Confidence scores:", [round(float(x), 2) for x in results[0].boxes.conf])
            else:
                st.warning("No text detected in the image")
                
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            if 'temp_path' in locals():
                try:
                    os.unlink(temp_path)
                except:
                    pass
elif not model:
    st.warning("Model not loaded - check your best_roboflow.pt file")
elif not source:
    st.info("Please provide an image (upload or enter URL)")

# Model information
with st.expander("Model Information"):
    if model:
        st.write(f"Using model: {os.path.abspath('best_roboflow.pt')}")
        st.write("Model loaded successfully")
    else:
        st.write("Model not loaded")
    st.write("Note: File upload takes priority over URL if both are provided")