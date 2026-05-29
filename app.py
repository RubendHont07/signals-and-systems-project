import streamlit as st
import cv2
import numpy as np
import io
import zipfile

# =========================================================================
# TU/e Signals and Systems Project (4CA20) - 2D DFT Blur Detection
# =========================================================================

def calculate_dft_blur_score(file_bytes, image_size=256, radius_ratio=0.15):
    # Convert uploaded bytes to an OpenCV image (grayscale)
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return None

    img = cv2.resize(img, (image_size, image_size))
    img = img.astype(np.float32)
    
    # Remove average brightness (DC component)
    img = img - np.mean(img)
    
    # Compute 2D DFT
    F = np.fft.fft2(img)
    F_shifted = np.fft.fftshift(F)
    
    # Power spectrum
    power = np.abs(F_shifted) ** 2
    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    radius = radius_ratio * min(h, w)
    high_freq_mask = distance > radius
    high_freq_energy = np.sum(power[high_freq_mask])
    total_energy = np.sum(power)
    score = high_freq_energy / total_energy
    
    return score

# =========================================================================
# STREAMLIT USER INTERFACE
# =========================================================================
st.set_page_config(page_title="Signals & Systems Project", page_icon="📸")

st.title("📸 Signals and Systems Project (4CA20)")
st.subheader("2D Discrete Fourier Transform - Image Sharpness Filter")
st.write("Upload a batch of photos. This application analyzes high frequencies in the frequency domain to determine if the photos are sharp or blurry.")

st.write("---")

# =========================================================================
# THE INTERACTIVE SLIDER (WITH VISUAL LABELS)
# =========================================================================
st.write("### Filter Settings")

# We create three columns to neatly align the red and green text to the left and right
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.markdown("<p style='text-align: left; color: #ff4b4b; font-weight: bold;'>🔴 Tolerant (More Blurry)</p>", unsafe_allow_html=True)
with col3:
    st.markdown("<p style='text-align: right; color: #00cc66; font-weight: bold;'>🟢 Strict (Only Sharp)</p>", unsafe_allow_html=True)

# The slider itself visually goes from 0 to 100%
slider_percentage = st.slider(
    "Select the classification threshold:", 
    min_value=0, 
    max_value=100, 
    value=32, # 32% maps roughly to your original 0.055 threshold
    step=1,
    format="%d%%",
    label_visibility="collapsed" 
)

# Under the hood, we convert the 0-100% scale back to the 0.010-0.150 scale your algorithm needs
classification_threshold = 0.010 + (slider_percentage / 100.0) * (0.150 - 0.010)

st.write("---")

# =========================================================================
# FILE UPLOAD & PROCESSING
# =========================================================================
# File uploader
uploaded_files = st.file_uploader(
    "Drag and drop your images here...", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"📊 **Analyzing {len(uploaded_files)} images using a strictness of {slider_percentage}%...**")
    
    sharp_images = []
    
    # Create a grid of 3 columns for displaying the sharp images neatly
    cols = st.columns(3)
    col_idx = 0
    
    for file in uploaded_files:
        file_bytes = file.read()
        score = calculate_dft_blur_score(file_bytes)
        
        if score is not None:
            # FILTERING LOGIC: Check image score against the mapped threshold
            if score >= classification_threshold:
                sharp_images.append((file.name, file_bytes))
                
                # Show a preview of the sharp image in the web app
                with cols[col_idx % 3]:
                    st.image(file_bytes, caption=f"{file.name}\nScore: {score:.4f}", use_container_width=True)
                col_idx += 1

    st.success(f"✅ Analysis complete! {len(sharp_images)} out of {len(uploaded_files)} images were classified as sharp.")

    # Create a ZIP file to easily export the sharp images
    if sharp_images:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for file_name, file_bytes in sharp_images:
                zip_file.writestr(file_name, file_bytes)
        
        st.write("---")
        # Large download button for the results
        st.download_button(
            label="⬇️ Download Sharp Images (.zip)",
            data=zip_buffer.getvalue(),
            file_name="signals_systems_sharp_images.zip",
            mime="application/zip"
        )
