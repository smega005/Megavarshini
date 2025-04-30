import os
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import time
from sklearn.metrics import precision_score, recall_score, jaccard_score

IMG_SIZE = 256
PIXEL_SPACING_MM = 0.5

# ----------- Metrics -----------
def dice_coefficient(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-6)

def calculate_iou(y_true, y_pred):
    return jaccard_score(y_true.flatten(), y_pred.flatten(), zero_division=1)

def calculate_precision(y_true, y_pred):
    return precision_score(y_true.flatten(), y_pred.flatten(), zero_division=1)

def calculate_recall(y_true, y_pred):
    return recall_score(y_true.flatten(), y_pred.flatten(), zero_division=1)

# ----------- Segmentation -----------
def simple_threshold_segmentation(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)
    tumor_pixels = np.sum(thresh == 255)
    return thresh, tumor_pixels

# ----------- Visualization -----------
def plot_results(image, mask, prediction):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image.squeeze(), cmap='gray')
    axs[0].set_title('Original MRI')
    axs[1].imshow(mask.squeeze(), cmap='gray')
    axs[1].set_title('Ground Truth')
    axs[2].imshow(prediction.squeeze(), cmap='gray')
    axs[2].set_title('Predicted Mask')
    for ax in axs:
        ax.axis('off')
    st.pyplot(fig)

# ----------- Main App -----------
def main():
    st.title("🧠 Brain Tumor Segmentation - Streamlit App")

    menu = ["Simple Thresholding", "Deep Learning (U-Net)"]
    choice = st.sidebar.selectbox("Select Method", menu)

    uploaded_file = st.file_uploader("Upload a Brain MRI Image", type=['png', 'jpg', 'jpeg'])
    uploaded_mask = st.file_uploader("Upload Ground Truth Mask (Optional)", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 0)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        st.image(image, caption='Uploaded MRI' )

        if choice == "Simple Thresholding":
            start = time.time()
            segmented, area_pixels = simple_threshold_segmentation(image)
            end = time.time()
            inference_time = end - start

            st.image(segmented, caption='Segmented Tumor Mask')
            st.success(f"Tumor Size: {area_pixels} pixels")
            st.info(f"Estimated Tumor Area: {area_pixels * (PIXEL_SPACING_MM ** 2):.2f} mm²")
            st.info(f"Inference Time: {inference_time:.3f} seconds")

            # If ground truth provided
            if uploaded_mask is not None:
                mask_bytes = np.asarray(bytearray(uploaded_mask.read()), dtype=np.uint8)
                mask = cv2.imdecode(mask_bytes, 0)
                mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
                st.image(mask, caption="Ground Truth Mask")

                y_true = (mask > 127).astype(np.uint8)
                y_pred = (segmented > 127).astype(np.uint8)

                dice = dice_coefficient(y_true, y_pred)
                iou = calculate_iou(y_true, y_pred)
                precision = calculate_precision(y_true, y_pred)
                recall = calculate_recall(y_true, y_pred)

                st.subheader("📊 Segmentation Metrics")
                st.write(f"**Dice Coefficient**: {dice:.4f}")
                st.write(f"**IoU**: {iou:.4f}")
                st.write(f"**Precision**: {precision:.4f}")
                st.write(f"**Recall**: {recall:.4f}")

                plot_results(image, y_true, y_pred)
            else:
                st.warning("⚠️ Ground Truth mask not uploaded. Metrics cannot be computed.")

        elif choice == "Deep Learning (U-Net)":
            st.error("❌ Deep Learning (U-Net) is not available because TensorFlow is not compatible with Python 3.13. Please use Python 3.10 or 3.11.")

if __name__ == '__main__':
    main()
