import streamlit as st
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import torch
import matplotlib.pyplot as plt
import os
import sys
from utils import load_model, preprocess_image, predict
from gradcam import generate_gradcam, overlay_gradcam

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

if hasattr(st, '_config'):
    st._config.set_option('server.fileWatcherType', 'none')

st.set_page_config(page_title="Plant Disease Detection", layout="wide")

st.markdown("""
    <style>
    body { background-color: #4CAF50; color: #4CAF50; }
    .main { background-color: #4CAF50; }
    .header { background-color: #4CAF50; padding: 1.5rem; text-align: center;
              font-size: 2rem; font-weight: bold; color: #ffffff; letter-spacing: 1px; }
    .left, .right { padding: 2rem; color: #4CAF50; }
    .left img { width: 150px; margin-bottom: 1rem; }
    .feature { margin: 1rem 0; font-size: 1.1rem; }
    .error-message { color: #FF4444; font-weight: bold; padding: 1rem; 
                     background-color: #FFE6E6; border-radius: 5px; margin: 1rem 0; }
    .success-message { color: #00AA00; font-weight: bold; padding: 1rem; 
                       background-color: #E6FFE6; border-radius: 5px; margin: 1rem 0; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">PLANT DISEASE PREDICTION</div>', unsafe_allow_html=True)

def get_proper_class_names(plant_type, num_classes):
    class_mappings = {
        "Apple": {
            4: ['Apple_scab', 'Apple_black_rot', 'Apple_cedar_apple_rust', 'Apple_healthy']
        },
        "Pepper": {
            2: ['Pepper_bacterial_spot', 'Pepper_healthy'],
            4: ['Pepper_bacterial_spot', 'Pepper_healthy', 'Pepper_class_2', 'Pepper_class_3']
        },
        "Corn": {
            4: ['Corn_cercospora_leaf_spot', 'Corn_common_rust', 'Corn_northern_leaf_blight', 'Corn_healthy']
        },
        "Potato": {
            3: ['Potato_early_blight', 'Potato_late_blight', 'Potato_healthy'],
            4: ['Potato_early_blight', 'Potato_late_blight', 'Potato_healthy', 'Potato_class_3']
        }
    }
    
    if plant_type in class_mappings and num_classes in class_mappings[plant_type]:
        return class_mappings[plant_type][num_classes]
    else:
        return [f"{plant_type}_class_{i}" for i in range(num_classes)]

plant_models = {
    "Apple": {
        "model_path": "models/inception_v3_model_apple.pth",
        "class_names": ['Black_rot', 'Healthy', 'Rust', 'Scab']
    },
    "Pepper": {
        "model_path": "models/inception_v3_model_pepper.pth",
        "class_names": ['Bacterial_spot', 'Healthy']
    },
    "Corn": {
        "model_path": "models/inception_v3_model_corn.pth",
        "class_names": ['Cercospora_leaf_spot', 'Common_rust', 'Healthy', 'Northern_leaf_blight']
    },
    "Potato": {
        "model_path": "models/inception_v3_model_potato.pth",
        "class_names": ['Early_blight', 'Healthy', 'Late_blight']
    }
}

def check_file_exists(file_path):
    return os.path.exists(file_path)

def safe_load_image_as_base64(file_path):
    try:
        if check_file_exists(file_path):
            with open(file_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        else:
            return None
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")
        return None

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="left">', unsafe_allow_html=True)
    gif_path = "groot.gif"
    gif_b64 = safe_load_image_as_base64(gif_path)
    
    if gif_b64:
        gif_html = f"""
            <div style="display: flex; justify-content: center; align-items: center;">
                <img src="data:image/gif;base64,{gif_b64}" width="240" 
                     style="border-radius: 15px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);" 
                     alt="Animated Groot GIF">
            </div>
        """
        st.markdown(gif_html, unsafe_allow_html=True)
    else:
        st.info("GIF file not found. Please ensure 'groot.gif' is in the project directory.")

    st.markdown("""
        <h4 style="color:#4CAF50;">Is your plant affected by an unknown disease?</h4>
        <h5 style="color:#4CAF50;">Don't worry!</h5>
        <div class="feature">1️⃣ Upload the infected part of your plant</div>
        <div class="feature">2️⃣ Select the plant type</div>
        <div class="feature">3️⃣ Take necessary actions based on diagnosis</div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="right">', unsafe_allow_html=True)
    st.markdown('<h2 style="color:#4CAF50;">Upload Plant Image</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose File", type=["jpg", "jpeg", "png"])
    plant_type = st.selectbox("Select Plant Type", list(plant_models.keys()), index=0)
    st.markdown('</div>', unsafe_allow_html=True)

model_info = plant_models[plant_type]
model_path = model_info["model_path"]
class_names = model_info["class_names"]

if not check_file_exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.info("Please ensure the model files are present in the 'models/' directory.")
    st.stop()

if 'models' not in st.session_state:
    st.session_state.models = {}

if plant_type not in st.session_state.models:
    try:
        with st.spinner(f"Loading {plant_type} model..."):
            model = load_model(model_path, num_classes=None)
            st.session_state.models[plant_type] = model
            actual_num_classes = model.fc.out_features
            proper_class_names = get_proper_class_names(plant_type, actual_num_classes)
            plant_models[plant_type]["class_names"] = proper_class_names
        st.success(f"{plant_type} model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model = st.session_state.models[plant_type]
class_names = plant_models[plant_type]["class_names"]

def image_to_base64(image: Image.Image):
    try:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        st.error(f"Error converting image to base64: {str(e)}")
        return None

def extract_feature_maps(model, image_tensor, layers):
    fmap_dict = {}
    hooks = []

    def get_hook(name):
        def hook_fn(module, input, output):
            try:
                if isinstance(output, torch.Tensor):
                    fmap_dict[name] = output.detach().cpu().squeeze(0)
                else:
                    fmap_dict[name] = output[0].detach().cpu().squeeze(0)
            except Exception as e:
                st.warning(f"Error extracting feature map for {name}: {str(e)}")
        return hook_fn

    try:
        for name, layer in layers.items():
            if layer is not None:
                hooks.append(layer.register_forward_hook(get_hook(name)))

        with torch.no_grad():
            model(image_tensor)
            
    except Exception as e:
        st.error(f"Error during feature extraction: {str(e)}")
    finally:
        for h in hooks:
            try:
                h.remove()
            except:
                pass

    return fmap_dict

def display_image_with_download_button(image_b64, filename, icon_b64):
    if image_b64 and icon_b64:
        st.markdown(f"""
            <div style="position: relative; display: inline-block;">
                <img src="data:image/png;base64,{image_b64}" style="width: 300px; border-radius: 10px;" />
                <a href="data:image/png;base64,{image_b64}" download="{filename}"
                   style="position: absolute; top: 10px; right: 10px; background: rgba(255,255,255,0.7); padding: 5px; border-radius: 5px;">
                    <img src="data:image/png;base64,{icon_b64}" style="width: 16px;" />
                </a>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.image(f"data:image/png;base64,{image_b64}", width=300)

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_resized = image.resize((300, 300))
        
        with st.spinner("Processing image..."):
            image_tensor = preprocess_image(image)
            prediction, probs = predict(image_tensor, model, class_names)

        pred_color = "#32CD32" if prediction == "Healthy" else "#FF5733"
        st.markdown(f"<h3 style='text-align: center; color: {pred_color}; margin-bottom: 1rem;'>Prediction: {prediction}</h3>", unsafe_allow_html=True)

        st.markdown("### Confidence Scores:")
        for i, (class_name, prob) in enumerate(zip(class_names, probs)):
            percentage = prob * 100
            st.write(f"**{class_name}**: {percentage:.2f}%")

        try:
            with st.spinner("Generating Grad-CAM visualization..."):
                target_layer = model.Mixed_7c
                class_idx = np.argmax(probs)
                heatmap = generate_gradcam(model, image_tensor.detach(), target_layer, class_idx)
                gradcam_overlay = overlay_gradcam(image, heatmap)
                gradcam_resized = Image.fromarray(gradcam_overlay).resize((300, 300))
        except Exception as e:
            st.error(f"Error generating Grad-CAM: {str(e)}")
            gradcam_resized = image_resized

        image_b64 = image_to_base64(image_resized)
        gradcam_b64 = image_to_base64(gradcam_resized)

        icon_b64 = safe_load_image_as_base64("download_icon.png")
        if not icon_b64:
            st.info("Download icon not found. Images will be displayed without download buttons.")

        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.markdown("**Uploaded Leaf**")
            display_image_with_download_button(image_b64, "uploaded_leaf.png", icon_b64)

        with col_img2:
            st.markdown("**Grad-CAM Overlay**")
            display_image_with_download_button(gradcam_b64, "gradcam_overlay.png", icon_b64)

        st.markdown("### Feature Maps")
        
        layers_to_extract = {}
        layer_names = ["Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3", 
                      "Conv2d_3b_1x1", "Conv2d_4a_3x3", "Mixed_5d", "Mixed_6e", "Mixed_7c"]
        
        for layer_name in layer_names:
            try:
                layer = getattr(model, layer_name, None)
                if layer is not None:
                    layers_to_extract[layer_name] = layer
            except AttributeError:
                st.warning(f"Layer {layer_name} not found in model")

        if layers_to_extract:
            try:
                with st.spinner("Extracting feature maps..."):
                    feature_maps = extract_feature_maps(model, image_tensor, layers_to_extract)

                for layer_name, fmap in feature_maps.items():
                    if fmap is not None and len(fmap.shape) >= 2:
                        st.markdown(f"**{layer_name}**")
                        try:
                            num_maps = min(8, fmap.shape[0])
                            fig, axes = plt.subplots(1, num_maps, figsize=(15, 5))
                            if num_maps == 1:
                                axes = [axes]
                            
                            for i in range(num_maps):
                                axes[i].imshow(fmap[i], cmap='viridis')
                                axes[i].axis("off")
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                        except Exception as e:
                            st.warning(f"Error displaying feature map for {layer_name}: {str(e)}")
            except Exception as e:
                st.error(f"Error extracting feature maps: {str(e)}")
        else:
            st.warning("No valid layers found for feature extraction")

    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")
        st.info("Please ensure you've uploaded a valid image file (JPG, JPEG, or PNG)")