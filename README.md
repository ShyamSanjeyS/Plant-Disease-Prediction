# Plant Disease Detection System

![MIT License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
![Platform](https://img.shields.io/badge/Platform-Web%20App-lightgrey)

A plant disease detection application built with Streamlit and PyTorch. This project uses Inception V3 deep learning models to classify plant diseases and provides visual explanations through Grad-CAMs.

## Features

- Detection for Apple, Pepper, Corn, and Potato plants
- Real-time prediction with image upload
- Grad-CAM visualization to show which parts of the image influenced the model's decision
- Feature map analysis to understand how the neural network processes images
- Web-based interface that's easy to use
- Confidence scores for each prediction

## Dataset Source

This project uses the [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset), a comprehensive collection of labeled plant disease images from Kaggle.

## Project Structure

The project is organized as follows:

```
plant-disease-detection/
├── app.py                            # Main Streamlit application
├── utils.py                          # Model loading and preprocessing utilities
├── gradcam.py                        # Grad-CAM implementation
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore file
├── README.md                         # This file
├── models/                           # Directory for trained model files
│   ├── inception_v3_model_apple.pth
│   ├── inception_v3_model_pepper.pth
│   ├── inception_v3_model_corn.pth
│   └── inception_v3_model_potato.pth
├── training/
│   └── Final Inceptionv3 Model.ipynb # Model training
├── groot.gif                         # Animated GIF for the interface
└── download_icon.png                 # Download button icon
```

### Installation Steps

#### Step 1: Get the Code

First, download or clone this repository to your computer.

```bash
git clone https://github.com/ShyamSanjeyS/Plant-Disease-Prediction.git
cd Plant-Disease-Prediction
```

#### Step 2: Set Up a Virtual Environment

This keeps your project dependencies separate from other Python projects on your system.

**If you're on Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**If you're on Mac or Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install the Required Packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Application

Once everything is set up, you can start the application:

```bash
streamlit run app.py
```

## How to Use the Application

Using the application is straightforward:

1. **Upload an Image**: Click the "Choose File" button and select a photo of a plant leaf. The app accepts JPG, JPEG, and PNG files.

2. **Select Plant Type**: Use the dropdown menu to choose whether your image shows an Apple, Pepper, Corn, or Potato plant.

3. **View the Results**: The app will show you:
   - What disease it thinks the plant has (if any)
   - How confident it is in each possible diagnosis
   - A heat map showing which parts of the image were most important for the decision
   - Feature maps showing how different layers of the neural network processed your image

4. **Download Results**: You can download the processed images to save them on your computer.

## What Diseases Can It Detect?

### Apple Plants
- Apple Scab
- Black Rot
- Cedar Apple Rust
- Healthy (no disease)

### Pepper Plants
- Bacterial Spot
- Healthy (no disease)

### Corn Plants
- Cercospora Leaf Spot
- Common Rust
- Northern Leaf Blight
- Healthy (no disease)

### Potato Plants
- Early Blight
- Late Blight
- Healthy (no disease)

## Technical Information

The system uses Inception V3, a convolutional neural network that's particularly good at image classification. Images are resized to 299x299 pixels and normalized using ImageNet standards before being fed to the model.

The Grad-CAM (Gradient-weighted Class Activation Mapping) feature creates heat maps that show which parts of your image the model focused on when making its decision. This helps you understand whether the model is looking at the right things.
