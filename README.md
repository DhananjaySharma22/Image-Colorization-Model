# Image Colorization using GANs

## Overview
This project leverages a **Generative Adversarial Network (GAN)** to colorize black and white images. The model is trained using a dataset from Kaggle and is capable of generating high-quality colorized images while preserving details and textures. The model has been trained on **256x256** images and performs exceptionally well in converting grayscale images to realistic color outputs.

## Dataset
The dataset used for training can be found on Kaggle:
https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization

## Model
A **Conditional GAN (cGAN)** architecture is used, where:
- **Generator** learns to colorize grayscale images.
- **Discriminator** evaluates the realism of the generated color images.

The pre-trained model is available on Kaggle:
https://www.kaggle.com/models/dhananjaysharma07/colorizer/

## Code
The model was implemented and trained using **Kaggle Notebooks**.

## Usage
To use the trained model for colorizing grayscale images:
1. Download the pre-trained model.
2. Load the model in Python using TensorFlow/Keras.
3. Pass grayscale images of size **256x256** through the model to obtain colorized versions.

Example usage:
```python
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the trained model
model = load_model("path_to_model.h5")

# Load and preprocess grayscale image
image = cv2.imread("grayscale_image.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (256, 256))
image = np.expand_dims(image, axis=-1)  # Add channel dimension
image = np.expand_dims(image, axis=0)  # Add batch dimension
image = image / 255.0  # Normalize

# Generate colorized image
colorized_image = model.predict(image)
colorized_image = (colorized_image * 255).astype(np.uint8)

# Save the output
cv2.imwrite("colorized_image.jpg", colorized_image)
```

## Results
The model effectively colorizes black and white images with high accuracy. It has been trained on a diverse dataset, allowing it to generalize well across different image categories.

## Dependencies
- TensorFlow/Keras
- OpenCV
- NumPy
- Kaggle Notebook (for training)

## Future Improvements
- Fine-tune the model with larger datasets for better generalization.
- Experiment with other GAN architectures such as CycleGAN for improved color consistency.

