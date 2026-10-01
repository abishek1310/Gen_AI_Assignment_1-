# Real or AI? Human vs. Model

CS6180 HW1

A game where you compete against a deep learning model to tell real human faces
from AI-generated ones.

## How to play
1. Look at the face and click "Real" or "AI-Generated".
2. The model makes its own prediction.
3. The correct answer is revealed, and both scores are updated.
4. Click "Next Image" to continue.
5. After 10 rounds, the higher score wins.

## Model
- MobileNetV3Small, pretrained on ImageNet
- Fine-tuned on 1,200 real and AI-generated face images (128x128)
- Last 50 layers unfrozen, Adam (learning rate 1e-4), BatchNorm kept frozen
- Test accuracy: 77.0% on the fixed 300-image test set

## Files
- app.py: the Streamlit game
- best_model.keras: the fine-tuned model
- test_images/: 50 images from the test set (25 real, 25 AI), never used for training
- requirements.txt: required libraries

## Deployment
Built with Streamlit and deployed on Streamlit Community Cloud

