AI/ML Trainee Assignment: Real-Time Object Classification

Project Objective
The goal of this assignment is to build a Convolutional Neural Network (CNN) model to classify basic objects/images (MNIST digits) and perform real-time predictions using a webcam input.

Features

CNN Model: Built using TensorFlow/Keras with a specific architecture including Conv2D, MaxPooling2D, Flatten, and Dense layers.
Real-Time Inference: Utilizes OpenCV to access the laptop webcam, process live frames, and display predictions on the screen instantly .
Data Preprocessing: Live frames are converted to grayscale, resized, and normalized to ensure accurate classification.

Repository Structure
model_training.py: Contains the training script, model architecture, and code to plot accuracy/loss.
main.py: The script for real-time inference using the laptop webcam.
mnist_model.h5: The saved trained model file.

Setup & Installation
Clone this repository:
git clone https://github.com/uscsaman2718-xyz/Acelucid-AI-ML-Trainee-Assignments.git
Install Required Libraries:
pip install tensorflow opencv-python matplotlib numpy
How to Run
Train the Model: Run the training script to generate the model file and view performance plots.
python model_training.py
Launch Real-Time Detection: Start the webcam inference script.
python main.py

Results
1. Training Accuracy and Loss
Below is the plot showing the model's training and validation accuracy over the epochs.
https://github.com/uscsaman2718-xyz/Acelucid-AI-ML-Trainee-Assignments/blob/main/Training%20Accuracy%20and%20Loss%20Screenshot.png
3. Real-Time Webcam Prediction
Screenshot of the system correctly classifying a digit in real-time.
https://github.com/uscsaman2718-xyz/Acelucid-AI-ML-Trainee-Assignments/blob/main/Real-Time%20Webcam%20Prediction%20Screenshot.png
