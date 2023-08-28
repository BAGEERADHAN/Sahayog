# Hand Gesture Recognition System README
This repository contains the code and documentation for a Hand Gesture Recognition System using EMG signals. The system allows individuals with spinal cord injuries to control an electric wheelchair using hand gestures.


## Overview
This project focuses on developing a Hand Gesture Recognition System using EMG signals for individuals with spinal cord injuries. The system utilizes two EMG sensors to capture forearm muscle activity, extracts hand-crafted features from the EMG signals, and employs an optimized Artificial Neural Network (ANN) model for gesture classification. The trained model is deployed on a Raspberry Pi 3 Model B as an edge device, allowing real-time control of an electric wheelchair based on the recognized hand gestures. The integration of DC motors, servo motors, and the L298 motor driver ensures precise and smooth wheelchair motion. Overall, this project demonstrates a practical and affordable solution to enhance mobility and independence for individuals with spinal cord injuries.


## Hardware Requirements
- Muscle BioAmp Candy (EMG sensor)
- Raspberry Pi 3 Model B (edge device)
- Arduino UNO
- DC Motors
- L298 Motor Driver

## Software Requirements
- Arduino IDE
- Geany (text editor)
- Google Colab (for training and fine-tuning the ML model)
- Raspbian (OS for Raspberry Pi)

## Installation and Setup
1. Connect the Muscle BioAmp Candy to the Arduino UNO and Raspberry Pi 3 Model B.
2. Install the Arduino IDE on your development machine and upload the necessary Arduino code to the Arduino UNO.
3. Install Geany or any preferred text editor for modifying and running the code.
4. Use Google Colab to train and fine-tune the ML model. Ensure you have the required dependencies installed.
5. Install Raspbian on the Raspberry Pi 3 Model B and set up the necessary configurations.
6. Transfer the trained ML model (using joblib) to the Raspberry Pi.

## Usage
1. Connect the DC Motors to the L298 Motor Driver and make the necessary connections with the Raspberry Pi.
2. Run the code on the Raspberry Pi to initialize the motor control.
3. Perform the hand gestures in front of the Muscle BioAmp Candy to control the electric wheelchair.
4. The ML model deployed on the Raspberry Pi will interpret the gestures and control the wheelchair accordingly.

## Troubleshooting
- Ensure all the hardware components are properly connected and powered.
- Check the dependencies and versions of software packages used in the project.
- Verify the ML model file (joblib) is present on the Raspberry Pi.

## Acknowledgements
- Muscle BioAmp Candy- EMG sensor used in the project.
- Raspberry Pi - Edge device for real-time gesture recognition.
- Arduino - Platform for motor control and interfacing.
- L298 Motor Driver - Motor driver used for controlling DC motors.

Feel free to contribute to this project by submitting bug reports, feature requests, or pull requests.
