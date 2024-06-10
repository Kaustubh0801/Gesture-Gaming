# Gesture Gaming Using Machine Learning and Raspberry Pi

This project combines machine learning techniques with hardware integration to create a gesture-based gaming system using a Raspberry Pi. The system allows users to interact with a game environment using hand gestures recognized by a trained Convolutional Neural Network (CNN). Servo motors and seven-segment displays are used to provide visual feedback and enhance the gaming experience.

## Features

  Recognize hand gestures using a CNN model trained on TensorFlow/Keras.
  
  Control servo motors connected to a Raspberry Pi GPIO for physical feedback.
  
  Display game scores or other information on two seven-segment displays.
  
  Seamless integration of hardware components with the CNN model for real-time interaction.
  
  Easily customizable and extensible for different gaming scenarios.
  
## Hardware Requirements

  Raspberry Pi (Model 3B+ or later recommended)
  
  Servo motors (3x)
  
  Seven-segment displays (2x)
  
  Jumper wires and breadboard for connections
  
## Software Requirements

  Raspbian OS installed on Raspberry Pi
  
  Python 3.x with libraries: TensorFlow, RPi.GPIO, serial
  
  Trained CNN model (saved in TensorFlow SavedModel format)
  
## Installation

   1.Clone this repository to your Raspberry Pi:
     git clone https://github.com/yourusername/gesture-gaming.git
     
   2.Install required Python libraries:
    pip install tensorflow RPi.GPIO pyserial
    
   3.Upload your trained CNN model to the models/ directory.

## Machine Learning Model
The gesture recognition system in this project is powered by a Convolutional Neural Network (CNN) model trained using TensorFlow/Keras. The model architecture is based on MobileNet, a lightweight and efficient CNN architecture designed for mobile and embedded devices. MobileNet achieves high accuracy with lower computational resources, making it ideal for real-time gesture recognition on the Raspberry Pi.

## Tools Used
  TensorFlow/Keras: Used for training and deploying the CNN model.
  
  RPi.GPIO: Python library for controlling GPIO pins on the Raspberry Pi.
  
  pyserial: Python library for serial communication with external devices (e.g., Arduino).

## Usage
  Connect the servo motors and seven-segment displays to the GPIO pins of the Raspberry Pi 
  according to the provided wiring diagram.
  
  Run the main Python script
  
  Follow the on-screen instructions to interact with the gesture-based gaming system.

## Configuration

  Adjust GPIO pin numbers in the code if using different pins for servo motors or displays.
  
  Fine-tune servo motor positions and display updates as needed for your specific gaming 
  scenario.
  
## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for any improvements or bug fixes.

## Acknowledgments

  Inspired by the work of researchers and developers in the fields of machine learning and 
  physical computing.
  
  Special thanks to the open-source community for providing valuable resources and libraries.

