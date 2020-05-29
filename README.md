# Gesture Controlled WinCC
WinCC Advanced RT controlled using hand gestures.

## Introduction:
WinCC Advanced is an industrial 'Human Machine Interface' (HMI) system from SIEMENS. You can read more about it in [this page](http://siemens.com/wincc-advanced).
This project aims on sending commands to WinCC Advanced RT using hand gestures.
  
## Project Structure:
The gesture recognition logic is written in Python. It uses [OpenCV](https://opencv.org/) for the image processing and [Tensorflow](https://www.tensorflow.org/) for the machine learning. 
    
The visualization is done with WinCC Advanced. The Python script communicates with WinCC using OPC-UA channel. WinCC has native OPC-UA channel driver. For the Python module, "[opcua](https://github.com/FreeOpcUa/python-opcua)" package is used. 
  
## Usage:
The Python part comprises of three files. 
* A module to create image dataset for training the Neural Network (1_create_dataset.py).
* A module to train the Neural Network model (2_train_model.py).
* A module that uses the trained model to detect gestures and send commands to WinCC (3_gesture_control.py).

A detailed description on usage of individual module is given in module docstring. Trained model "**_gest_recog_model.h5_**" is also uploaded.

The **__WinCC RT__** project just has one screen. It reads 6 tags over OPC-UA to complete the demo.

## Demo Video
Please see [this LinkedIn post](https://www.linkedin.com/feed/update/urn:li:activity:6672167250360967168/) for a video demo. 
