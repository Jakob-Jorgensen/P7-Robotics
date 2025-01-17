from ultralytics import YOLO
import os
import cv2
import numpy as np

if __name__ == '__main__':
    # Load the YOLO model
    model = YOLO("yolov8s-worldv2.pt")  # Replace with your model

    # Train the model
    model.train(
        data="dataset.yaml",  # Path to your data.yaml file
        epochs=50,           # Number of epochs
        batch=16,            # Batch size
        imgsz=640,           # Image size
        lr0=0.0001,
        patience=20,
        project="YOLO_Training",  # Directory for training outputs
        name="custom_model",     # Experiment name
        pretrained=True          # Use pretrained weights
    )

