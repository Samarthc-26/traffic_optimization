from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Load YOLO model (use a better model for accuracy)
model = YOLO("yolov8m.pt")  # Use 'yolov8m.pt' instead of 'yolov8n.pt' for better accuracy


def process_image(image_path):
    """Detects vehicles and calculates green light timing without saving output image."""

    # Load image
    image = cv2.imread(image_path)

    # Perform detection
    results = model(image)

    # Get detected objects
    vehicle_classes = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck
    vehicle_count = sum(1 for box in results[0].boxes if int(box.cls) in vehicle_classes)

    # Define green light timing formula
    green_time = max(10, min(vehicle_count * 2, 60))

    # Convert image to PIL format for drawing
    image_pil = Image.open(image_path)
    draw = ImageDraw.Draw(image_pil)

    # Load font (larger size for visibility)
    font = ImageFont.truetype("arial.ttf", 40)

    # Define text
    text = f"Vehicles: {vehicle_count} | Green Light: {green_time}s"

    # Get image dimensions
    img_width, img_height = image_pil.size

    # Calculate text position (bottom-center)
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
    text_x = (img_width - text_width) // 2
    text_y = img_height - text_height - 10

    # Add a background rectangle for visibility
    draw.rectangle([(text_x - 10, text_y - 10), (text_x + text_width + 10, text_y + text_height + 10)], fill="black")

    # Draw text in white
    draw.text((text_x, text_y), text, fill="white", font=font)

    return vehicle_count, green_time, image_pil  # Return the modified PIL image instead of saving it

