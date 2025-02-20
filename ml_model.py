from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

def process_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform object detection
    results = model(image_rgb)

    # Count vehicles
    vehicle_classes = [2, 3, 5, 7]  # Car, motorcycle, bus, truck (COCO dataset)
    vehicle_count = sum(1 for r in results[0].boxes.cls if int(r) in vehicle_classes)

    # Estimate green signal time (simple formula: 5 sec per vehicle)
    green_time = min(vehicle_count * 5, 60)

    # Convert back to PIL for drawing
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)

    # Load font (Try arial.ttf, else use default)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()

    # Draw text
    text = f"Vehicles: {vehicle_count} | Time: {green_time}s"
    text_position = (20, 20)
    draw.text(text_position, text, font=font, fill=(255, 0, 0))

    # Convert back to OpenCV format
    processed_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return vehicle_count, green_time, processed_image
