from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model (using medium version for better accuracy)
model = YOLO("yolov8m.pt")

# Define vehicle classes (COCO dataset: 2=car, 3=motorcycle, 5=bus, 7=truck)
VEHICLE_CLASSES = {2, 3, 5, 7}

def process_image(image_path):
    # Load and resize image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform object detection with tuned confidence & IoU thresholds
    results = model(image_rgb, conf=0.4, iou=0.5)

    # Extract detected objects
    detected_boxes = results[0].boxes
    detected_classes = detected_boxes.cls.cpu().numpy()

    # Count vehicles based on class
    vehicle_count = sum(1 for cls in detected_classes if int(cls) in VEHICLE_CLASSES)

    # Estimate green signal time (more refined: base + per vehicle time)
    base_time = 10  # Base green time (seconds)
    extra_time_per_vehicle = 3  # Additional seconds per vehicle
    green_time = min(base_time + vehicle_count * extra_time_per_vehicle, 60)

    # Convert image to PIL for annotation
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)

    # Load font safely
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()

    # Draw text with vehicle count and time
    text = f"Vehicles: {vehicle_count} | Time: {green_time}s"
    text_position = (20, 20)
    draw.text(text_position, text, font=font, fill=(255, 0, 0))

    # Convert back to OpenCV format
    processed_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return vehicle_count, green_time, processed_image

