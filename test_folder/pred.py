from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import requests
# Load pretrained pose model
model = YOLO("yolo26n-pose.pt")
# model.export(format="onnx")  # creates 'yolo26n.onnx'

# # Load the exported ONNX model
# model = YOLO("yolo26n-pose.onnx")
# model.export(format="openvino")  # creates 'yolo26n_openvino_model/'

# # Load the exported OpenVINO model
# model = YOLO("yolo26n-pose_openvino_model/")

# Image path (can be URL or local path)
url = "https://content.api.news/v3/images/bin/f2f1ab8914f1527c527cd0513bb16b1f"
image = Image.open(requests.get(url, stream=True).raw)
# Run inference
results = model(image)

# Get annotated image
annotated_frame = results[0].plot()

# Convert BGR (OpenCV) to RGB (matplotlib)
annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

# Show image
plt.figure(figsize=(10, 8))
plt.imshow(annotated_frame)
plt.axis("off")
plt.show()