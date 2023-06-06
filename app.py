import torch
from utils.helpers import load_missing, extract_opencv
from flask import Flask, request, jsonify
from model.model import VideoModel
import numpy as np
import tempfile
import cv2

app = Flask(__name__)

labels = []
with open('label_sorted_full.txt') as myfile:
    labels = myfile.read().splitlines()

video_model = VideoModel(500)
weights_file = "weights/lrw-cosine-lr-acc-0.85080.pt"
weight = torch.load(weights_file, map_location=torch.device('cpu'))
load_missing(video_model, weight.get('video_model'))
video_model.eval()


def preprocess_frames(frames):
    input_shape = (1, 1, 1, 88, 88)  # Adjust the dimensions according to the model's requirements
    tensor_frames = []

    for frame in frames:
        if isinstance(frame, bytes):
            # Handle frames received as bytes (assuming encoded frames)
            nparr = np.frombuffer(frame, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            # Handle frames received as numpy arrays
            img = frame

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize the frame to the desired input shape
        resized_frame = cv2.resize(gray_frame, (input_shape[-1], input_shape[-1]))

        # Normalize the frame
        normalized_frame = (resized_frame / 255.0).astype(np.float32)

        # Add frame to the list of processed frames
        tensor_frame = np.expand_dims(normalized_frame, axis=(0, 1))  # Add extra dimensions
        tensor_frames.append(tensor_frame)

    # Stack frames to create a tensor with shape [num_frames, channels, time, height, width]
    tensor_frames = np.concatenate(tensor_frames, axis=0)

    # Add an extra dimension for batch size
    tensor_frames = np.expand_dims(tensor_frames, axis=0)

    # Convert frames to tensor
    tensor_frames = torch.tensor(tensor_frames)

    return tensor_frames



@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp_file:
        file.save(tmp_file.name)

        # Extract frames using OpenCV from the temporary file
        raw_frames = extract_opencv(tmp_file.name)

    # Preprocess the frames
    frames = preprocess_frames(raw_frames)

    # Pass the frames through the model
    with torch.no_grad():
        predictions = video_model(frames)
        predicted_label = torch.argmax(predictions)
        predicted_label = predicted_label.item()

    predicted_class = labels[predicted_label]
    print(predicted_label)
    return jsonify({'predicted_class': predicted_class})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)