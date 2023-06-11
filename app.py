import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from scripts.mp4_converter import convert_mp4_files, convert_mp4_file
from utils.face_detector import get_faces, anno_img
from utils.helpers import load_missing, extract_opencv
from flask import Flask, request, jsonify
from flask_cors import CORS
from model.model import VideoModel
import numpy as np
import tempfile
import cv2
import json

# To be able to run face_alignment
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
CORS(app, origins='*', allow_headers=['*'])

# Currently doesn't support plotting the frames after preprocessing due to dimension incompatibility.
def plot_frames(frames_to_plot):
    normalized_frames_to_plot = frames_to_plot.clone()
    if len(frames_to_plot.shape) == 5:
        normalized_frames_to_plot = normalized_frames_to_plot[0]

    num_frames = normalized_frames_to_plot.shape[0]

    for i in range(num_frames):
        frame = normalized_frames_to_plot[i]  # Extract the frame
        if frame.ndim == 3:  # If the frame is 3D, reshape it to 2D
            frame = frame.squeeze()
        if frame.ndim == 4:
            frame = frame.squeeze().squeeze()
        plt.imshow(frame, cmap='gray')
        plt.axis('off')
        plt.show()


def preprocess_frames(frames_bytes):
    input_shape = (1, 1, 88, 88)  # Adjust the dimensions according to the model's requirements
    tensor_frames = []

    for frame in frames_bytes:
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


def get_top_10_tuples(predictions_to_probabilities):
    sorted_tuples = sorted(predictions_to_probabilities, key=lambda x: x[1], reverse=True)
    top = sorted_tuples[:10]
    return top


def map_labels(tuples_list, labels):
    tuples = []
    for tup in tuples_list:
        index = tup[0]
        value = tup[1]
        label = labels[index]
        tuples.append((label, value))

    return tuples


def load_model(model_type):
    if model_type == "lrw":
        with open('label_sorted_full.txt') as myfile:
            labels = myfile.read().splitlines()

        video_model = VideoModel(500)
        weights_file = "weights/lrw-cosine-lr-acc-0.85080.pt"
        weight = torch.load(weights_file, map_location=torch.device('cpu'))

    elif model_type == "unseen":
        with open('label_sorted_5.txt') as myfile:
            labels = myfile.read().splitlines()

        video_model = VideoModel(5)
        weights_file = "weights/best-custom-weight-0.97.pt"
        weight = torch.load(weights_file, map_location=torch.device('cpu'))

    else:
        raise ValueError("WTF")

    load_missing(video_model, weight.get('video_model'))
    video_model.eval()

    return video_model, labels


@app.route('/predict/<model_type>', methods=['POST'])
def predict(model_type):
    video_model, labels = load_model(model_type)

    file = request.files['file']

    with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp_file:
        with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp_file_out:
            file.save(tmp_file.name)

            convert_mp4_file(tmp_file.name, tmp_file_out.name)

            # Extract frames using OpenCV from the temporary file
            raw_frames = extract_opencv(tmp_file_out.name)

    faces_landmark = get_faces(raw_frames)
    frames = anno_img(raw_frames, faces_landmark)

    # Preprocess the frames
    frames = preprocess_frames(frames)

    # Pass the frames through the model
    with torch.no_grad():
        predictions = video_model(frames)

    probabilities = F.softmax(predictions, dim=1)

    class_percentages = [(idx, p.item() * 100) for idx, p in enumerate(probabilities[0])]
    top_10 = get_top_10_tuples(class_percentages)
    top_10_labels = map_labels(top_10, labels)

    json_data = json.dumps([[label, value] for label, value in top_10_labels])
    return json_data


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
