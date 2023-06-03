import torch
from utils.helpers import load_missing
from scripts.prepare_lrw import extract_opencv
from flask import Flask, request, jsonify
from model.model import VideoModel
import cv2
import numpy as np
app = Flask(__name__)

labels = []
with open('/tf/loqui/label_sorted_full.txt') as myfile:
    labels = myfile.read().splitlines()

video_model = VideoModel(5)
weights_file = "/tf/weights/lrw-cosine-lr-acc-0.85080.pt"
weight = torch.load(weights_file, map_location=torch.device('cpu'))
load_missing(video_model, weight.get('video_model'))
video_model.eval()


import tempfile

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    # Create a temporary file to write the resized and processed frames
    with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp_file:
        # Read the input video file
        video = cv2.VideoCapture(file)

        # Get the total number of frames in the video
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine the indices of frames to extract
        target_num_frames = 29
        if total_frames > target_num_frames:
            indices = np.linspace(0, total_frames - 1, target_num_frames, dtype=np.int)
        else:
            indices = np.arange(total_frames)

        # OpenCV VideoWriter to write the resized frames to the temporary file
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(tmp_file.name, fourcc, fps, (width, height))

        # Read and process each frame
        frame_counter = 0
        while frame_counter < total_frames:
            success, frame = video.read()
            if success:
                if frame_counter in indices:
                    # Resize the frame
                    resized_frame = cv2.resize(frame, (256, 256))
                    writer.write(resized_frame)
                frame_counter += 1
            else:
                break

        # Release the VideoCapture and VideoWriter objects
        video.release()
        writer.release()

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
        return jsonify({'predicted_class': predicted_class})


def preprocess_frames(frames, input_shape=(88,88)):
    # Convert frames to numpy arrays
    frames = [np.array(frame, dtype=np.uint8) for frame in frames]

    # Resize the frames
    resized_frames = [cv2.resize(frame, input_shape) for frame in frames]

    # Convert frames to grayscale
    grayscale_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in resized_frames]

    # Normalize the frames
    normalized_frames = [(frame / 255.0).astype(np.float32) for frame in grayscale_frames]

    # Stack frames to create a tensor with shape [num_frames, height, width]
    tensor_frames = np.stack(normalized_frames)

    # Add a channel dimension to the tensor
    tensor_frames = np.expand_dims(tensor_frames, axis=1)

    # Convert frames to tensor
    tensor_frames = torch.tensor(tensor_frames)

    return tensor_frames


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
