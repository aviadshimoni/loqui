import torch
from torchvision.transforms import CenterCrop, Grayscale
import transforms
from utils.helpers import load_missing
from scripts.prepare_lrw import extract_opencv
from flask import Flask, request, jsonify
from model.model import VideoModel
app = Flask(__name__)

labels = []
with open('label_sorted.txt') as myfile:
    labels = myfile.read().splitlines()

video_model = VideoModel(5)
weights_file = "weights/loqui_daniel/checkpoints/few_shot/_iter_4837_epoch_58_v_acc_0.97710_.pt"
weight = torch.load(weights_file, map_location=torch.device('cpu'))
load_missing(video_model, weight.get('video_model'))
video_model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    raw_frames = extract_opencv(file)
    frames = preprocess_frames(raw_frames)

    # Pass the frames through the model
    with torch.no_grad():
        predictions = video_model(frames)
        predicted_label = torch.argmax(predictions)
        predicted_label = predicted_label.item()

    predicted_class = labels[predicted_label]
    return jsonify({'predicted_class': predicted_class})

def preprocess_frames(frames):
    transform = transforms.Compose([
        CenterCrop(88),
        Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        # Add any other transformations you need
    ])
    preprocessed_frames = torch.stack([transform(frame) for frame in frames])
    return preprocessed_frames

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
