import cv2
import os
import numpy as np
import shutil
import argparse


def convert_mp4_files(source_dir, target_dir):
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Get the list of .mp4 files in the source directory
    mp4_files = [file for file in os.listdir(source_dir) if file.endswith(".mp4")]

    for file in mp4_files:
        # Get the file path
        file_path = os.path.join(source_dir, file)

        # Open the video file
        video = cv2.VideoCapture(file_path)

        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create a directory for the converted frames
        frame_dir = os.path.join(target_dir, os.path.splitext(file)[0])
        os.makedirs(frame_dir, exist_ok=True)

        # Calculate the frame indices to extract 29 frames evenly
        frame_indices = [int(idx) for idx in np.linspace(0, total_frames - 1, 29)]

        # Extract and resize frames
        frames = []
        for frame_index in frame_indices:
            # Set the frame index
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

            # Read the frame
            ret, frame = video.read()

            # Resize the frame to 256x256 pixels
            frame = cv2.resize(frame, (256, 256))

            # Append the resized frame to the list
            frames.append(frame)

        # Release the video file
        video.release()

        # Create the output video file path
        output_file = os.path.join(target_dir, f"{os.path.splitext(file)[0]}.mp4")

        # Write the resized frames to a new video file
        out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, (256, 256))
        for frame in frames:
            out.write(frame)
        out.release()

        # Delete the directory containing the frames
        shutil.rmtree(frame_dir)


def convert_mp4_file(file_path, file_path_out):
    # Open the video file
    video = cv2.VideoCapture(file_path)

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the frame indices to extract 29 frames evenly
    frame_indices = [int(idx) for idx in np.linspace(0, total_frames - 1, 29)]

    # Extract and resize frames
    frames = []
    for frame_index in frame_indices:
        # Set the frame index
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # Read the frame
        ret, frame = video.read()

        # Resize the frame to 256x256 pixels
        frame = cv2.resize(frame, (256, 256))

        # Append the resized frame to the list
        frames.append(frame)

    # Release the video file
    video.release()

    # Write the resized frames to the input file
    out = cv2.VideoWriter(file_path_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (256, 256))
    for frame in frames:
        out.write(frame)
    out.release()

    # Return the frames
    return frames


# USAGE:
# python convert_mp4_script.py /path/to/source/directory /path/to/target/directory
# Converts .mp4 files to prepare_lrw ready: (256*256, 29 frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .mp4 files in a source directory to a target directory")
    parser.add_argument("source_dir", type=str, help="Path to the source directory containing .mp4 files")
    parser.add_argument("target_dir", type=str, help="Path to the target directory for storing the converted .mp4 files")
    args = parser.parse_args()

    convert_mp4_files(args.source_dir, args.target_dir)
