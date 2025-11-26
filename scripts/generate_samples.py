import os 
import json 
import argparse
from datasets import load_dataset
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

DATA_ROOT = "/projects/vig/Datasets/VSI-Bench/videos"
DATASET_PATH = "nyu-visionx/VSI-Bench"

def sample_frames(video_path: str, n_frames: int = 32):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        print(f"⚠️ Empty or broken video: {video_path}")
        return []
    idxs = np.linspace(0, total - 1, min(n_frames, total)).astype(int)
    frames = []
    frame_filenames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
        frame_filenames.append(f"frame_{idx:05d}.jpg")
    cap.release()
    return frames, frame_filenames

def generate_samples(output_dir: str, n_frames: int = 32):
    os.makedirs(output_dir, exist_ok=True)
    vsi_dataset = load_dataset(DATASET_PATH)["test"]
    total_samples = len(vsi_dataset)
    vsi_dataset = vsi_dataset.unique(key="scene_name")
    total_samples = len(vsi_dataset)
    print(f"Total samples: {total_samples}")

    seen_path = set()
    pbar = tqdm(total=total_samples, desc="Generating samples")
    for sample in vsi_dataset:
        dataset = sample["dataset"]
        scene_name = sample["scene_name"]
        scene_path = os.path.join(output_dir, dataset, scene_name)
        os.makedirs(scene_path, exist_ok=True)
        video_path = os.path.join(DATA_ROOT, sample["dataset"], f"{sample['scene_name']}.mp4")
        if video_path in seen_path:
            pbar.update(1)
            continue
        seen_path.add(video_path)
        frames, frame_filenames = sample_frames(video_path, n_frames)
        if not frames:
            continue
        for frame, frame_filename in zip(frames, frame_filenames):
            frame.save(os.path.join(scene_path, frame_filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/projects/vig/Datasets/VSI-Bench/sampled_frames")
    parser.add_argument("--n_frames", type=int, default=32)
    args = parser.parse_args()
    generate_samples(args.output_dir, args.n_frames)