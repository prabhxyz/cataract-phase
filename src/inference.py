import os
import argparse
import torch
import cv2
import numpy as np

# If you are using TorchVision or Albumentations transforms, import them here
# from torchvision import transforms
# from albumentations import Compose, Resize, etc...

from model import build_model  # The same model definition with MobileNetV2
from dataset import CataractPhaseDataset  # We just want the same label mappings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to your 'model_latest.pth' or a similar checkpoint file")
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to the .mp4 video you want to do quick inference on")
    parser.add_argument("--frame_skip", type=int, default=30,
                        help="Analyze every Nth frame, e.g., skip=30 means 1 frame every 30 frames")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Rebuild the model (4 classes)
    model = build_model(num_classes=4)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. Use the same phase names and label map from the dataset
    #    We can create a dummy dataset to get the label map:
    dummy_dataset = CataractPhaseDataset(root_dir=".", transform=None, frame_skip=1000000)
    # The above won't actually load anything (assuming '.' has no data),
    # but it gives us an empty dataset object with the internal label map structures.
    # If you saved the label map separately, load from there.

    # However, if the dataset can't be built empty, just define the same 4 phases manually:
    # phases = ["Capsulorhexis phase", "I/A phase", "Phaco phase", "IOL insertion phase"]
    # If the dataset can't be used to retrieve the map, do something like:
    # label_map = { p: i for i, p in enumerate(phases) }
    # inv_label_map = { i: p for p, i in label_map.items() }

    # If the dataset approach won't work empty, we'll do this approach:
    phases = ["Capsulorhexis phase", "I/A phase", "Phaco phase", "IOL insertion phase"]
    inv_label_map = {
        0: "Capsulorhexis phase",
        1: "I/A phase",
        2: "Phaco phase",
        3: "IOL insertion phase"
    }

    # If you had a real transform or something similar:
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize((224,224)),
    #     transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    # ])
    # For simplicity, we'll just do a basic conversion to Tensor.
    def basic_transform(frame_bgr):
        # Convert BGR -> RGB -> NCHW tensor
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.
        return tensor

    # 3. Open the video and do quick inference
    cap = cv2.VideoCapture(args.video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    print(f"Inference on video: {args.video_path}  (Total frames = {total_frames})")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # We'll only do inference on every Nth frame
        if frame_idx % args.frame_skip == 0:
            frame_tensor = basic_transform(frame_bgr).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(frame_tensor)
                _, predicted = torch.max(outputs, 1)
                pred_label = predicted.item()
                phase_name = inv_label_map[pred_label]

            print(f"Frame {frame_idx:5d}: predicted phase -> {phase_name}")

        frame_idx += 1

    cap.release()
    print("Done with inference!")

if __name__ == "__main__":
    main()