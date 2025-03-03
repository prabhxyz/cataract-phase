import os
import argparse
import torch
import cv2
import numpy as np

from model import build_model

def main():
    parser = argparse.ArgumentParser(description="Extract frames for a chosen phase with a confidence threshold.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model (e.g. checkpoints/model_latest.pth)")
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to the .mp4 video")
    parser.add_argument("--phase_name", type=str, required=True,
                        help='One of ["Capsulorhexis phase", "I/A phase", "Phaco phase", "IOL insertion phase"]')
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Folder to store extracted frames")
    parser.add_argument("--frame_skip", type=int, default=1,
                        help="Process every Nth frame (1=every frame)")
    parser.add_argument("--confidence_threshold", type=float, default=0.7,
                        help="Minimum probability to consider a frame as recognized")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of classes in your trained model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model, load weights
    model = build_model(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    model.to(device)

    # Match the order used during training
    phases = [
        "Capsulorhexis phase",
        "I/A phase",
        "Phaco phase",
        "IOL insertion phase"
    ]
    if args.phase_name not in phases:
        print(f"[ERROR] Unknown phase: {args.phase_name}")
        return

    target_index = phases.index(args.phase_name)

    def transform_frame(bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.
        return tensor

    os.makedirs(args.output_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video_path}")
        return

    frame_idx = 0
    saved_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing '{args.video_path}' with {total_frames} frames...")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % args.frame_skip == 0:
            frame_tensor = transform_frame(frame_bgr).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(frame_tensor)         # shape [1,4]
                probs = torch.softmax(outputs, dim=1) # shape [1,4]
                max_prob, pred_class = torch.max(probs, 1)
                max_prob = max_prob.item()
                pred_class = pred_class.item()

            if max_prob >= args.confidence_threshold and pred_class == target_index:
                filename = f"frame_{frame_idx:06d}.jpg"
                out_path = os.path.join(args.output_dir, filename)
                cv2.imwrite(out_path, frame_bgr)
                saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"Done. Saved {saved_count} frames for phase '{args.phase_name}' to '{args.output_dir}'.")

if __name__ == "__main__":
    main()