import os
import csv
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class CataractPhaseDataset(Dataset):
    """
    Focuses only on four phases:
      1) Capsulorhexis phase
      2) I/A phase
      3) Phaco phase
      4) IOL insertion phase

    We skip everything else in the CSV (like incisions, hydrodissection, etc.).
    """

    def __init__(self, root_dir, transform=None, frame_skip=10):
        super().__init__()
        self.root_dir = root_dir
        self.videos_dir = os.path.join(root_dir, "videos")
        self.annotations_dir = os.path.join(root_dir, "annotations")
        self.transform = transform
        self.frame_skip = frame_skip

        # The final 4 canonical phases
        self.allowed_phases = {
            "Capsulorhexis phase",
            "I/A phase",
            "Phaco phase",
            "IOL insertion phase"
        }

        # Map from CSV 'comment' -> one of the above
        # Adjust the left side strings to match your actual CSV fields
        self.comment_to_phase = {
            "Capsulorhexis": "Capsulorhexis phase",
            "Irrigation/Aspiration": "I/A phase",
            "Phacoemulsification": "Phaco phase",
            "Lens Implantation": "IOL insertion phase",
            "Lens positioning": "IOL insertion phase"
        }

        self.all_videos = set()
        self.phase_map = {}
        self.samples = []

        self._scan_videos()
        self._scan_annotations()
        self._build_samples()

        # Build label map from what we actually found
        all_phase_names = sorted(list({s[2] for s in self.samples}))
        self.phase_label_map = {ph: i for i, ph in enumerate(all_phase_names)}

    def _scan_videos(self):
        if not os.path.isdir(self.videos_dir):
            return
        for file in os.listdir(self.videos_dir):
            if file.endswith(".mp4") and file.startswith("case_"):
                case_str = file.replace(".mp4","").split("_")[-1]
                if case_str.isdigit():
                    self.all_videos.add(case_str)

    def _scan_annotations(self):
        if not os.path.isdir(self.annotations_dir):
            return
        for folder in os.listdir(self.annotations_dir):
            folder_path = os.path.join(self.annotations_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            if folder.startswith("case_"):
                case_id_str = folder[len("case_"):]
                if not case_id_str.isdigit():
                    continue
                csv_filename = f"{folder}_annotations_phases.csv"
                csv_path = os.path.join(folder_path, csv_filename)
                if os.path.isfile(csv_path):
                    self._read_phase_csv(case_id_str, csv_path)

    def _read_phase_csv(self, case_id_str, csv_path):
        if case_id_str not in self.phase_map:
            self.phase_map[case_id_str] = []
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_comment = row["comment"].strip()
                if raw_comment in self.comment_to_phase:
                    canonical_phase = self.comment_to_phase[raw_comment]
                    start_f = int(row["frame"])
                    end_f = int(row["endFrame"])
                    self.phase_map[case_id_str].append((start_f, end_f, canonical_phase))

    def _build_samples(self):
        for case_id_str, segments in self.phase_map.items():
            if case_id_str not in self.all_videos:
                continue
            video_path = os.path.join(self.videos_dir, f"case_{case_id_str}.mp4")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                cap.release()
                continue
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            for (start_f, end_f, phase_name) in segments:
                if start_f >= total_frames:
                    continue
                if end_f >= total_frames:
                    end_f = total_frames - 1
                for fidx in range(start_f, end_f + 1, self.frame_skip):
                    if fidx < total_frames:
                        self.samples.append((video_path, fidx, phase_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, frame_idx, phase_name = self.samples[idx]
        phase_label = self.phase_label_map[phase_name]

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        cap.release()

        if not ret or frame_bgr is None:
            frame_bgr = np.zeros((224,224,3), dtype=np.uint8)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=frame_rgb)
            frame_tensor = transformed["image"]
        else:
            frame_tensor = torch.from_numpy(frame_rgb).permute(2,0,1).float() / 255.
        return frame_tensor, phase_label