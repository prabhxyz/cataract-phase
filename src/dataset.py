import os
import cv2
import csv
import torch
import numpy as np
from torch.utils.data import Dataset

class CataractPhaseDataset(Dataset):
    """
    Dataset for cataract surgery phase recognition that parses CSV files
    with start/end frames for each phase. Adapts to the folder structure:

        Cataract-1k-Phase/
          ├─ videos/
          │    ├─ case_XXXX.mp4
          │    └─ ...
          └─ annotations/
              ├─ case_XXXX/
              │    └─ case_XXXX_annotations_phases.csv
              └─ ...

    and filters to only 4 target phases by mapping from raw CSV 'comment'
    strings to the canonical 4-phase names.
    """

    def __init__(self, root_dir, transform=None, frame_skip=10):
        super().__init__()
        self.root_dir = root_dir
        
        # Where the .mp4 files live
        self.videos_dir = os.path.join(root_dir, "videos")
        # Where the CSV subfolders live
        self.annotations_dir = os.path.join(root_dir, "annotations")

        self.transform = transform
        self.frame_skip = frame_skip

        # The final 4 canonical phase names we want to keep
        self.allowed_phases = {
            "Capsulorhexis phase",
            "I/A phase",
            "Phaco phase",
            "IOL insertion phase"
        }

        # Map raw CSV 'comment' strings -> one of the allowed phases above
        # Anything not in this map is ignored.
        self.comment_to_phase = {
            "Capsulorhexis": "Capsulorhexis phase",
            "Phacoemulsification": "Phaco phase",
            "Irrigation/Aspiration": "I/A phase",
            "Lens Implantation": "IOL insertion phase",
            "Lens positioning": "IOL insertion phase"
        }

        self.all_videos = set()
        self.phase_map = {}  # { case_id_str: list of (startF, endF, canonical_phase) }
        self.samples = []    # final (video_path, frame_idx, canonical_phase)

        self._scan_videos()
        self._scan_annotations()
        self._build_samples()

        # Build a label map only from the phases we actually retained
        all_phase_names = sorted(list({s[2] for s in self.samples}))
        self.phase_label_map = {ph: i for i, ph in enumerate(all_phase_names)}

    def _scan_videos(self):
        """
        Scans the videos/ folder for case_XXXX.mp4 and collects valid case IDs in self.all_videos.
        """
        if not os.path.isdir(self.videos_dir):
            return
        
        for file in os.listdir(self.videos_dir):
            if file.endswith(".mp4"):
                base = file.replace(".mp4","")  # e.g. "case_4687"
                if base.startswith("case_"):
                    case_str = base[len("case_"):]
                    if case_str.isdigit():
                        self.all_videos.add(case_str)

    def _scan_annotations(self):
        """
        Walks annotations/ looking for subfolders named case_XXXX that contain
        case_XXXX_annotations_phases.csv. Each CSV is read to populate self.phase_map.
        """
        # e.g. path/to/Cataract-1k-Phase/annotations/case_4687/case_4687_annotations_phases.csv
        if not os.path.isdir(self.annotations_dir):
            return

        for case_folder in os.listdir(self.annotations_dir):
            case_path = os.path.join(self.annotations_dir, case_folder)
            if not os.path.isdir(case_path):
                continue

            # e.g. case_folder might be "case_4687"
            if case_folder.startswith("case_"):
                case_id_str = case_folder[len("case_"):]
                if not case_id_str.isdigit():
                    continue
                # Look for the CSV file named case_XXXX_annotations_phases.csv
                csv_filename = f"{case_folder}_annotations_phases.csv"
                csv_path = os.path.join(case_path, csv_filename)
                if os.path.isfile(csv_path):
                    self._read_phase_csv(case_id_str, csv_path)

    def _read_phase_csv(self, case_id_str, csv_path):
        """
        Reads lines from CSV with columns: [caseId, comment, frame, endFrame, sec, endSec]
        Only retains segments whose 'comment' maps to one of the 4 canonical phases we want.
        """
        if case_id_str not in self.phase_map:
            self.phase_map[case_id_str] = []

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_comment = row["comment"].strip()
                if raw_comment in self.comment_to_phase:
                    # Convert e.g. "Phacoemulsification" -> "Phaco phase"
                    canonical_phase = self.comment_to_phase[raw_comment]
                    start_f = int(row["frame"])
                    end_f = int(row["endFrame"])
                    self.phase_map[case_id_str].append((start_f, end_f, canonical_phase))
                else:
                    # This annotation (e.g. "Incision") is not in our 4-phase map, so skip it
                    pass

    def _build_samples(self):
        """
        Builds (video_path, frame_idx, canonical_phase) tuples from the segments in phase_map.
        We skip frames that exceed the total frames in the .mp4 file,
        and we step by self.frame_skip to reduce the sampling rate.
        """
        for case_id_str, segments in self.phase_map.items():
            # Only proceed if we actually have a video for this case
            if case_id_str not in self.all_videos:
                continue

            video_path = os.path.join(self.videos_dir, f"case_{case_id_str}.mp4")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                cap.release()
                continue
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # For each annotated segment, create samples
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
        """
        Returns a tuple: (frame_tensor, phase_label)
        """
        video_path, frame_idx, phase_name = self.samples[idx]
        phase_label = self.phase_label_map[phase_name]

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        cap.release()

        if not ret or frame_bgr is None:
            # If there's an error reading the frame, create a dummy black image
            frame_bgr = np.zeros((224,224,3), dtype=np.uint8)

        # Convert BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.transform:
            # If you use albumentations: transform(image=frame_rgb)["image"]
            # If you use TorchVision transforms, you might do: transform(Image.fromarray(frame_rgb))
            transformed = self.transform(image=frame_rgb)
            frame_tensor = transformed["image"]
        else:
            # Basic Torch tensor transform
            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.

        return frame_tensor, phase_label