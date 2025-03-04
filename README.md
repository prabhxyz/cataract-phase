# Cataract Surgery Phase Recognition

This project implements an end-to-end pipeline for recognizing key phases in cataract surgery videos. It focuses on **four main phases**:

1. **Capsulorhexis phase**  
2. **I/A (Irrigation/Aspiration) phase**  
3. **Phaco phase**  
4. **IOL insertion phase**

The model is trained to classify frames from surgical videos into one of these four phases, and an inference script can extract frames for a chosen phase from new videos.

---

## Overview of the Approach

- **Dataset Preparation**  
  - Videos (`case_XXXX.mp4`) live under `videos/`.  
  - Corresponding annotations (with start/end frames for each phase) reside under `annotations/case_XXXX/case_XXXX_annotations_phases.csv`.  
  - Only segments labeled with one of the four target phases are used for training.

- **Model Architecture**  
  - We use **MobileNetV2** (from PyTorch’s `torchvision.models`), a lightweight but accurate CNN.  
  - Its final layer is replaced with a fully connected layer outputting **4** classes.  
  - This setup allows relatively fast training while maintaining solid performance.

- **Training Pipeline**  
  - We sample frames at fixed intervals (e.g., skip every 10 frames) to reduce redundancy.  
  - For each epoch, the model:
    1. Computes forward passes on training frames (with their phase labels).  
    2. Backpropagates the errors using CrossEntropyLoss.  
    3. Updates weights via Adam optimizer.  
  - We also run a validation loop after each epoch to check how well the model generalizes.  
  - After every epoch, we overwrite our `model_latest.pth` checkpoint, so it always reflects the most recent training state.  
  - The project saves a **matplotlib** figure of the training and validation curves each epoch for monitoring progress.

---

## Sample Results (5 Epochs)

Below is a sample training plot after **5 epochs**, showcasing how **loss** (left) and **accuracy** (right) evolved over training:

![Training Plot](https://raw.githubusercontent.com/prabhxyz/cataract-phase/refs/heads/main/output/training_plot_epoch_5.png)

### Observations
- **Loss** curves show both training (blue) and validation (orange) decreasing from around **0.06** to under **0.02** within the first few epochs, indicating the model quickly learned to distinguish phases.
- **Accuracy** climbs beyond **99%** by epoch 3–4 for both training and validation, suggesting the model is performing extremely well at classifying the frames.  
- The slight fluctuations in validation metrics (e.g. a small dip around epoch 3) are normal as the model fine-tunes its parameters. The overall high accuracy implies strong generalization to unseen data in the validation set.

---

## Usage

1. **Installation**  
   - Make sure to install all necessary dependencies (PyTorch, Torchvision, OpenCV, TQDM, Matplotlib, etc.).  
   - Example:  
     ```bash
     pip install -r requirements.txt
     ```

2. **Training**  
   - Place your data in the expected folder structure:
     ```
     Cataract-1k-Phase/
       ├─ videos/
       └─ annotations/
     ```
   - Run the training script (example command):
     ```bash
     python train.py \
       --data_dir Cataract-1k-Phase \
       --epochs 5 \
       --batch_size 8 \
       --frame_skip 10 \
       --output_dir checkpoints
     ```
   - This will print epoch-by-epoch results (loss, accuracy) for both training and validation sets, overwrite `model_latest.pth` each epoch, and generate a **training curve plot** file (e.g. `training_plot_epoch_1.png`, `training_plot_epoch_2.png`, etc.) after every epoch.

3. **Inference / Frame Extraction**  
   - Use the inference script to extract frames that match a **specific** phase with sufficient confidence:
     ```bash
     python inference.py \
       --model_path checkpoints/model_latest.pth \
       --video_path Cataract-1k-Phase/videos/case_1234.mp4 \
       --phase_name "Phaco phase" \
       --output_dir output \
       --frame_skip 1 \
       --confidence_threshold 0.7
     ```
   - This processes each frame, predicts which of the 4 phases it belongs to, and saves frames in the requested phase to the `output` folder if the confidence exceeds the given threshold.

---

## Conclusion

- **High-Level Summary**  
  1. **Data**: We only train and classify four specific cataract phases.  
  2. **Model**: A MobileNetV2 backbone fine-tuned to output 4 classes.  
  3. **Performance**: Rapidly converges to high accuracy, as shown in the sample 5-epoch training curves.  
  4. **Inference**: Allows pinpointing of frames for any desired phase from new video footage.

- **Why This Matters**  
  - Automating the identification of crucial cataract surgery steps can **aid surgeons** and **facilitate research** by rapidly segmenting videos into clinically relevant phases. This project demonstrates a practical baseline architecture that performs strongly on such tasks.

*Built with ❤️ by [Prabhdeep](https://github.com/prabhxyz)*