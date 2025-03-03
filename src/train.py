import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # so we can save figures without needing a display
import matplotlib.pyplot as plt

from dataset import CataractPhaseDataset
from model import build_model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for frames, labels in tqdm(dataloader, desc="Training", leave=False):
        frames, labels = frames.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * frames.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total if total > 0 else 0
    epoch_acc = correct / total if total > 0 else 0
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for frames, labels in tqdm(dataloader, desc="Validation", leave=False):
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * frames.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total if total > 0 else 0
    epoch_acc = correct / total if total > 0 else 0
    return epoch_loss, epoch_acc

def plot_metrics(train_losses, val_losses, train_accs, val_accs, epoch, output_dir):
    """
    Save a matplotlib figure showing the training/validation curves up to 'epoch'.
    """
    epochs_range = range(1, epoch + 1)

    plt.figure(figsize=(10,4))

    # Plot the losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over epochs")
    plt.legend()

    # Plot the accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, label="Train Acc")
    plt.plot(epochs_range, val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over epochs")
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"training_plot_epoch_{epoch}.png")
    plt.savefig(plot_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to Cataract-1k-Phase with 'videos/' and 'annotations/'")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--frame_skip", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--num_classes", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading dataset...")
    train_dataset = CataractPhaseDataset(
        root_dir=args.data_dir,
        transform=None,
        frame_skip=args.frame_skip
    )
    print(f"  => Found {len(train_dataset)} samples.")
    
    # We'll reuse the same dataset for 'val_dataset', but you can split if needed
    val_dataset = train_dataset

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print("Building model...")
    model = build_model(num_classes=args.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = validate_one_epoch(model, val_loader, criterion, device)

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accs.append(t_acc)
        val_accs.append(v_acc)

        print(f"  Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.4f}")
        print(f"  Val   Loss: {v_loss:.4f} | Val   Acc: {v_acc:.4f}")

        # Save the model checkpoint (overwrite)
        ckpt_path = os.path.join(args.output_dir, "model_latest.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  -> Model saved to {ckpt_path}")

        # Save a training plot for the epochs so far
        plot_metrics(train_losses, val_losses, train_accs, val_accs, epoch, args.output_dir)
        print(f"  -> Plot saved for epoch {epoch}\n")

    print("Training complete.")

if __name__ == "__main__":
    main()