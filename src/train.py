import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# If these imports differ, adjust to match your own module/file names:
from dataset import CataractPhaseDataset   # This dataset only yields the 4 phases
from model import build_model              # A MobileNetV2 or similar with num_classes=4

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    I run one epoch of training over the dataloader, returning the average loss and accuracy.
    """
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

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device):
    """
    I evaluate the model on the validation set, returning the average loss and accuracy.
    """
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

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to root folder containing 'video/' and 'annotations/'")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Where to save checkpoints")
    parser.add_argument("--frame_skip", type=int, default=10,
                        help="Frame sampling skip in the dataset (e.g., 10 picks 1 frame out of every 10)")
    # We fix the number of classes to 4 for the 4 recognized phases
    parser.add_argument("--num_classes", type=int, default=4, help="Number of output classes")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the training dataset
    train_dataset = CataractPhaseDataset(
        root_dir=args.data_dir,
        transform=None,        # or pass your own transformations
        frame_skip=args.frame_skip
    )
    print(f"Found {len(train_dataset)} total samples.")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # In many real workflows, you'd have a separate validation set or
    # a method to split the data. This minimal example reuses train_dataset
    # as validation. In practice, separate them for a real project.
    val_dataset = train_dataset
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Build the model with 4 classes
    model = build_model(num_classes=args.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Lists to store metrics each epoch for graphing
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Print the epoch metrics
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # Save (and overwrite) the model for each epoch
        # The user specifically asked for overwriting the same file every epoch.
        model_path = os.path.join(args.output_dir, "model_latest.pth")
        torch.save(model.state_dict(), model_path)
        print(f"  Overwrote model checkpoint -> {model_path}\n")

    # After training, plot the metrics over epochs
    epochs_range = range(1, args.epochs + 1)

    plt.figure(figsize=(10,4))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Val Loss")
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, label="Train Acc")
    plt.plot(epochs_range, val_accs, label="Val Acc")
    plt.title("Accuracy over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "training_plot.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Training complete! The training plot was saved at: {plot_path}")

if __name__ == "__main__":
    main()