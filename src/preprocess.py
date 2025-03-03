# I'm defining transformations for training and validation sets here.

import torchvision.transforms as T

def get_train_transforms():
    # Here, I'm applying some data augmentation for training
    return T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms():
    # For validation, I'm just resizing and normalizing
    return T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])