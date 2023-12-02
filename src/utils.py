import os
import torch
import torchvision
from data_loading import FaceDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    try:
        print("=> Saving checkpoint")
        torch.save(state, filename)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def load_checkpoint(checkpoint, model):
    try:
        print("=> Loading checkpoint")
        model.load_state_dict(checkpoint["state_dict"])
    except KeyError as e:
        print(f"Missing key in checkpoint data: {e}")

def get_loaders(train_dir, train_maskdir, val_dir, val_maskdir, batch_size, train_transform, val_transform, num_workers=4, pin_memory=True):
    train_ds = FaceDataset(image_dir=train_dir, mask_dir=train_maskdir, transform=train_transform)
    val_ds = FaceDataset(image_dir=val_dir, mask_dir=val_maskdir, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

    print(f"Accuracy: {num_correct/num_pixels*100:.2f}%")
    model.train()

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    os.makedirs(folder, exist_ok=True)
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, os.path.join(folder, f"pred_{idx}.png"))
        torchvision.utils.save_image(y.unsqueeze(1), os.path.join(folder, f"{idx}.png"))
    model.train()
