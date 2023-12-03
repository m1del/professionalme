import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from model_UNET_face_segmentation import UNET
from collections import OrderedDict


# Load the trained model
def load_trained_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNET(in_channels=3, out_channels=1)

    # Load state dict with handling for 'module.' prefix
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k  # remove `module.` prefix
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


# Preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((640, 480))  # Resize to match training data
    image = TF.to_tensor(image)
    image = TF.normalize(image, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    return image.unsqueeze(0)  # Add batch dimension


# Perform segmentation
def segment_image(model, image_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        mask = model(image_tensor)
        mask = torch.sigmoid(mask)
        mask = (
            mask.squeeze().cpu().numpy()
        )  # Remove batch dimension and transfer to numpy
    return mask


# Convert mask to image and save
def save_mask(mask, save_path):
    mask = mask > 0.5  # Apply threshold
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    mask_image.save(save_path)


# Main function to run the process
def main(image_path, model_path, output_path):
    model = load_trained_model(model_path)
    image_tensor = preprocess_image(image_path)
    mask = segment_image(model, image_tensor)
    save_mask(mask, output_path)


if __name__ == "__main__":
    main("test.png", "model.pth", "output.png")
