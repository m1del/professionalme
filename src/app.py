import gradio as gr
import numpy as np
import torch
from PIL import Image

from diffusers import AutoPipelineForInpainting

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
def save_mask(mask, save_path, image_path):
    mask = mask > 0.5  # Apply threshold
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))

    image = Image.open(image_path).convert("RGB")
    target_width, target_height = image.size
    resized_mask = mask_image.resize((target_width, target_height))

    resized_mask.save(save_path)


# Main function to run the process
def run_inference(image_path, model_path, output_path):
    model = load_trained_model(model_path)
    image_tensor = preprocess_image(image_path)
    mask = segment_image(model, image_tensor)
    save_mask(mask, output_path, image_path)


# Load Stable Diffusion model
pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
)
count = 0


def generate_photo(input_image):
    global count
    count += 1

    # Prompts for Stable Diffusion
    prompt = "Professional LinkedIn photo, person in a suit, realistic skin texture, photorealistic, hyper realism, 85mm portrait photography, hard rim lighting photography, centered, plain or blurred background"
    negative_prompt = "deformed, disfigured, poor details, bad anatomy"

    # Generate mask
    run_inference(input_image, "model.pth", "../results/mask.png")
    mask_image = Image.open("../results/mask.png").convert("RGB")

    # Process input image
    input_image = Image.open(input_image).convert("RGB")

    # Generate output image
    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_image,
        mask_image=mask_image,
        strength=0.99,
        num_inference_steps=20,
        guidance_scale=7.5,
    ).images

    # Save image
    images[0].save(f"../results/{count}.png")

    return images[0]


# Interface
demo = gr.Interface(
    fn=generate_photo,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Image(),
    live=True,
    title="ProfessionalMe",
    description="Upload a selfie.",
)

# Launch app
demo.launch(share=False)
