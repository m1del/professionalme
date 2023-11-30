from flask import Flask, request, send_file
import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline
import torch

# Initialize Flask app
app = Flask(__name__)

# Initialize the Stable Diffusion model
device = "cpu"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path)
pipe = pipe.to(device)


@app.route("/transform-image", methods=["POST"])
def transform_image():
    # Extract image URL and prompt from the request
    image_url = request.json.get("image_url")
    prompt = request.json.get("prompt")

    # Download and process the image
    response = requests.get(image_url)
    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    init_image = init_image.resize((768, 512))

    # Generate the transformed image
    transformed_images = pipe(
        prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5
    ).images
    output_image = transformed_images[0]

    # Save the image to a buffer
    buf = BytesIO()
    output_image.save(buf, format="PNG")
    buf.seek(0)

    # Return the image as a response
    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)
