import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline

device = "cpu"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path)
pipe = pipe.to(device)

url = "https://media.licdn.com/dms/image/D4E03AQGLfcAWhwLwQw/profile-displayphoto-shrink_400_400/0/1692482316403?e=1706745600&v=beta&t=vVvr3F3lhf-8hPdkQZTOM2GwY59CzJtXnlg2vbBUNkI"

response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
# init_image = init_image.resize((768, 512))

prompt = "A professional LinkedIn photo"

images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
images[0].save("generated_photo.png")