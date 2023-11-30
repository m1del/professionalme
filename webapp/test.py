import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline

device = "cpu"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path)
pipe = pipe.to(device)

url = "https://media.licdn.com/dms/image/D5603AQEyDwh8hntAdg/profile-displayphoto-shrink_400_400/0/1683381148217?e=1706745600&v=beta&t=4Nz_xajYDAC0Bay2GAUpUw5qIhKKgT7hVxAXTDhq-iA"

response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((768, 512))

prompt = "A fantasy landscape, trending on artstation"

images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
images[0].save("fantasy_landscape.png")