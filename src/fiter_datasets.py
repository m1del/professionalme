import os
from PIL import Image

def filter_images_by_dimensions(directory_path, target_width=640, target_height=480):
    filtered_images = []

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        try:
            with Image.open(file_path) as img:
                width, height = img.size
                if width == target_width and height == target_height:
                    filtered_images.append(file_path)
        except (IOError, OSError, Image.UnidentifiedImageError):
            # Handle cases where the file is not a valid image
            pass

    return filtered_images

# Replace 'your_directory_path' with the path to your directory containing photos
directory_path = '../data/raw/Humans'
filtered_images = filter_images_by_dimensions(directory_path)

print("Filtered Images:")
for img_path in filtered_images:
    print filtered_images.
    print(img_path)
