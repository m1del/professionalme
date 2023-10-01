# Architecture Overview
### Input Layer
Accepts user-uploaded selfies.
- Validates image (format, size, etc.)
### Facial Detection Layer
Uses a facial detection model to detect and center-crop the face.
- Pretrained model from OpenCV/DLib/whatever (transfer learning & hyperparemeter tuning with our own data).
- Detect face, obtain bounding box boundaries.
- Crop the image centered on the face.
  - Add padding if necessary.
### Preprocessing Layer
Any additional preprocessing needed to resize or normalize the image to be compatible with the stable diffusion model's input requrements.
### Stable Diffusion Layer
Transforms the processed image based on text prompts to generate a professional headshot.
- Load pre-trained stable diffusion model.
- Pass the preprocessed image and text prompts to transform the image.
### Post-processing Layer
Final adjustments, e.g. resizing to a standard dimension, color gradining, brigthness, contrast, etc.
### Output Layer
Return the final image.

##### Todo:
Flowcharts?
Description (tools, models, librries, e.g. how we are doing facial detections
Requirements (setup-tools, dependencies, maybe wheel)
Model documentation and citing.
