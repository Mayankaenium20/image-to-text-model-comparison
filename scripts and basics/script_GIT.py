from transformers import AutoProcessor, AutoModelForCausalLM
import requests
from PIL import Image

# Load processor and model
processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

# Generate a random image (as we don't have an uploaded image)
url = "https://t4.ftcdn.net/jpg/06/15/16/55/360_F_615165535_fJUsWZTZhCpeLBwBkpXMkIEhjTiCxEIu.jpg"
image = Image.open(requests.get(url, stream=True).raw)
pixel_values = processor(images=image, return_tensors="pt").pixel_values

# Generate some text using the model (not required for saving the model, but for demonstration)
generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Generated Caption:", generated_caption)

# Save the model
model.save_pretrained("git_base_coco_model")
