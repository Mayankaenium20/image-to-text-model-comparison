import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Save the model
model.save_pretrained("blip_image_captioning_model")

def generate_caption(img_url, text=None):
    # Load image from URL
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    # conditional image captioning
    if text:
        inputs = processor(raw_image, text, return_tensors="pt")
    else:
        inputs = processor(raw_image, return_tensors="pt")

    # Generate caption
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

if __name__ == "__main__":
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    text = "a photography of"

    # Generate caption with text
    caption_with_text = generate_caption(img_url, text)
    print("Caption with text:", caption_with_text)

    # Generate caption without text
    caption_without_text = generate_caption(img_url)
    print("Caption without text:", caption_without_text)
