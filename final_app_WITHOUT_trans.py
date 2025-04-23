import streamlit as st
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration, AutoProcessor, AutoModelForCausalLM
import torch

def predict_step_vision(image):
    # Load VisionEncoderDecoderModel
    model = VisionEncoderDecoderModel.from_pretrained("vision_encoder_decoder_model")
    feature_extractor = ViTImageProcessor.from_pretrained("vision_encoder_decoder_model")
    tokenizer = AutoTokenizer.from_pretrained("vision_encoder_decoder_model")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    max_length = 32
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    i_image = Image.open(image)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    pixel_values = feature_extractor(images=[i_image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    preds = [pred.split('<end>')[0].strip() for pred in preds]
    return preds[0]

def predict_step_blip(image):
    # Load BLIP model directly
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blip_model.to(device)

    raw_image = Image.open(image).convert('RGB')
    inputs = blip_processor(raw_image, return_tensors="pt")

    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

def predict_step_git_base_coco(image):
    # Load git_base_coco model directly
    processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    raw_image = Image.open(image)
    pixel_values = processor(images=raw_image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption

def main():
    st.title("Image to Text Generation")
    
    model_option = st.selectbox("Select Model", ("Vision Encoder Decoder", "GIT-base COCO", "BLIP"))

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if model_option == "Vision Encoder Decoder":
            caption = predict_step_vision(uploaded_file)
        elif model_option == "BLIP":
            caption = predict_step_blip(uploaded_file)
        elif model_option == "git_base_coco":
            caption = predict_step_git_base_coco(uploaded_file)

        st.write("Generated Caption:", caption)

if __name__ == "__main__":
    main()
