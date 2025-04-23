import streamlit as st
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, MBartForConditionalGeneration, MBart50TokenizerFast
import torch
import warnings as ws

ws.filterwarnings('ignore')

def predict_step_vision(image):     #caption generator model1 - ViT
    model = VisionEncoderDecoderModel.from_pretrained("vision_encoder_decoder_model")               #using the pretrained model
    feature_extractor = ViTImageProcessor.from_pretrained("vision_encoder_decoder_model")
    tokenizer = AutoTokenizer.from_pretrained("vision_encoder_decoder_model")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                           #CUDA for mayuresh
    model.to(device)

    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    i_image = Image.open(image)
    if i_image.mode != "RGB":                               #imag3 processing - rgb - model requirement 
        i_image = i_image.convert(mode="RGB")

    pixel_values = feature_extractor(images=[i_image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)            #tokenisation
    preds = [pred.strip() for pred in preds]
    preds = [pred.split('<end>')[0].strip() for pred in preds]
    return preds[0]

def predict_step_blip(image):               #blip model
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
    processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    raw_image = Image.open(image)
    pixel_values = processor(images=raw_image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    generated_caption_cleaned = generated_caption.split(" - ")[0]
    return generated_caption_cleaned

# Function to translate text to Hindi and Tamil using MBart model
def translate_to_language(text, language_code):
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    tokenizer.src_lang = "en_XX"
    encoded_text = tokenizer(text, return_tensors="pt")
    
    # Translate to the specified language
    generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.lang_code_to_id[language_code])
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
    return translated_text[0]

def main():
    st.title("Image to Text Generation and Translation")
    
    model_option = st.selectbox("Select Model", ("Vision Encoder Decoder", "GIT-base COCO", "BLIP"))

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if model_option == "Vision Encoder Decoder":
            caption = predict_step_vision(uploaded_file)
        elif model_option == "BLIP":
            caption = predict_step_blip(uploaded_file)
        elif model_option == "GIT-base COCO":
            caption = predict_step_git_base_coco(uploaded_file)
        
        st.write("Generated Caption:", caption)

        # TRANSLATION
        translate_option = st.selectbox("Translate Caption", ("Hindi", "Tamil"))
        if translate_option == "Hindi":
            translated_caption = translate_to_language(caption, "hi_IN")
        elif translate_option == "Tamil":
            translated_caption = translate_to_language(caption, "te_IN")
        
        st.write(f"**Translated Caption ({translate_option}):**", translated_caption)
        st.balloons()

if __name__ == "__main__":
    main()