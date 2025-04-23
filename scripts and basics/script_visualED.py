from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

# Load model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Save model and tokenizer
model.save_pretrained("vision_encoder_decoder_model")
feature_extractor.save_pretrained("vision_encoder_decoder_model")
tokenizer.save_pretrained("vision_encoder_decoder_model")
