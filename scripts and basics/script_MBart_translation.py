from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

def translate_to_german_french(input_text):
    # Load MBart model and tokenizer
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    # Set source language as English
    tokenizer.src_lang = "en_XX"

    # Tokenize input text
    encoded_input = tokenizer(input_text, return_tensors="pt")

    # Generate translations to German
    generated_tokens_de = model.generate(
        **encoded_input,
        forced_bos_token_id=tokenizer.lang_code_to_id["de_DE"]
    )
    translations_de = tokenizer.batch_decode(generated_tokens_de, skip_special_tokens=True)

    # Generate translations to French
    generated_tokens_fr = model.generate(
        **encoded_input,
        forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"]
    )
    translations_fr = tokenizer.batch_decode(generated_tokens_fr, skip_special_tokens=True)

    return translations_de, translations_fr

def main():
    input_text = input("Enter the text in English: ")
    translations_de, translations_fr = translate_to_german_french(input_text)

    print("\nTranslated text in German:")
    for translation in translations_de:
        print(translation)

    print("\nTranslated text in French:")
    for translation in translations_fr:
        print(translation.encode('utf-8').decode('unicode-escape'))

if __name__ == "__main__":
    main()
