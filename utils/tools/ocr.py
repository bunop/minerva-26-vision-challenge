import os

import easyocr
import numpy as np
from PIL import Image

# declare stuff
env_vars = {
    "HF_HUB_CACHE": "./MLLM_challenge/hf_models",
    "HF_HOME": "./MLLM_challenge/hf_models",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
}
os.environ.update(env_vars)
ocr_reader = easyocr.Reader(['en'], model_storage_directory=env_vars["HF_HUB_CACHE"], download_enabled=False) 

def tool_ocr_extractor(pil_image: Image.Image) -> str:
    """
    Legge il testo da un'istanza PIL.Image e lo restituisce come stringa.
    """
    
    
    img_array = np.array(pil_image)
    
    results = ocr_reader.readtext(img_array)
    # print("OCR Results:", results)  # Debug: print OCR results
    
    # Extract and concatenate the text from the OCR results
    extracted_text = " ".join([text for _, text, _ in results])
    
    return extracted_text

if __name__ == "__main__":
    # Example usage
    image_path = "data/images_stop.jpg"  # Replace with your image path
    pil_image = Image.open(image_path)
    
    extracted_text = tool_ocr_extractor(pil_image)
    print("Extracted Text:", extracted_text)