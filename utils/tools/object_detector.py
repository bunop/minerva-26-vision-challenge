import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
import spacy
import os
import re
# env_vars= {
#     "HF_HUB_CACHE": "./hf_models",
#     "HF_HOME": "./hf_models",
#     "TRANSFORMERS_OFFLINE": "1",
#     "HF_HUB_OFFLINE": "1",
#     # "HF_DATASETS_OFFLINE": "1",
# }
env_vars = {
    "HF_HUB_CACHE": "/leonardo_scratch/fast/tra26_minwinsc/MLLM_challenge/hf_models",
    "HF_HOME": "/leonardo_scratch/fast/tra26_minwinsc/MLLM_challenge/hf_models",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
}

os.environ.update(env_vars)


class Detector:
    _nlp = None

    def __init__(self):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(self.device)


    @staticmethod
    def get_entity_from_query(query: str) -> str:
        text = query.strip().rstrip("?.!").lower()

        # Fast path for common Amber-style questions:
        # "Is there a/an/the <object> ...".
        pattern = r"^is\s+there\s+(?:a|an|the)?\s*(.+?)(?:\s+(?:in|on|at|inside|within)\b.*)?$"
        match = re.match(pattern, text)
        if match:
            candidate = match.group(1).strip(" ,.;:")
            if candidate:
                return candidate

        if Detector._nlp is None:
            try:
                Detector._nlp = spacy.load("en_core_web_sm")
            except Exception:
                Detector._nlp = None

        if Detector._nlp is not None:
            doc = Detector._nlp(text)
            noun_chunks = [chunk.text.strip() for chunk in doc.noun_chunks if chunk.text.strip()]
            if noun_chunks:
                return noun_chunks[-1]

            nouns = [tok.lemma_.strip() for tok in doc if tok.pos_ in {"NOUN", "PROPN"} and tok.lemma_.strip()]
            if nouns:
                return " ".join(nouns)

        # Minimal fallback if spaCy model is unavailable.
        tokens = [tok for tok in re.findall(r"[a-zA-Z0-9'-]+", text) if tok not in {"is", "there", "a", "an", "the"}]
        return " ".join(tokens) if tokens else text

    @torch.inference_mode()
    def detect(self, image: Image.Image, query: str, box_threshold=0.4, text_threshold=0.3):
        entity: str = self.get_entity_from_query(query)
        inputs = self.processor(images=image, text=[entity], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )

        print("Detection results:", results)
        print("shape of results:", results[0])
        return results
    
if __name__ == "__main__":
    detector = Detector()
    
    # image_path = "/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/dataset/iNaturalist_2021/val/06214_Plantae_Tracheophyta_Liliopsida_Liliales_Smilacaceae_Smilax_bona-nox/09dd9c22-8cbc-4b19-bbdd-1003986c048c.jpg" # Replace with your image path
    image_path = "/leonardo/home/usertrain/a08trc0u/Vision_Challenge/data/amber_disc/images/AMBER_1.jpg" # Replace with your image path
    question = "sky"  # Replace with your question
# [
#     {
#         "id": 1005,
#         "image": "AMBER_1.jpg",
#         "query": "Is the sky sunny in this image?"
#     },
    image = Image.open(image_path).convert("RGB")
    result = detector.detect(image, question)
    