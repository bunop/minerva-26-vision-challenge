#! /usr/bin/env python

import os

# declare stuff
env_vars = {
    "HF_HUB_CACHE": "./MLLM_challenge/hf_models",
    "HF_HOME": "./MLLM_challenge/hf_models",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
}
os.environ.update(env_vars)

# Never move this before the os environ update, otherwise the model loading will fail because it won't find the models in the specified cache directory
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Step 2: Load the Model and Processor
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

processor = AutoProcessor.from_pretrained(
    model_name,
    padding_side="left",
    trust_remote_code=True
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

from PIL import Image


# Step 3: Perform Basic Inference
def perform_inference(image_path, question):
    # Load the image
    image = Image.open(image_path)

    # Prepare the input
    messages = [
        {
            "role": "system",
            "content": "You are a multimodal agent. Answer the user's question based on the image."
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        videos=None,
        padding=True,
        return_tensors="pt",
        truncation=True,
    )
    inputs = inputs.to(model.device)

    # Generate the response
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=2048,
        use_cache=True,
    )

    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return response


image_path = "/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/dataset/iNaturalist_2021/val/06214_Plantae_Tracheophyta_Liliopsida_Liliales_Smilacaceae_Smilax_bona-nox/09dd9c22-8cbc-4b19-bbdd-1003986c048c.jpg"  # Replace with your image path

question = "What is shown in this image?"  # Replace with your question
response = perform_inference(image_path, question)
print("Response:", response)
