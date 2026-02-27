#! /usr/bin/env python

from itertools import islice
from load_datasets import EVQADataset, AmberDiscDataset
import json
import re



# Never move this before the os environ update, otherwise the model loading will fail because it won't find the models in the specified cache directory
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
# import JSON
# Step 2: Load the Model and Processor
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

processor = AutoProcessor.from_pretrained(
    model_name, padding_side="left", trust_remote_code=True
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)

from PIL import Image

# Step 3: define the query
SEARCH_FOR_TOOL_PROMPT = """You are an expert multimodal routing agent. Your primary function is to analyze an input image alongside a user query, determine the precise analytical intent, and route the request to the correct specialized tool.

AVAILABLE TOOLS & TASK ALIGNMENT:
1. Retriever
   - Target Task: Encyclopedic-VQA (EVQA).
   - Trigger Condition: Queries requiring external knowledge synthesis or the identification of specific, named entities (e.g., historical monuments, biological species classifications, specific artworks).

2. Detector
   - Target Task: Discriminative Queries (Amber / Hallucination Checks).
   - Trigger Condition: Binary or verification queries focused on object presence, absence, or spatial existence (e.g., "Is there a [object] in the image?", "Can you see a [object]?").

3. tool_ocr_extractor
   - Target Task: Document VQA (DocVQA).
   - Trigger Condition: Queries demanding the extraction, reading, or comprehension of alphanumeric data from dense visual text sources (e.g., scanned documents, invoices, receipts, schematics).

OPERATIONAL DIRECTIVES:
- Evaluate the linguistic structure of the query and the overarching visual context of the image.
- Classify the task into exactly one of the three established domains.
- Output a strictly formatted JSON routing command to trigger the designated tool. Do not generate conversational padding or supplementary text outside of the JSON structure.

OUTPUT SCHEMA:
{
  "classification": "<ENCYCLOPEDIC | DISCRIMINATIVE | DOCUMENT>",
  "selected_tool": "<Retriever | Detector | tool_ocr_extractor>",
  "extraction_target": "<The specific object to detect, text to read, or entity to identify>",
  "reasoning": "<A strictly technical, one-sentence justification for the tool selection>"
}
"""


def perform_inference(image, question):
    # Prepare the input
    messages = [
        {"role": "system", "content": SEARCH_FOR_TOOL_PROMPT},
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}],
        },
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


# Step 3: get dataset
# evqa_dataset = EVQADataset()
amber_dataset = AmberDiscDataset()
# print(amber_dataset[0]['question'])
# response = perform_inference(amber_dataset[0]['image'], amber_dataset[0]['question'])
# print("Question:", amber_dataset[0]['question'])
# print("Model Response:", response)

tool_choosed = []

for amber_sample in islice(amber_dataset, 10):
    # print(amber_sample["question"], amber_sample["image"])
    response = perform_inference(amber_sample["image"], amber_sample["question"])
    print("Question:", amber_sample["question"])
    print("Model Response:", response)
    try:
        selected_tool = json.loads(response).get("selected_tool")
    except json.JSONDecodeError:
        match = re.search(r'"selected_tool"\s*:\s*"([^"]+)"', response)
        selected_tool = match.group(1) if match else None
    tool_choosed.append(selected_tool)
    print("Selected Tool:", selected_tool)


# for evqa_sample in islice(evqa_dataset, 1):
#     # print(evqa_sample["question"], evqa_sample["image"])
#     response = perform_inference(evqa_sample["image"], evqa_sample["question"])
#     print("Question:", evqa_sample["question"])
#     print("Model Response:", response)
