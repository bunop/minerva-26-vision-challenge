#! /usr/bin/env python

import json
import re
import cv2
import numpy as np
from utils.tools.object_detector import Detector
import os
import argparse

env_vars = {
    "HF_HUB_CACHE": "/leonardo_scratch/fast/tra26_minwinsc/MLLM_challenge/hf_models",
    "HF_HOME": "/leonardo_scratch/fast/tra26_minwinsc/MLLM_challenge/hf_models",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
}
os.environ.update(env_vars)

from PIL import Image, ImageDraw

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


def perform_inference(image, question, system_prompt):
    # Prepare the input
    messages = [
        {"role": "system", "content": system_prompt},
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


def parse_selected_tool(response_text):
    try:
        data = json.loads(response_text)
        if isinstance(data, dict):
            return data.get("selected_tool")
    except json.JSONDecodeError:
        pass

    match = re.search(r'"selected_tool"\s*:\s*"([^"]+)"', response_text)
    return match.group(1) if match else None


def to_jsonable_detector_output(detector_output):
    jsonable = []
    for item in detector_output:
        box_values = []
        for box in item.get("boxes", []):
            if hasattr(box, "tolist"):
                box_values.append([float(v) for v in box.tolist()])
            else:
                box_values.append([float(v) for v in box])

        score_values = []
        for score in item.get("scores", []):
            if hasattr(score, "item"):
                score_values.append(float(score.item()))
            else:
                score_values.append(float(score))

        label_values = [str(label) for label in item.get("labels", [])]

        jsonable.append(
            {
                "boxes": box_values,
                "scores": score_values,
                "labels": label_values,
            }
        )
    return jsonable


def draw_bb(image, results):
    if image is None or not results:
        print("No image or detection results to draw.")
        return None

    pred = results[0] if isinstance(results, list) else results
    boxes = pred.get("boxes", [])
    scores = pred.get("scores", [])
    labels = pred.get("labels", [])

    draw = ImageDraw.Draw(image)

    for idx, box in enumerate(boxes):
        if hasattr(box, "tolist"):
            box = box.tolist()
        x1, y1, x2, y2 = [float(v) for v in box]

        score = scores[idx]
        if hasattr(score, "item"):
            score = float(score.item())
        else:
            score = float(score) if score is not None else 0.0

        label = str(labels[idx]) if idx < len(labels) else "object"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 12)), f"{label}: {score:.2f}", fill="red")
    display = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    window_name = "Amber Detection (press ESC to close)"
    cv2.imshow(window_name, display)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            break
    cv2.destroyWindow(window_name)
    return True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system_prompt", type=str, default=SEARCH_FOR_TOOL_PROMPT)
    return parser.parse_args()


def main(system_prompt):
    detector = Detector()

    while True:
        image_path = input(" image path：").strip()
        if image_path.lower() == "q":
            break

        question = input(" question：").strip()
        if question.lower() == "q":
            break

        print(f" image_path={image_path}, question={question}")
        image = Image.open(image_path).convert("RGB")

        first_round_response = perform_inference(image, question, system_prompt)
        print("First-round response:", first_round_response)

        selected_tool = parse_selected_tool(first_round_response)
        detector_result = []
        if selected_tool and "Detector" in selected_tool:
            detector_result = detector.detect(image, question)
            print("Detector output:", detector_result)
            draw_bb(image.copy(), detector_result)
        else:
            print("First-round selected tool is not Detector.")

        detector_json = json.dumps(to_jsonable_detector_output(detector_result), ensure_ascii=False)
        second_round_input = (
            f"Original question: {question}\n"
            f"First-round response: {first_round_response}\n"
            f"Detector output: {detector_json}\n"
            "Use detector output as evidence in your reasoning."
        )
        second_round_response = perform_inference(image, second_round_input, system_prompt)
        print("Second-round response:", second_round_response)

    print("程序结束")


if __name__ == "__main__":
    args = get_args()
    main(args.system_prompt)
