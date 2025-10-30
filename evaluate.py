import os
import json
import re
import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from datasets import load_dataset
import gc

def sample_frames(video_path: str, n_frames: int = 8):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        print(f"⚠️ Empty or broken video: {video_path}")
        return []
    idxs = np.linspace(0, total - 1, min(n_frames, total)).astype(int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    return frames

def run_vsi_eval(
    model_name: str, out_dir: str, n_frames: int = 16, max_samples: int = None, batch_size: int = 4
):
    DATA_ROOT = "/projects/vig/Datasets/VSI-Bench/videos"
    DATASET_PATH = "nyu-visionx/VSI-Bench"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"vsi_preds_{model_name.split('/')[-1].replace('-', '_').replace('.', '_').lower()}.json",
    )

    print("📦 Loading VSI-Bench...")
    vsi = load_dataset(DATASET_PATH)
    split = vsi["test"]
    if max_samples:
        split = split.select(range(max_samples))

    total_questions = len(split)
    print(f"Total questions in split: {total_questions}")

    completed_ids = set()
    existing_results = []

    if os.path.exists(out_path):
        print(f"🔄 Found existing results at {out_path}, loading...")
        try:
            with open(out_path, "r") as f:
                existing_results = json.load(f)
                completed_ids = {r["id"] for r in existing_results}
            print(f"✅ Loaded {len(completed_ids)} completed samples.")
        except Exception as e:
            print(f"⚠️ Failed to load existing JSON ({e}), starting fresh.")

    if len(completed_ids) >= total_questions:
        print(
            f"🎉 All {total_questions} questions already completed for {model_name}. Skipping."
        )
        return

    print(f"🚀 Loading model: {model_name}")
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=True, use_fast=True
    )
    
    # Set left padding for decoder-only models during batch generation
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()

    print("🗂️ Grouping questions by video...")
    videos = defaultdict(list)
    for row in split:
        if f"vsi_{row['id']}" in completed_ids:
            continue
        video_path = os.path.join(DATA_ROOT, row["dataset"], f"{row['scene_name']}.mp4")
        videos[video_path].append(row)

    print(f"Total unique videos to process: {len(videos)}")

    results = existing_results.copy()
    new_results = []

    pbar = tqdm(total=total_questions, desc="Evaluating videos")
    for i in range(len(results)):
        pbar.update(1)
        
    with torch.inference_mode(), pbar:
        for video_path, samples in videos.items():
            if not os.path.exists(video_path):
                print(f"⚠️ Missing: {video_path}")
                continue

            frames = sample_frames(video_path, n_frames)
            if not frames:
                continue

            samples_to_process = []
            for row in samples:
                if f"vsi_{row['id']}" in completed_ids:
                    pbar.update(1)
                    continue
                samples_to_process.append(row)
            
            if not samples_to_process:
                pbar.update(len(samples))
                continue

            for batch_start in range(0, len(samples_to_process), batch_size):
                batch_rows = samples_to_process[batch_start:batch_start + batch_size]
                
                batch_prompts = []
                batch_metadata = []
                
                for row in batch_rows:
                    q = row["question"]
                    opts = row["options"] or []
                    gt = row["ground_truth"]
                    
                    pre_prompt = "These are frames of a video."
                    if opts:
                        options_str = " ".join(opts)
                        prompt = f"{pre_prompt} {q} {options_str}\nAnswer with the option’s letter from the given choices directly."
                    else:
                        prompt = f"{pre_prompt} {q}\nPlease answer the question using a single word or phrase."
                    
                    batch_prompts.append(prompt)
                    batch_metadata.append({
                        "row": row,
                        "opts": opts,
                        "gt": gt,
                        "q": q
                    })
                
                all_text_prompts = []
                for prompt in batch_prompts:
                    messages = [
                        {
                            "role": "user",
                            "content": (
                                [{"type": "image", "image": frame} for frame in frames]
                                + [{"type": "text", "text": prompt}]
                            ),
                        }
                    ]
                    text_prompt = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    all_text_prompts.append(text_prompt)
                
                # Batch process with padding
                inputs = processor(
                    text=all_text_prompts, 
                    images=[frames] * len(all_text_prompts),
                    return_tensors="pt",
                    padding=True
                ).to(DEVICE)
                
                outputs = model.generate(
                    **inputs, 
                    do_sample=False, 
                    max_new_tokens=32,
                    pad_token_id=processor.tokenizer.pad_token_id
                )
                
                generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
                predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                for meta, pred in zip(batch_metadata, predictions):
                    pred = pred.strip()
                    row = meta["row"]
                    opts = meta["opts"]
                    gt = meta["gt"]
                    q = meta["q"]
                    qid = f"vsi_{row['id']}"

                    if opts:
                        # Improved matching for MCA: expect letter, but handle variations
                        pred_upper = pred.upper()
                        letter_match = re.match(r'^([A-D])[\.\)\:\s]?', pred_upper)
                        if letter_match:
                            pred_letter = letter_match.group(1)
                            # Find matching option
                            pred_text = next(
                                (opt for opt in opts if opt.startswith(f"{pred_letter}.")),
                                pred
                            )
                        else:
                            pred_text = next(
                                (opt for opt in opts if opt.split('. ', 1)[-1].lower() in pred.lower()),
                                pred
                            )
                        pred_text = pred_text.strip()

                        pred_content = pred_text.split('. ', 1)[-1].lower()
                        gt_content = gt.split('. ', 1)[-1].lower() if '.' in gt else gt.lower()
                        correct = pred_content == gt_content or pred_text.lower() == gt.lower()
                    else:
                        pred_text = pred
                        try:
                            correct = abs(float(pred_text) - float(gt)) < 1e-2
                        except Exception:
                            correct = pred_text.lower().strip() == gt.lower().strip()

                    result = {
                        "id": qid,
                        "video": video_path,
                        "question": q,
                        "options": opts,
                        "pred": pred_text,
                        "gt": gt,
                        "match": int(correct),
                        "question_type": row.get("question_type", "unknown"),
                        "dataset": row["dataset"],
                    }

                    new_results.append(result)

                    if len(new_results) % 25 == 0:
                        with open(out_path, "w") as f:
                            json.dump(results + new_results, f, indent=2)
                    pbar.update(1)
                    gc.collect()
                    torch.cuda.empty_cache()

    results += new_results
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved {len(results)} total predictions to {out_path}")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    models = [
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen3-VL-2B-Instruct",
        "Qwen/Qwen3-VL-4B-Instruct",
        "Qwen/Qwen3-VL-8B-Instruct",
    ]

    for m in models:
        run_vsi_eval(m, out_dir="vsi_outputs/", n_frames=32, batch_size=4)