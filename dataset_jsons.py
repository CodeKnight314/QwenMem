import json, os, glob
from tqdm import tqdm
import numpy as np
from typing import List, Set
import argparse

SCANNET_DATA_ROOT   = "/projects/vig/Datasets/ScanNet"
ARKITSCENE_DATA_ROOT = "/projects/vig/Datasets/ARKitScenes/3dod/Training"
SCANNETPP_DATA_ROOT = "/projects/vig/Datasets/ScanNetpp/data"

CHOICES = [
    "object_rel_direction", 
]

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def evenly_sample_frames(frames: List[str], n_frames: int):
    if len(frames) == 0:
        return []
    idxs = np.linspace(0, len(frames) - 1, n_frames).astype(int)
    return [frames[idx] for idx in idxs]

def load_existing_ids(file_path: str) -> Set[str]:
    """Load IDs of already processed samples to enable resume."""
    if not os.path.exists(file_path):
        return set()
    
    existing_ids = set()
    with open(file_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if "id" in entry:
                    existing_ids.add(entry["id"])
            except json.JSONDecodeError:
                continue
    
    return existing_ids

def create_data_json(choices: List[str], resume: bool = True):
    data = []
    # ---- merge all QA files ----
    for choice in choices:
        qa_path = os.path.join("/projects/vig/Datasets/Spatial-Reasoning/QA-Pairs/clean_qa/", f"{choice}.json")
        if not os.path.exists(qa_path):
            print(f"❌ Missing QA JSON: {os.path.abspath(qa_path)}")
            continue
        json_data = load_json(qa_path)
        data.extend(json_data)

    # ---- setup output file ----
    out_file_path = os.path.join(os.getcwd(), "qwenmem_struct2d.json")
    
    # ---- check for existing progress ----
    existing_ids = set()
    if resume:
        existing_ids = load_existing_ids(out_file_path)
        if existing_ids:
            print(f"🔄 Resuming: Found {len(existing_ids)} already processed samples")
    
    # Open in append mode if resuming, write mode otherwise
    mode = "a" if (resume and existing_ids) else "w"
    out_file = open(out_file_path, mode)
    
    # ---- process data ----
    skipped = 0
    processed = 0
    errors = 0

    for item in tqdm(data, desc="Building dataset"):
        dataset  = item["dataset"].lower().strip()
        scene    = item["scene_name"]
        sample_id = f"{dataset}_{scene}"
        
        # Skip if already processed
        if sample_id in existing_ids:
            skipped += 1
            continue
        
        question = item["question"].strip()
        answer   = item["answer"].strip()

        # absolute roots (they already start with "/projects/...")
        if dataset == "scannet":
            scene_path = os.path.join(SCANNET_DATA_ROOT, "scans_uncomp", scene, "color")
            frame_glob = "*.jpg"
        elif dataset == "arkitscenes":
            scene_path = os.path.join(ARKITSCENE_DATA_ROOT, scene, scene + "_frames", "lowres_wide")
            frame_glob = "*.png"
        elif dataset == "scannetpp":
            scene_path = os.path.join(SCANNETPP_DATA_ROOT, scene, "dslr", "resized_images")
            frame_glob = "*.JPG"
        else:
            print(f"⚠️ Unknown dataset type: {dataset}")
            errors += 1
            continue

        if not os.path.isdir(scene_path):
            print(f"⚠️ Scene folder not found: {os.path.abspath(scene_path)}")
            errors += 1
            continue

        all_frames = sorted(glob.glob(os.path.join(scene_path, frame_glob)))
        if len(all_frames) == 0:
            print(f"⚠️ No frame files in {scene_path}")
            errors += 1
            continue

        frames_32 = evenly_sample_frames(all_frames, 32)

        # LlamaFactory-compatible format
        entry = {
            "id": sample_id,
            "messages": [
                {
                    "role": "user",
                    "content": "<video>" + question  # ← Simple string, not nested array
                },
                {
                    "role": "assistant",
                    "content": answer  # ← Simple string, not nested array
                }
            ],
            "videos": [frames_32]  # ← Top-level videos key (list of video frame lists)
        }
        out_file.write(json.dumps(entry) + "\n")
        out_file.flush()  # Ensure data is written immediately
        processed += 1

    out_file.close()
    
    # ---- summary ----
    print(f"\n{'='*50}")
    print(f"✅ Saved dataset to {out_file_path}")
    print(f"📊 Summary:")
    print(f"   - Total items in QA files: {len(data)}")
    print(f"   - Previously processed (skipped): {skipped}")
    print(f"   - Newly processed: {processed}")
    print(f"   - Errors/warnings: {errors}")
    print(f"   - Total in output file: {skipped + processed}")
    print(f"{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create training data JSONL with resume capability")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, overwriting existing file")
    args = parser.parse_args()
    
    create_data_json(CHOICES, resume=not args.no_resume)