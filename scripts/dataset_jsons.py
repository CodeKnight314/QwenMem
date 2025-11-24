import json, os, glob
from tqdm import tqdm
import numpy as np
from typing import List, Set
import argparse

SCANNET_DATA_ROOT   = "/projects/vig/Datasets/ScanNet"
ARKITSCENE_DATA_ROOT = "/projects/vig/Datasets/ARKitScenes/3dod/Training"
SCANNETPP_DATA_ROOT = "/projects/vig/Datasets/ScanNetpp/data"

CHOICES = [
    "object_counting", 
    "object_abs_distance", 
    "object_rel_distance",
    "object_rel_direction",
    "room_size_estimation"
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

def create_data_json(choices: List[str], resume: bool = True, nframes: int = 32, nsamples: int = None):
    if nsamples is not None:
        remaining_samples = nsamples
    else:
        remaining_samples = None

    data = []

    # ---- setup output file ----
    out_file_path = os.path.join(os.getcwd(), f"qwenmem_nframes_{nframes}.json")
    
    # ---- check for existing progress ----
    existing_ids = set()
    if resume:
        existing_ids = load_existing_ids(out_file_path)
        if existing_ids:
            print(f"🔄 Resuming: Found {len(existing_ids)} already processed samples")
            if nsamples is not None:
                remaining_samples = nsamples - len(existing_ids)
    
    if nsamples is not None and remaining_samples <= 0:
        print(f"🎉 All {nsamples} samples already processed. Exiting.")
        return
    
    mode = "a" if (resume and existing_ids) else "w"
    out_file = open(out_file_path, mode)
    
    skipped = 0
    processed = 0
    errors = 0

    # Track samples collected per choice and available samples per choice
    samples_per_choice = {choice: 0 for choice in choices}
    available_samples_per_choice = {}
    choice_data = {}
    choice_indices = {}  # Track current index for each choice

    # ---- Load all QA files and count available samples ----
    print("\n" + "="*60)
    print("📊 PHASE 1: Loading and analyzing QA files")
    print("="*60)
    
    for choice in tqdm(choices, desc="Loading QA files"):
        qa_path = os.path.join("/projects/vig/Datasets/Spatial-Reasoning/QA-Pairs/clean_qa/", f"{choice}.json")
        if not os.path.exists(qa_path):
            print(f"❌ Missing QA JSON: {os.path.abspath(qa_path)}")
            available_samples_per_choice[choice] = 0
            choice_data[choice] = []
            choice_indices[choice] = 0
            continue
        
        json_data = load_json(qa_path)
        choice_data[choice] = json_data
        choice_indices[choice] = 0
        
        # Count available samples (excluding already processed ones)
        available_count = 0
        for item in json_data:
            dataset = item["dataset"].lower().strip()
            scene = item["scene_name"]
            sample_id = f"{dataset}_{scene}"
            if sample_id not in existing_ids:
                available_count += 1
        
        available_samples_per_choice[choice] = available_count
        print(f"  {choice}: {available_count} available samples (total: {len(json_data)})")
    
    total_available = sum(available_samples_per_choice.values())
    print(f"\n📈 Total available samples across all choices: {total_available}")
    
    if nsamples is None:
        print("⚠️  No sample limit specified. Processing all available samples.")
        nsamples = total_available
        remaining_samples = total_available
    
    print(f"🎯 Target samples: {nsamples}")
    print(f"📋 Remaining samples to collect: {remaining_samples}")
    print("="*60 + "\n")

    if nsamples is not None:
        print("\n" + "="*60)
        print("📊 PHASE 2: Even sampling from each choice")
        print("="*60)
        
        n_samples_per_choice = remaining_samples // len(choices)
        print(f"📐 Initial target per choice: {n_samples_per_choice} samples\n")
        
        for choice in tqdm(choices, desc="Even sampling"):
            if choice not in choice_data or len(choice_data[choice]) == 0:
                continue
            
            json_data = choice_data[choice]
            collected = 0
            start_idx = choice_indices[choice]
            
            for i in range(start_idx, len(json_data)):
                if collected >= n_samples_per_choice:
                    break
                
                item = json_data[i]
                dataset = item["dataset"].lower().strip()
                scene = item["scene_name"]
                sample_id = f"{dataset}_{scene}"
                
                if sample_id in existing_ids:
                    skipped += 1
                    continue
                
                question = item["question"].strip()
                answer = item["answer"].strip()

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

                frames_32 = evenly_sample_frames(all_frames, nframes)

                image_tags = "<image>" * nframes
                entry = {
                    "messages": [
                        {
                            "content": image_tags + question,
                            "role": "user"
                        },
                        {
                            "content": answer,
                            "role": "assistant"
                        }
                    ],
                    "images": frames_32
                }
                
                out_file.write(json.dumps(entry) + "\n")
                out_file.flush()
                processed += 1
                collected += 1
                samples_per_choice[choice] += 1
                choice_indices[choice] = i + 1
            
            print(f"  {choice}: Collected {collected} samples (target was {n_samples_per_choice})")
        
        total_collected = sum(samples_per_choice.values())

        remaining_samples = remaining_samples - total_collected
        print(f"\n📊 After even sampling: {total_collected} samples collected, {remaining_samples} remaining")
        print("="*60 + "\n")

        # ---- PHASE 3: Redistribute remaining quota if needed ----
        if remaining_samples > 0:
            print("\n" + "="*60)
            print("📊 PHASE 3: Redistributing remaining quota")
            print("="*60)
            
            iteration = 0
            while remaining_samples > 0:
                iteration += 1
                print(f"\n🔄 Redistribution iteration {iteration}")
                
                available_choices = []
                for choice in choices:
                    remaining_in_choice = available_samples_per_choice[choice] - samples_per_choice[choice]
                    if remaining_in_choice > 0:
                        available_choices.append((choice, remaining_in_choice))
                
                if not available_choices:
                    print("⚠️  No more samples available in any choice. Stopping.")
                    break
                
                if iteration == 1:
                    print(f"📋 Choices with remaining samples:")
                    for choice, remaining_count in available_choices:
                        print(f"  {choice}: {remaining_count} samples remaining")
                
                total_remaining_available = sum(count for _, count in available_choices)
                print(f"📐 Allocating {remaining_samples} samples among {len(available_choices)} choices (total available: {total_remaining_available})")
                
                iteration_collected = 0
                for choice, remaining_count in available_choices:
                    if remaining_samples <= 0:
                        break
                    
                    proportion = remaining_count / total_remaining_available if total_remaining_available > 0 else 0
                    allocation = min(int(remaining_samples * proportion), remaining_count)
                    if allocation == 0 and remaining_samples > 0 and remaining_count > 0:
                        allocation = min(remaining_samples, remaining_count)
                    
                    if allocation <= 0:
                        continue
                    
                    print(f"  Processing {choice}: allocating {allocation} more samples")
                    
                    json_data = choice_data[choice]
                    collected = 0
                    start_idx = choice_indices[choice]
                    
                    for i in range(start_idx, len(json_data)):
                        if collected >= allocation or remaining_samples <= 0:
                            break
                        
                        item = json_data[i]
                        dataset = item["dataset"].lower().strip()
                        scene = item["scene_name"]
                        sample_id = f"{dataset}_{scene}"
                        
                        if sample_id in existing_ids:
                            skipped += 1
                            continue
                        
                        question = item["question"].strip()
                        answer = item["answer"].strip()

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

                        frames_32 = evenly_sample_frames(all_frames, nframes)

                        image_tags = "<image>" * nframes
                        entry = {
                            "messages": [
                                {
                                    "content": image_tags + question,
                                    "role": "user"
                                },
                                {
                                    "content": answer,
                                    "role": "assistant"
                                }
                            ],
                            "images": frames_32
                        }
                        
                        out_file.write(json.dumps(entry) + "\n")
                        out_file.flush()
                        processed += 1
                        collected += 1
                        samples_per_choice[choice] += 1
                        remaining_samples -= 1
                        iteration_collected += 1
                        choice_indices[choice] = i + 1
                    
                    print(f"    Collected {collected} additional samples from {choice}")
                
                print(f"  Iteration {iteration} collected: {iteration_collected} samples, {remaining_samples} remaining")
                
                if iteration_collected == 0:
                    print("⚠️  No samples collected in this iteration. Stopping.")
                    break
            
            print("="*60 + "\n")

    print("\n" + "="*60)
    print("📊 FINAL SUMMARY")
    print("="*60)
    print(f"✅ Total samples processed: {processed}")
    print(f"⏭️  Samples skipped (already processed): {skipped}")
    print(f"❌ Errors encountered: {errors}")
    print(f"\n📋 Samples per choice:")
    for choice in choices:
        print(f"  {choice}: {samples_per_choice[choice]} samples")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create training data JSONL with resume capability")
    parser.add_argument("--nframes", type=int, default=32, help="Number of frames to sample")
    parser.add_argument("--nsamples", type=int, default=None, help="Number of samples to gather")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, overwriting existing file")
    args = parser.parse_args()
    
    create_data_json(CHOICES, resume=not args.no_resume, nframes=args.nframes, nsamples=args.nsamples if args.nsamples is not None else None)