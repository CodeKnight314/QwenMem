#!/usr/bin/env python3
"""
Comprehensive data analysis script for all JSON files in data/qa_jsons/
Analyzes each dataset and provides statistics about samples, scenes, and question-specific metrics.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
import statistics

def analyze_dataset(json_path, dataset_name):
    """Analyze a single dataset JSON file."""
    
    print(f"\n{'='*80}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'='*80}\n")
    
    print(f"Loading {json_path.name}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Basic statistics
    total_samples = len(data)
    print(f"Total number of samples: {total_samples:,}")
    
    # Dataset distribution
    datasets = [item.get('dataset', 'unknown') for item in data]
    dataset_counts = Counter(datasets)
    if len(dataset_counts) > 1:
        print(f"\nDataset distribution:")
        for dataset, count in dataset_counts.most_common():
            print(f"  {dataset}: {count:,} samples ({count/total_samples*100:.2f}%)")
    
    # Scene distribution
    scenes = [item.get('scene_name', 'unknown') for item in data]
    unique_scenes = len(set(scenes))
    scene_counts = Counter(scenes)
    print(f"\nScene statistics:")
    print(f"  Unique scenes: {unique_scenes:,}")
    if unique_scenes > 0:
        print(f"  Average samples per scene: {total_samples/unique_scenes:.2f}")
        print(f"  Most samples in a scene: {scene_counts.most_common(1)[0][1]}")
        print(f"  Least samples in a scene: {min(scene_counts.values())}")
    
    # Question type verification
    question_types = set(item.get('question_type', 'unknown') for item in data)
    print(f"\nQuestion types:")
    for qtype in question_types:
        count = sum(1 for item in data if item.get('question_type') == qtype)
        print(f"  {qtype}: {count:,} samples")
    
    # Answer analysis based on question type
    question_type = list(question_types)[0] if question_types else 'unknown'
    
    if 'abs_distance' in question_type or 'rel_distance' in question_type:
        # Distance-based questions
        answers = []
        for item in data:
            try:
                if isinstance(item.get('answer'), (int, float)):
                    distance = float(item['answer'])
                elif isinstance(item.get('answer'), str):
                    distance = float(item['answer'])
                else:
                    continue
                answers.append(distance)
            except (ValueError, TypeError):
                pass
        
        if answers:
            print(f"\nDistance statistics:")
            print(f"  Total valid distance values: {len(answers):,}")
            print(f"  Minimum distance: {min(answers):.2f} meters")
            print(f"  Maximum distance: {max(answers):.2f} meters")
            print(f"  Average distance: {sum(answers)/len(answers):.2f} meters")
            print(f"  Median distance: {sorted(answers)[len(answers)//2]:.2f} meters")
            
            # Distance ranges
            ranges = {
                "0-1m": sum(1 for d in answers if 0 <= d < 1),
                "1-2m": sum(1 for d in answers if 1 <= d < 2),
                "2-5m": sum(1 for d in answers if 2 <= d < 5),
                "5-10m": sum(1 for d in answers if 5 <= d < 10),
                "10m+": sum(1 for d in answers if d >= 10),
            }
            print(f"\nDistance range distribution:")
            for range_name, count in ranges.items():
                if count > 0:
                    print(f"  {range_name}: {count:,} samples ({count/len(answers)*100:.2f}%)")
    
    elif 'counting' in question_type:
        # Counting questions
        answers = []
        for item in data:
            try:
                if isinstance(item.get('answer'), (int, float)):
                    count = int(item['answer'])
                elif isinstance(item.get('answer'), str):
                    count = int(item['answer'])
                else:
                    continue
                answers.append(count)
            except (ValueError, TypeError):
                pass
        
        if answers:
            print(f"\nCount statistics:")
            print(f"  Total valid count values: {len(answers):,}")
            print(f"  Minimum count: {min(answers)}")
            print(f"  Maximum count: {max(answers)}")
            print(f"  Average count: {sum(answers)/len(answers):.2f}")
            print(f"  Median count: {sorted(answers)[len(answers)//2]}")
            
            count_dist = Counter(answers)
            print(f"\nCount distribution (top 10):")
            for count_val, freq in count_dist.most_common(10):
                print(f"  {count_val} object(s): {freq:,} samples ({freq/len(answers)*100:.2f}%)")
    
    elif 'direction' in question_type:
        # Direction questions
        answers = [item.get('answer', '') for item in data if item.get('answer')]
        answer_counts = Counter(answers)
        print(f"\nDirection answer distribution:")
        for direction, count in answer_counts.most_common():
            print(f"  {direction}: {count:,} samples ({count/len(answers)*100:.2f}%)")
    
    elif 'room_size' in question_type:
        # Room size questions
        answers = []
        for item in data:
            try:
                if isinstance(item.get('answer'), (int, float)):
                    size = float(item['answer'])
                elif isinstance(item.get('answer'), str):
                    size = float(item['answer'])
                else:
                    continue
                answers.append(size)
            except (ValueError, TypeError):
                pass
        
        if answers:
            print(f"\nRoom size statistics:")
            print(f"  Total valid size values: {len(answers):,}")
            print(f"  Minimum size: {min(answers):.2f} m²")
            print(f"  Maximum size: {max(answers):.2f} m²")
            print(f"  Average size: {sum(answers)/len(answers):.2f} m²")
            print(f"  Median size: {sorted(answers)[len(answers)//2]:.2f} m²")
            
            # Size ranges
            ranges = {
                "0-10 m²": sum(1 for s in answers if 0 <= s < 10),
                "10-20 m²": sum(1 for s in answers if 10 <= s < 20),
                "20-30 m²": sum(1 for s in answers if 20 <= s < 30),
                "30-50 m²": sum(1 for s in answers if 30 <= s < 50),
                "50-100 m²": sum(1 for s in answers if 50 <= s < 100),
                "100+ m²": sum(1 for s in answers if s >= 100),
            }
            print(f"\nSize range distribution:")
            for range_name, count in ranges.items():
                if count > 0:
                    print(f"  {range_name}: {count:,} samples ({count/len(answers)*100:.2f}%)")
    
    elif 'route_planning' in question_type:
        # Route planning questions
        answers = [item.get('answer', []) for item in data]
        answer_lengths = [len(a) if isinstance(a, list) else 0 for a in answers]
        valid_lengths = [l for l in answer_lengths if l > 0]
        
        if valid_lengths:
            print(f"\nRoute planning statistics:")
            print(f"  Total valid routes: {len(valid_lengths):,}")
            print(f"  Minimum route length: {min(valid_lengths)} actions")
            print(f"  Maximum route length: {max(valid_lengths)} actions")
            print(f"  Average route length: {sum(valid_lengths)/len(valid_lengths):.2f} actions")
            print(f"  Median route length: {sorted(valid_lengths)[len(valid_lengths)//2]} actions")
            
            length_dist = Counter(valid_lengths)
            print(f"\nRoute length distribution:")
            for length, freq in sorted(length_dist.items()):
                print(f"  {length} action(s): {freq:,} samples ({freq/len(valid_lengths)*100:.2f}%)")
        
        # Action type distribution
        all_actions = []
        for answer in answers:
            if isinstance(answer, list):
                all_actions.extend([a.lower().strip() for a in answer if isinstance(a, str)])
        action_counts = Counter(all_actions)
        if action_counts:
            print(f"\nAction type distribution:")
            for action, count in action_counts.most_common():
                print(f"  {action}: {count:,} occurrences")
    
    # Object-related statistics
    object_labels_list = []
    object_pairs = []
    for item in data:
        if 'object_labels' in item:
            labels = item['object_labels']
            if labels:
                object_labels_list.extend(labels)
                if len(labels) == 2:
                    pair = tuple(sorted(labels))
                    object_pairs.append(pair)
    
    if object_labels_list:
        unique_objects = len(set(object_labels_list))
        object_counts = Counter(object_labels_list)
        print(f"\nObject statistics:")
        print(f"  Unique object types: {unique_objects:,}")
        print(f"  Most common objects (top 10):")
        for obj, count in object_counts.most_common(10):
            print(f"    {obj}: {count:,} occurrences")
    
    if object_pairs:
        unique_pairs = len(set(object_pairs))
        pair_counts = Counter(object_pairs)
        print(f"\nObject pair statistics:")
        print(f"  Unique object pairs: {unique_pairs:,}")
        if unique_pairs > 0:
            print(f"  Most common pairs (top 10):")
            for pair, count in pair_counts.most_common(10):
                print(f"    {pair[0]} <-> {pair[1]}: {count:,} samples")
    
    # Samples per dataset-scene combination
    dataset_scene_combos = defaultdict(set)
    for item in data:
        dataset_scene_combos[item.get('dataset', 'unknown')].add(item.get('scene_name', 'unknown'))
    
    if len(dataset_scene_combos) > 1:
        print(f"\nScenes per dataset:")
        for dataset, scenes_set in dataset_scene_combos.items():
            print(f"  {dataset}: {len(scenes_set):,} unique scenes")
    
    return {
        'total_samples': total_samples,
        'unique_scenes': unique_scenes,
        'unique_datasets': len(dataset_counts),
        'question_type': question_type
    }

def main():
    """Analyze all JSON files in the data/qa_jsons directory."""
    
    data_dir = Path(__file__).parent.parent / "data" / "qa_jsons"
    
    if not data_dir.exists():
        print(f"Error: Directory not found at {data_dir}")
        return
    
    json_files = sorted(data_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return
    
    print(f"\n{'#'*80}")
    print(f"COMPREHENSIVE DATASET ANALYSIS")
    print(f"Found {len(json_files)} JSON file(s) to analyze")
    print(f"{'#'*80}")
    
    all_stats = {}
    
    for json_file in json_files:
        dataset_name = json_file.stem
        try:
            stats = analyze_dataset(json_file, dataset_name)
            all_stats[dataset_name] = stats
        except Exception as e:
            print(f"\nError analyzing {json_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY ACROSS ALL DATASETS")
    print(f"{'='*80}\n")
    
    total_samples_all = sum(s['total_samples'] for s in all_stats.values())
    total_scenes_all = sum(s['unique_scenes'] for s in all_stats.values())
    
    print(f"Total samples across all datasets: {total_samples_all:,}")
    print(f"Total unique scenes across all datasets: {total_scenes_all:,}")
    print(f"\nBreakdown by dataset file:")
    for name, stats in sorted(all_stats.items()):
        print(f"  {name}: {stats['total_samples']:,} samples, {stats['unique_scenes']:,} scenes")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()

