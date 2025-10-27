import json
import os
import re
import csv
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: str):
    """Load JSON file and extract model name."""
    filename = os.path.basename(path).split(".")[0]
    name = filename.replace("vsi_preds_", "").replace("_instruct", "").replace("_", "-")
    with open(path, "r") as f:
        return name, json.load(f)


def extract_answer(pred: str):
    """Extract the answer from prediction text."""
    index = pred.find("\nassistant\n")
    if index == -1:
        return pred.strip()
    return pred[index + len("\nassistant\n"):].strip()


def check_match(pred: str, gt: str, options: list = None, margin_percent: float = 5.0):
    """
    Check if prediction matches ground truth.
    Handles both multiple choice (letter-based) and numeric answers.
    
    Args:
        pred: Model prediction
        gt: Ground truth answer
        options: List of options for multiple choice questions
        margin_percent: Percentage margin for numeric answers (default: 5%)
    """
    pred_clean = extract_answer(pred).strip()
    gt_clean = gt.strip()
    
    if options and len(options) > 0:
        pred_parts = pred_clean.split()
        if pred_parts:
            first_part = pred_parts[0]
            letter_match = re.match(r'^([A-Za-z])[\.\)\:]?$', first_part)
            if letter_match:
                pred_letter = letter_match.group(1).upper()
                
                gt_letter = gt_clean[0].upper() if gt_clean and gt_clean[0].isalpha() else None
                
                if pred_letter and gt_letter:
                    return pred_letter == gt_letter
        
        return pred_clean.lower() == gt_clean.lower()
    
    else:
        try:
            pred_num = float(pred_clean)
            gt_num = float(gt_clean)
            
            # If ground truth is zero, use absolute difference
            if gt_num == 0:
                return abs(pred_num - gt_num) < 1e-2
            
            # Calculate percentage difference
            percent_diff = abs((pred_num - gt_num) / gt_num) * 100
            return percent_diff <= margin_percent
        except (ValueError, TypeError):
            return pred_clean.lower() == gt_clean.lower()


def compute_metrics(path: str, margin_percent: float = 5.0):
    """Compute metrics for a single model."""
    model_name, data = load_json(path)
    
    qaTypes = defaultdict(lambda: {"total": 0, "correct": 0})
    datasets = defaultdict(lambda: {"total": 0, "correct": 0})
    
    total_correct = 0
    total_questions = 0
    mc_correct = 0
    mc_total = 0
    numeric_correct = 0
    numeric_total = 0
    
    for sample in tqdm(data, desc=f"Processing {model_name}"):
        pred = sample["pred"]
        gt = sample["gt"]
        options = sample.get("options")
        question_type = sample.get("question_type", "unknown")
        dataset = sample.get("dataset", "unknown")
        
        is_correct = check_match(pred, gt, options, margin_percent)
        
        total_questions += 1
        if is_correct:
            total_correct += 1
        
        qaTypes[question_type]["total"] += 1
        if is_correct:
            qaTypes[question_type]["correct"] += 1
        
        datasets[dataset]["total"] += 1
        if is_correct:
            datasets[dataset]["correct"] += 1
        
        if options and len(options) > 0:
            mc_total += 1
            if is_correct:
                mc_correct += 1
        else:
            numeric_total += 1
            if is_correct:
                numeric_correct += 1
    
    return {
        "model_name": model_name,
        "overall": {
            "correct": total_correct,
            "total": total_questions,
            "accuracy": (total_correct / total_questions * 100) if total_questions > 0 else 0
        },
        "mc": {
            "correct": mc_correct,
            "total": mc_total,
            "accuracy": (mc_correct / mc_total * 100) if mc_total > 0 else 0
        },
        "numeric": {
            "correct": numeric_correct,
            "total": numeric_total,
            "accuracy": (numeric_correct / numeric_total * 100) if numeric_total > 0 else 0
        },
        "by_question_type": dict(qaTypes),
        "by_dataset": dict(datasets)
    }


def save_to_csv(all_results: list, output_dir: str):
    with open(os.path.join(output_dir, "overall_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Overall Accuracy", "Overall Correct", "Overall Total",
            "MC Accuracy", "MC Correct", "MC Total",
            "Numeric Accuracy", "Numeric Correct", "Numeric Total"
        ])
        
        for result in all_results:
            writer.writerow([
                result["model_name"],
                f"{result['overall']['accuracy']:.2f}",
                result['overall']['correct'],
                result['overall']['total'],
                f"{result['mc']['accuracy']:.2f}",
                result['mc']['correct'],
                result['mc']['total'],
                f"{result['numeric']['accuracy']:.2f}",
                result['numeric']['correct'],
                result['numeric']['total']
            ])
    
    all_qtypes = set()
    for result in all_results:
        all_qtypes.update(result["by_question_type"].keys())
    
    with open(os.path.join(output_dir, "question_type_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Question Type"] + [r["model_name"] for r in all_results]
        writer.writerow(header)
        
        for qtype in sorted(all_qtypes):
            row = [qtype]
            for result in all_results:
                if qtype in result["by_question_type"]:
                    stats = result["by_question_type"][qtype]
                    acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
                    row.append(f"{acc:.2f}")
                else:
                    row.append("0.00")
            writer.writerow(row)
    
    all_datasets = set()
    for result in all_results:
        all_datasets.update(result["by_dataset"].keys())
    
    with open(os.path.join(output_dir, "dataset_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Dataset"] + [r["model_name"] for r in all_results]
        writer.writerow(header)
        
        for dataset in sorted(all_datasets):
            row = [dataset]
            for result in all_results:
                if dataset in result["by_dataset"]:
                    stats = result["by_dataset"][dataset]
                    acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
                    row.append(f"{acc:.2f}")
                else:
                    row.append("0.00")
            writer.writerow(row)


def create_bar_charts(all_results: list, output_dir: str):
    """Create bar charts comparing model performance."""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    models = [r["model_name"] for r in all_results]
    overall_acc = [r["overall"]["accuracy"] for r in all_results]
    mc_acc = [r["mc"]["accuracy"] for r in all_results]
    numeric_acc = [r["numeric"]["accuracy"] for r in all_results]
    
    x = np.arange(len(models))
    width = 0.25
    
    bars1 = ax.bar(x - width, overall_acc, width, label='Overall', color=colors[0], alpha=0.8)
    bars2 = ax.bar(x, mc_acc, width, label='Multiple Choice', color=colors[1], alpha=0.8)
    bars3 = ax.bar(x + width, numeric_acc, width, label='Numeric', color=colors[2], alpha=0.8)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overall_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    all_qtypes = set()
    for result in all_results:
        all_qtypes.update(result["by_question_type"].keys())
    
    qtypes_sorted = sorted(all_qtypes)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(qtypes_sorted))
    width = 0.8 / len(all_results)
    
    for i, result in enumerate(all_results):
        accuracies = []
        for qtype in qtypes_sorted:
            if qtype in result["by_question_type"]:
                stats = result["by_question_type"][qtype]
                acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
                accuracies.append(acc)
            else:
                accuracies.append(0)
        
        offset = (i - len(all_results)/2 + 0.5) * width
        ax.bar(x + offset, accuracies, width, label=result["model_name"], 
               color=colors[i % len(colors)], alpha=0.8)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance by Question Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([qt.replace('_', ' ').title() for qt in qtypes_sorted], 
                       rotation=45, ha='right', fontsize=10)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "question_type_comparison.png"), dpi=300, bbox_inches='tight')
    print(f"   - {output_dir}/question_type_comparison.png")
    plt.close()
    
    all_datasets = set()
    for result in all_results:
        all_datasets.update(result["by_dataset"].keys())
    
    datasets_sorted = sorted(all_datasets)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(datasets_sorted))
    width = 0.8 / len(all_results)
    
    for i, result in enumerate(all_results):
        accuracies = []
        for dataset in datasets_sorted:
            if dataset in result["by_dataset"]:
                stats = result["by_dataset"][dataset]
                acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
                accuracies.append(acc)
            else:
                accuracies.append(0)
        
        offset = (i - len(all_results)/2 + 0.5) * width
        bars = ax.bar(x + offset, accuracies, width, label=result["model_name"], 
                     color=colors[i % len(colors)], alpha=0.8)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance by Dataset', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([ds.title() for ds in datasets_sorted], fontsize=11)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dataset_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()


def print_summary(all_results: list, margin_percent: float = 5.0):
    """Print summary of results."""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY".center(80))
    print("="*80)
    print(f"Numeric answer margin: ±{margin_percent}%".center(80))
    print("="*80)
    
    print(f"\n{'Model':<30s} {'Overall':>12s} {'MC':>12s} {'Numeric':>12s}")
    print("-"*80)
    
    for result in all_results:
        print(f"{result['model_name']:<30s} "
              f"{result['overall']['accuracy']:>11.2f}% "
              f"{result['mc']['accuracy']:>11.2f}% "
              f"{result['numeric']['accuracy']:>11.2f}%")
    
    print("\n" + "="*80)


def main(margin_percent: float = 5.0):
    """
    Main function to process all models.
    
    Args:
        margin_percent: Percentage margin for numeric answers (default: 5%)
    """
    eval_dir = "QwenEval"
    csv_dir = os.path.join(eval_dir, "csv")
    charts_dir = os.path.join(eval_dir, "charts")
    
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)
    
    json_files = [
        os.path.join(eval_dir, f) 
        for f in os.listdir(eval_dir)
        if f.startswith("vsi_preds_") and f.endswith(".json") and "_corrected" not in f
    ]
    
    if not json_files:
        print(f"❌ No prediction files found in {eval_dir}/")
        return
    
    print(f"\n📊 Evaluating {len(json_files)} model(s) with ±{margin_percent}% margin for numeric answers")
    print("="*80)
    
    all_results = []
    for json_file in sorted(json_files):
        result = compute_metrics(json_file, margin_percent)
        all_results.append(result)
    
    print_summary(all_results, margin_percent)
    
    print("\n💾 Saving results to CSV files...")
    save_to_csv(all_results, csv_dir)
    print(f"   ✓ {csv_dir}/overall_results.csv")
    print(f"   ✓ {csv_dir}/question_type_results.csv")
    print(f"   ✓ {csv_dir}/dataset_results.csv")
    
    print("\n📊 Creating bar charts...")
    create_bar_charts(all_results, charts_dir)
    print(f"   ✓ {charts_dir}/overall_comparison.png")
    print(f"   ✓ {charts_dir}/question_type_comparison.png")
    print(f"   ✓ {charts_dir}/dataset_comparison.png")
    
    print("\n✅ Analysis complete!\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate model predictions with customizable margin for numeric answers")
    parser.add_argument("--margin", type=float, default=5.0, 
                       help="Percentage margin for numeric answers (default: 5.0)")
    
    args = parser.parse_args()
    main(margin_percent=args.margin)
