"""
Visualize the distribution of problem types from SpaceR-151k.jsonl

This script analyzes the JSONL file and creates visualizations for:
- Problem type distribution
- Data type distribution
- Problem sub-type distribution
- Data source distribution
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
import argparse


def load_jsonl(file_path):
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                continue
    return data


def analyze_distribution(data, field_name):
    """Analyze distribution of a specific field."""
    values = [item.get(field_name, 'Unknown') for item in data]
    # Filter out empty strings
    values = [v if v else 'Unknown' for v in values]
    return Counter(values)


def plot_distribution(counter, title, output_file=None, top_n=None, figsize=(12, 6)):
    """Create a bar plot for the distribution."""
    plt.figure(figsize=figsize)
    
    # Sort by frequency
    items = counter.most_common(top_n) if top_n else counter.most_common()
    labels, counts = zip(*items) if items else ([], [])
    
    # Create bar plot
    colors = sns.color_palette("husl", len(labels))
    bars = plt.bar(range(len(labels)), counts, color=colors)
    
    plt.xlabel('Category', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({count/sum(counts)*100:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    
    plt.show()


def plot_pie_chart(counter, title, output_file=None, top_n=10):
    """Create a pie chart for the distribution."""
    plt.figure(figsize=(10, 8))
    
    # Get top N items and group rest as "Others"
    items = counter.most_common(top_n)
    labels, counts = zip(*items) if items else ([], [])
    
    # Calculate "Others" if there are more items
    if len(counter) > top_n:
        others_count = sum(counter.values()) - sum(counts)
        labels = list(labels) + ['Others']
        counts = list(counts) + [others_count]
    
    # Create pie chart
    colors = sns.color_palette("husl", len(labels))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors,
            startangle=90, textprops={'fontsize': 10})
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.axis('equal')
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    
    plt.show()


def print_statistics(counter, title):
    """Print statistics about the distribution."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    total = sum(counter.values())
    print(f"Total items: {total:,}")
    print(f"Unique categories: {len(counter)}")
    print(f"\nTop 20 categories:")
    print(f"{'Rank':<6} {'Category':<40} {'Count':<12} {'Percentage'}")
    print("-" * 80)
    
    for i, (category, count) in enumerate(counter.most_common(20), 1):
        percentage = (count / total) * 100
        print(f"{i:<6} {category:<40} {count:<12,} {percentage:>6.2f}%")


def create_combined_plot(data, output_file=None):
    """Create a combined plot with multiple distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('SpaceR-151k Dataset Distribution Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Problem Type
    problem_types = analyze_distribution(data, 'problem_type')
    items = problem_types.most_common(15)
    labels, counts = zip(*items) if items else ([], [])
    colors = sns.color_palette("husl", len(labels))
    axes[0, 0].bar(range(len(labels)), counts, color=colors)
    axes[0, 0].set_title('Problem Type Distribution (Top 15)', fontweight='bold')
    axes[0, 0].set_xlabel('Problem Type')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_xticks(range(len(labels)))
    axes[0, 0].set_xticklabels(labels, rotation=45, ha='right')
    for i, (bar, count) in enumerate(zip(axes[0, 0].patches, counts)):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{count:,}', ha='center', va='bottom', fontsize=8)
    
    # Data Type
    data_types = analyze_distribution(data, 'data_type')
    items = data_types.most_common()
    labels, counts = zip(*items) if items else ([], [])
    colors = sns.color_palette("Set2", len(labels))
    axes[0, 1].bar(range(len(labels)), counts, color=colors)
    axes[0, 1].set_title('Data Type Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Data Type')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_xticks(range(len(labels)))
    axes[0, 1].set_xticklabels(labels, rotation=45, ha='right')
    for i, (bar, count) in enumerate(zip(axes[0, 1].patches, counts)):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{count:,}', ha='center', va='bottom', fontsize=9)
    
    # Data Source
    data_sources = analyze_distribution(data, 'data_source')
    items = data_sources.most_common(15)
    labels, counts = zip(*items) if items else ([], [])
    colors = sns.color_palette("muted", len(labels))
    axes[1, 0].barh(range(len(labels)), counts, color=colors)
    axes[1, 0].set_title('Data Source Distribution (Top 15)', fontweight='bold')
    axes[1, 0].set_ylabel('Data Source')
    axes[1, 0].set_xlabel('Count')
    axes[1, 0].set_yticks(range(len(labels)))
    axes[1, 0].set_yticklabels(labels)
    axes[1, 0].invert_yaxis()
    for i, (bar, count) in enumerate(zip(axes[1, 0].patches, counts)):
        axes[1, 0].text(bar.get_width(), bar.get_y() + bar.get_height()/2.,
                       f' {count:,}', ha='left', va='center', fontsize=8)
    
    # Problem Sub-Type (only non-empty ones)
    sub_types = analyze_distribution(data, 'problem_sub_type')
    # Filter out Unknown/empty
    sub_types_filtered = {k: v for k, v in sub_types.items() if k != 'Unknown'}
    if sub_types_filtered:
        items = Counter(sub_types_filtered).most_common(15)
        labels, counts = zip(*items) if items else ([], [])
        colors = sns.color_palette("pastel", len(labels))
        axes[1, 1].barh(range(len(labels)), counts, color=colors)
        axes[1, 1].set_title('Problem Sub-Type Distribution (Top 15)', fontweight='bold')
        axes[1, 1].set_ylabel('Problem Sub-Type')
        axes[1, 1].set_xlabel('Count')
        axes[1, 1].set_yticks(range(len(labels)))
        axes[1, 1].set_yticklabels(labels)
        axes[1, 1].invert_yaxis()
        for i, (bar, count) in enumerate(zip(axes[1, 1].patches, counts)):
            axes[1, 1].text(bar.get_width(), bar.get_y() + bar.get_height()/2.,
                           f' {count:,}', ha='left', va='center', fontsize=8)
    else:
        axes[1, 1].text(0.5, 0.5, 'No sub-type data available',
                       ha='center', va='center', fontsize=12)
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved combined plot to {output_file}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize problem type distribution from JSONL file'
    )
    parser.add_argument(
        '--file',
        type=str,
        default='SpaceR-151k.jsonl',
        help='Path to JSONL file (default: SpaceR-151k.jsonl)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='QwenEval/visualizations',
        help='Output directory for plots (default: visualizations)'
    )
    parser.add_argument(
        '--field',
        type=str,
        default='all',
        choices=['all', 'problem_type', 'data_type', 'data_source', 'problem_sub_type'],
        help='Field to visualize (default: all)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=None,
        help='Show only top N categories'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Only print statistics without creating plots'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.file}...")
    data = load_jsonl(args.file)
    print(f"Loaded {len(data):,} items")
    
    if not data:
        print("No data loaded. Exiting.")
        return
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Analyze and visualize
    if args.field == 'all' and not args.no_plot:
        # Create combined plot
        create_combined_plot(data, output_dir / 'combined_distribution.png')
        
        # Print statistics for all fields
        for field in ['problem_type', 'data_type', 'data_source', 'problem_sub_type']:
            counter = analyze_distribution(data, field)
            print_statistics(counter, f"{field.replace('_', ' ').title()} Distribution")
    
    elif args.field == 'all':
        # Just print statistics
        for field in ['problem_type', 'data_type', 'data_source', 'problem_sub_type']:
            counter = analyze_distribution(data, field)
            print_statistics(counter, f"{field.replace('_', ' ').title()} Distribution")
    
    else:
        # Single field analysis
        counter = analyze_distribution(data, args.field)
        print_statistics(counter, f"{args.field.replace('_', ' ').title()} Distribution")
        
        if not args.no_plot:
            # Bar plot
            plot_distribution(
                counter,
                f"{args.field.replace('_', ' ').title()} Distribution",
                output_dir / f'{args.field}_distribution_bar.png',
                top_n=args.top_n
            )
            
            # Pie chart (for fewer categories)
            if len(counter) <= 30:
                plot_pie_chart(
                    counter,
                    f"{args.field.replace('_', ' ').title()} Distribution",
                    output_dir / f'{args.field}_distribution_pie.png',
                    top_n=10
                )
    
    print(f"\nVisualization complete! Check the '{output_dir}' directory for plots.")


if __name__ == "__main__":
    main()

