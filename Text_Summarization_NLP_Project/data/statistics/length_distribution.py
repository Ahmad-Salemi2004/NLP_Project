#!/usr/bin/env python3
"""
Generate length distribution plot for DialogSum dataset
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load statistics
stats_path = Path("data/statistics/dataset_stats.json")
with open(stats_path, 'r') as f:
    stats = json.load(f)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('DialogSum Dataset Statistics', fontsize=16)

# 1. Dialogue length distribution
dialogue_data = stats['statistics']['dialogue_stats']
ax1 = axes[0, 0]
categories = ['Mean', 'Median', 'Min', 'Max']
values = [
    dialogue_data['mean_words'],
    dialogue_data['median_words'],
    dialogue_data['min_words'],
    dialogue_data['max_words']
]
bars = ax1.bar(categories, values, color=['blue', 'green', 'red', 'orange'])
ax1.set_title('Dialogue Length (words)')
ax1.set_ylabel('Number of Words')
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{value:.1f}', ha='center', va='bottom')

# 2. Summary length distribution
summary_data = stats['statistics']['summary_stats']
ax2 = axes[0, 1]
categories = ['Mean', 'Median', 'Min', 'Max']
values = [
    summary_data['mean_words'],
    summary_data['median_words'],
    summary_data['min_words'],
    summary_data['max_words']
]
bars = ax2.bar(categories, values, color=['blue', 'green', 'red', 'orange'])
ax2.set_title('Summary Length (words)')
ax2.set_ylabel('Number of Words')
ax2.grid(True, alpha=0.3)

# Add value labels
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{value:.1f}', ha='center', va='bottom')

# 3. Compression ratio
comp_data = stats['statistics']['compression_ratio']
ax3 = axes[1, 0]
labels = ['Mean', 'Median', 'Min', 'Max']
values = [
    comp_data['mean'],
    comp_data['median'],
    comp_data['min'],
    comp_data['max']
]
bars = ax3.bar(labels, values, color=['blue', 'green', 'red', 'orange'])
ax3.set_title('Compression Ratio (Dialogue/Summary)')
ax3.set_ylabel('Ratio')
ax3.grid(True, alpha=0.3)

# Add value labels
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{value:.1f}x', ha='center', va='bottom')

# 4. Topics distribution
topics_data = stats['statistics']['topics_distribution']
ax4 = axes[1, 1]

# Sort topics by count
topics = list(topics_data.keys())
counts = list(topics_data.values())

# Sort both lists by counts
sorted_indices = np.argsort(counts)[::-1]
topics = [topics[i] for i in sorted_indices]
counts = [counts[i] for i in sorted_indices]

# Create horizontal bar chart
y_pos = np.arange(len(topics))
ax4.barh(y_pos, counts, color='skyblue')
ax4.set_yticks(y_pos)
ax4.set_yticklabels(topics)
ax4.set_xlabel('Number of Samples')
ax4.set_title('Topics Distribution')
ax4.grid(True, alpha=0.3, axis='x')

# Adjust layout
plt.tight_layout()

# Save figure
output_path = Path("data/statistics/length_distribution.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved plot to {output_path}")

# Show plot
plt.show()
