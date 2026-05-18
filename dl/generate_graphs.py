import matplotlib.pyplot as plt
import numpy as np
import os

# Data extracted from the screenshot
classes = ['all', 'car', 'bus', 'truck', 'motorcycle', 'auto_rickshaw']
instances = [43021, 30349, 642, 6596, 4983, 451]
precision = [0.657, 0.714, 0.816, 0.710, 0.512, 0.531]
recall = [0.579, 0.824, 0.732, 0.649, 0.278, 0.412]
map50 = [0.616, 0.846, 0.782, 0.711, 0.313, 0.428]
map50_95 = [0.442, 0.620, 0.656, 0.553, 0.123, 0.256]

# Exclude 'all' for class-wise comparison charts to make them cleaner
plot_classes = classes[1:]
plot_precision = precision[1:]
plot_recall = recall[1:]
plot_map50 = map50[1:]
plot_instances = instances[1:]

os.makedirs("results_graphs", exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# 1. mAP@50 Class-wise Bar Chart
plt.figure(figsize=(10, 6))
bars = plt.bar(plot_classes, plot_map50, color=colors)
plt.title('Validation mAP@50 by Vehicle Class (Fine-Tuned YOLOv8n)', fontsize=14, pad=15)
plt.ylabel('mAP@50 Score', fontsize=12)
plt.ylim(0, 1.0)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom', fontweight='bold')
plt.savefig('results_graphs/map50_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Precision vs Recall Grouped Bar Chart
x = np.arange(len(plot_classes))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, plot_precision, width, label='Precision', color='#3498db')
rects2 = ax.bar(x + width/2, plot_recall, width, label='Recall', color='#2ecc71')

ax.set_ylabel('Scores', fontsize=12)
ax.set_title('Precision & Recall by Vehicle Class', fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(plot_classes)
ax.legend()
ax.set_ylim(0, 1.0)

for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
plt.savefig('results_graphs/precision_recall_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Class Instance Distribution (Pie Chart)
plt.figure(figsize=(8, 8))
plt.pie(plot_instances, labels=plot_classes, autopct='%1.1f%%', startangle=140, colors=colors, shadow=False)
plt.title('Dataset Instance Distribution (Total: 43,021)', fontsize=14, pad=20)
plt.savefig('results_graphs/dataset_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("[SUCCESS] Successfully generated 3 presentation-ready graphs in 'results_graphs/' folder.")
