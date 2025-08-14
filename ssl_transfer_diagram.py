import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure with subplots for the two stages
fig = plt.figure(figsize=(16, 10))

# Define colors
color_data = '#E8F4FD'
color_ssl = '#FFE5CC'
color_transfer = '#D4EDDA'
color_arrow = '#666666'

# ========== STAGE 1: SSL PRETEXT TASK ==========
ax1 = plt.subplot(2, 1, 1)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 5)
ax1.axis('off')
ax1.set_title('STAGE 1: Self-Supervised Learning (Rotation Prediction)', fontsize=16, fontweight='bold', pad=20)

# Original Data
data_box = FancyBboxPatch((0.5, 2), 1.5, 1.5, 
                          boxstyle="round,pad=0.1", 
                          facecolor=color_data, 
                          edgecolor='black', linewidth=2)
ax1.add_patch(data_box)
ax1.text(1.25, 2.75, 'Original Data\nX: (1797, 64)\nPixel values\nDigit labels NOT used!', 
         ha='center', va='center', fontsize=9, fontweight='bold')

# Rotation Dataset (4x expanded)
rot_box = FancyBboxPatch((2.5, 1.5), 1.5, 2.5,
                         boxstyle="round,pad=0.1",
                         facecolor=color_ssl,
                         edgecolor='black', linewidth=2)
ax1.add_patch(rot_box)
ax1.text(3.25, 3.3, 'Rotation Dataset', ha='center', fontweight='bold', fontsize=10)
ax1.text(3.25, 2.9, '(7188, 64)', ha='center', fontsize=9)
ax1.text(3.25, 2.5, '4 rotations each:\n0°, 90°, 180°, 270°', ha='center', fontsize=8)
ax1.text(3.25, 1.9, 'Labels: 0, 1, 2, 3\n(rotation angles)', ha='center', fontsize=8, style='italic')

# Neural Network
nn_box = FancyBboxPatch((4.5, 2), 2, 1.5,
                        boxstyle="round,pad=0.1",
                        facecolor='#F0F0F0',
                        edgecolor='black', linewidth=2)
ax1.add_patch(nn_box)
ax1.text(5.5, 3.1, 'TwoLayerNet', ha='center', fontweight='bold', fontsize=10)
ax1.text(5.5, 2.75, 'Input (64) →', ha='center', fontsize=9)
ax1.text(5.5, 2.45, 'Hidden (32) →', ha='center', fontsize=9, color='red', fontweight='bold')
ax1.text(5.5, 2.15, 'Output (4)', ha='center', fontsize=9)

# Trained on Rotation
train_box = FancyBboxPatch((7, 2), 1.5, 1.5,
                          boxstyle="round,pad=0.1",
                          facecolor=color_ssl,
                          edgecolor='black', linewidth=2)
ax1.add_patch(train_box)
ax1.text(7.75, 2.75, 'Trained to\npredict\nrotation angles', ha='center', va='center', fontsize=9)

# Arrows for Stage 1
arrow1 = FancyArrowPatch((2, 2.75), (2.5, 2.75), 
                        connectionstyle="arc3", 
                        arrowstyle='->', mutation_scale=20, 
                        color=color_arrow, linewidth=2)
ax1.add_patch(arrow1)

arrow2 = FancyArrowPatch((4, 2.75), (4.5, 2.75),
                        connectionstyle="arc3",
                        arrowstyle='->', mutation_scale=20,
                        color=color_arrow, linewidth=2)
ax1.add_patch(arrow2)

arrow3 = FancyArrowPatch((6.5, 2.75), (7, 2.75),
                        connectionstyle="arc3",
                        arrowstyle='->', mutation_scale=20,
                        color=color_arrow, linewidth=2)
ax1.add_patch(arrow3)

# Add important note
ax1.text(5.5, 1.2, '⚠️ Hidden layer learns visual features WITHOUT seeing digit labels!', 
         ha='center', fontsize=10, color='darkred', fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.3))

# ========== STAGE 2: TRANSFER LEARNING ==========
ax2 = plt.subplot(2, 1, 2)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 6)
ax2.axis('off')
ax2.set_title('STAGE 2: Transfer Learning (Digit Classification)', fontsize=16, fontweight='bold', pad=20)

# Original Data Again
data_box2 = FancyBboxPatch((0.5, 3), 1.5, 1.5,
                          boxstyle="round,pad=0.1",
                          facecolor=color_data,
                          edgecolor='black', linewidth=2)
ax2.add_patch(data_box2)
ax2.text(1.25, 3.75, 'Same Data\nX: (1797, 64)\n+ Labels y: (1797,)\nDigits 0-9', 
         ha='center', va='center', fontsize=9, fontweight='bold')

# Feature Extraction
extract_box = FancyBboxPatch((2.5, 3), 1.8, 1.5,
                            boxstyle="round,pad=0.1",
                            facecolor='#FFE5CC',
                            edgecolor='black', linewidth=2)
ax2.add_patch(extract_box)
ax2.text(3.4, 3.75, 'Extract Features\nnet.hidden_repr(X)\n↓\nssl_features:\n(1797, 32)', 
         ha='center', va='center', fontsize=9, fontweight='bold')

# Train/Test Split
split_box = FancyBboxPatch((4.8, 2.5), 1.8, 2.5,
                          boxstyle="round,pad=0.1",
                          facecolor='#E0E0E0',
                          edgecolor='black', linewidth=2)
ax2.add_patch(split_box)
ax2.text(5.7, 4.5, 'train_test_split', ha='center', fontweight='bold', fontsize=10)
ax2.text(5.7, 4.1, 'ssl_features + y', ha='center', fontsize=9)
ax2.text(5.7, 3.6, '↓', ha='center', fontsize=12)
ax2.text(5.7, 3.3, 'Train: 70%\n(1258, 32)', ha='center', fontsize=9, color='blue')
ax2.text(5.7, 2.8, 'Test: 30%\n(539, 32)', ha='center', fontsize=9, color='green')

# Logistic Regression
lr_box = FancyBboxPatch((7.2, 3), 1.5, 1.5,
                        boxstyle="round,pad=0.1",
                        facecolor=color_transfer,
                        edgecolor='black', linewidth=2)
ax2.add_patch(lr_box)
ax2.text(7.95, 3.75, 'Logistic\nRegression\n32 features →\n10 digit classes', 
         ha='center', va='center', fontsize=9, fontweight='bold')

# Arrows for Stage 2
arrow4 = FancyArrowPatch((2, 3.75), (2.5, 3.75),
                        connectionstyle="arc3",
                        arrowstyle='->', mutation_scale=20,
                        color=color_arrow, linewidth=2)
ax2.add_patch(arrow4)

arrow5 = FancyArrowPatch((4.3, 3.75), (4.8, 3.75),
                        connectionstyle="arc3",
                        arrowstyle='->', mutation_scale=20,
                        color=color_arrow, linewidth=2)
ax2.add_patch(arrow5)

arrow6 = FancyArrowPatch((6.6, 3.75), (7.2, 3.75),
                        connectionstyle="arc3",
                        arrowstyle='->', mutation_scale=20,
                        color=color_arrow, linewidth=2)
ax2.add_patch(arrow6)

# Comparison Note
ax2.text(5.7, 1.8, '📊 Compare: SSL features (32D) vs Raw pixels (64D)', 
         ha='center', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.3))

# Key insight boxes
ax2.text(1.25, 1.2, '💡 Key: Same 1797 samples throughout!\nJust different representations:', 
         ha='left', fontsize=10, fontweight='bold')
ax2.text(1.25, 0.7, '• Raw pixels: (1797, 64)', ha='left', fontsize=9)
ax2.text(1.25, 0.4, '• SSL features: (1797, 32)', ha='left', fontsize=9, color='red')
ax2.text(1.25, 0.1, 'Each row = same digit, different representation', ha='left', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig('/home/thh3/work/SSNN_lab/ssl_transfer_diagram.png', dpi=150, bbox_inches='tight')
plt.show()

print("Diagram saved as ssl_transfer_diagram.png")