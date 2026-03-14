import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

def plot_cnn_architecture():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Input
    input_box = patches.FancyBboxPatch((0.5, 4), 1.5, 1, boxstyle="round,pad=0.1", facecolor='lightblue', edgecolor='black')
    ax.add_patch(input_box)
    ax.text(1.25, 4.5, 'Input\n4 × 2048', ha='center', va='center', fontsize=10)

    # Conv Block 1
    conv1_box = patches.FancyBboxPatch((2.5, 4), 1.5, 1, boxstyle="round,pad=0.1", facecolor='lightgreen', edgecolor='black')
    ax.add_patch(conv1_box)
    ax.text(3.25, 4.5, 'Conv Block 1', ha='center', va='center', fontsize=10)

    # Conv Block 2
    conv2_box = patches.FancyBboxPatch((4.5, 4), 1.5, 1, boxstyle="round,pad=0.1", facecolor='lightgreen', edgecolor='black')
    ax.add_patch(conv2_box)
    ax.text(5.25, 4.5, 'Conv Block 2', ha='center', va='center', fontsize=10)

    # Conv Block 3
    conv3_box = patches.FancyBboxPatch((6.5, 4), 1.5, 1, boxstyle="round,pad=0.1", facecolor='lightgreen', edgecolor='black')
    ax.add_patch(conv3_box)
    ax.text(7.25, 4.5, 'Conv Block 3', ha='center', va='center', fontsize=10)

    # Pooling/Global Pooling
    pool_box = patches.FancyBboxPatch((8.5, 4), 1.5, 1, boxstyle="round,pad=0.1", facecolor='lightyellow', edgecolor='black')
    ax.add_patch(pool_box)
    ax.text(9.25, 4.5, 'Pooling /\nGlobal Pooling', ha='center', va='center', fontsize=10)

    # Fully Connected Output
    fc_box = patches.FancyBboxPatch((10.5, 4), 1.5, 1, boxstyle="round,pad=0.1", facecolor='lightcoral', edgecolor='black')
    ax.add_patch(fc_box)
    ax.text(11.25, 4.5, 'Fully Connected\nOutput', ha='center', va='center', fontsize=10)

    # Softmax Classes
    softmax_box = patches.FancyBboxPatch((10.5, 2), 1.5, 1, boxstyle="round,pad=0.1", facecolor='lightpink', edgecolor='black')
    ax.add_patch(softmax_box)
    ax.text(11.25, 2.5, 'Softmax\nClasses', ha='center', va='center', fontsize=10)

    # Arrows
    ax.arrow(2, 4.5, 0.5, 0, head_width=0.05, head_length=0.1, fc='black', ec='black')
    ax.arrow(4, 4.5, 0.5, 0, head_width=0.05, head_length=0.1, fc='black', ec='black')
    ax.arrow(6, 4.5, 0.5, 0, head_width=0.05, head_length=0.1, fc='black', ec='black')
    ax.arrow(8, 4.5, 0.5, 0, head_width=0.05, head_length=0.1, fc='black', ec='black')
    ax.arrow(10, 4.5, 0.5, 0, head_width=0.05, head_length=0.1, fc='black', ec='black')
    ax.arrow(11.25, 3.5, 0, -0.5, head_width=0.05, head_length=0.1, fc='black', ec='black')

    plt.title('CNN Architecture Diagram', fontsize=14)
    plt.savefig('results/cnn_architecture_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_ais_pipeline():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # FIUS Ultrasonic Sensor
    sensor_box = patches.FancyBboxPatch((0.5, 6), 2, 1, boxstyle="round,pad=0.1", facecolor='lightblue', edgecolor='black')
    ax.add_patch(sensor_box)
    ax.text(1.5, 6.5, 'FIUS Ultrasonic\nSensor', ha='center', va='center', fontsize=10)

    # ADC Waveform + FFT Input
    adc_box = patches.FancyBboxPatch((3.5, 6), 2, 1, boxstyle="round,pad=0.1", facecolor='lightgreen', edgecolor='black')
    ax.add_patch(adc_box)
    ax.text(4.5, 6.5, 'ADC Waveform +\nFFT Input', ha='center', va='center', fontsize=10)

    # Preprocessing
    preproc_box = patches.FancyBboxPatch((6.5, 6), 2, 1, boxstyle="round,pad=0.1", facecolor='lightyellow', edgecolor='black')
    ax.add_patch(preproc_box)
    ax.text(7.5, 6.5, 'Preprocessing', ha='center', va='center', fontsize=10)

    # CNN Classifier
    cnn_box = patches.FancyBboxPatch((9.5, 6), 2, 1, boxstyle="round,pad=0.1", facecolor='lightcoral', edgecolor='black')
    ax.add_patch(cnn_box)
    ax.text(10.5, 6.5, 'CNN Classifier', ha='center', va='center', fontsize=10)

    # Pulse Confidence Aggregation
    agg_box = patches.FancyBboxPatch((6.5, 4), 2, 1, boxstyle="round,pad=0.1", facecolor='lightpink', edgecolor='black')
    ax.add_patch(agg_box)
    ax.text(7.5, 4.5, 'Pulse Confidence\nAggregation', ha='center', va='center', fontsize=10)

    # Final Object Label
    label_box = patches.FancyBboxPatch((9.5, 4), 2, 1, boxstyle="round,pad=0.1", facecolor='lightcyan', edgecolor='black')
    ax.add_patch(label_box)
    ax.text(10.5, 4.5, 'Final Object\nLabel', ha='center', va='center', fontsize=10)

    # AIS Action Block
    action_box = patches.FancyBboxPatch((9.5, 2), 2, 1, boxstyle="round,pad=0.1", facecolor='lightgray', edgecolor='black')
    ax.add_patch(action_box)
    ax.text(10.5, 2.5, 'AIS Action\n(Stop/Avoid/Continue)', ha='center', va='center', fontsize=10)

    # Arrows
    ax.arrow(2.5, 6.5, 1, 0, head_width=0.05, head_length=0.1, fc='black', ec='black')
    ax.arrow(5.5, 6.5, 1, 0, head_width=0.05, head_length=0.1, fc='black', ec='black')
    ax.arrow(8.5, 6.5, 1, 0, head_width=0.05, head_length=0.1, fc='black', ec='black')
    ax.arrow(10.5, 5.5, 0, -0.5, head_width=0.05, head_length=0.1, fc='black', ec='black')
    ax.arrow(8.5, 4.5, 1, 0, head_width=0.05, head_length=0.1, fc='black', ec='black')
    ax.arrow(10.5, 3.5, 0, -0.5, head_width=0.05, head_length=0.1, fc='black', ec='black')

    plt.title('AIS Pipeline Diagram', fontsize=14)
    plt.savefig('results/ais_pipeline_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_cnn_architecture()
    plot_ais_pipeline()
    print("Diagrams generated and saved to results/")