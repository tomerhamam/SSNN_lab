import matplotlib.pyplot as plt
import seaborn as sns

# Configure plotting
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

def visualize_samples(images, labels=None, title="Sample Visualization", n_samples=10, figsize=(12, 2)):
    """
    Visualize a grid of image samples.
    
    TODO: Complete this function
    Args:
        images: Array of shape (n_samples, height, width) or (n_samples, height*width)
        labels: Optional labels for each image
        title: Plot title
        n_samples: Number of samples to show
        figsize: Figure size
    """
    # Ensure we don't try to show more samples than available
    n_show = min(n_samples, len(images))
    
    # TODO: Create figure and subplots
    fig, axes = plt.subplots(1, n_show, figsize=figsize)
    
    # Handle single subplot case
    if n_show == 1:
        axes = [axes]
    
    for i in range(n_show):
        # TODO: Get the image and reshape if needed
        img = images[i]
        if len(img.shape) == 1:  # If flattened
            img = img.reshape(8, 8) # FILL: Reshape to (8, 8) for digits

        # TODO: Display the image
        axes[i].imshow(img, cmap='gray')
        
        # TODO: Set title with label if provided
        if labels is not None:
            axes[i].set_title(f"Label: {labels[i]}")

        axes[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

