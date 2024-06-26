import matplotlib.pyplot as plt
import numpy as np
from config import BATCH_SIZE


def save_image_v2(images, caption, filename):

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    fig.suptitle(caption, fontsize=14, style='italic')

    for idx, image in enumerate(images):

        if BATCH_SIZE > 1:
            image = image[0]
            
        # Plot the first image
        axes[idx].imshow(np.transpose(image.squeeze(), (1, 2, 0)))
        axes[idx].set_title(f"Image {idx + 1}")

    plt.tight_layout()
    plt.savefig(filename)
