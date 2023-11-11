import matplotlib.pyplot as plt
import random
import numpy as np



def displayImageGrid(images: list, H: int, W: int=0, shuffle=False, figsize=None):
    """
    Display list of images in a grid (H, W) without boundaries. The images MUST be the same size or this will probably look weird.

    Parameters:
    images: List of numpy arrays representing the images. The images should be the same size
    H: Number of rows.
    W: Number of columns.
    """
    
    numImages = len(images)
    
    # Shuffle images before so we can get a good sampling
    if shuffle:
        random.shuffle(images)
    
    # If no width is defined, we assume a single row of images
    if W == 0:
        W = numImages
    
    if numImages < H * W:
        raise ValueError(f"Number of images ({len(images)}) is smaller than given grid size!")
    
    # Shrink figure size if plotting lots of images
    if figsize is None:
        fig = plt.figure(figsize=(W/5, H/5))
    else:
        fig = plt.figure(figsize=figsize)

    for i in range(H * W):
        img = images[i]
        ax = fig.add_subplot(H, W, i+1)
        ax.imshow(img)

        # Remove axis details
        ax.axis('off')
        
        # Adjust the position of the axis for each image
        ax.set_position([i%W/W, 1-(i//W+1)/H, 1/W, 1/H])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def displayImageGrid2(images: list, H: int, W: int=0, shuffle=False, figsize=None):
    """
    Display list of images in a grid (H, W) with proper spacing for different image sizes.

    Parameters:
    images: List of numpy arrays representing the images. The images can be different sizes.
    H: Number of rows.
    W: Number of columns.
    """
    
    numImages = len(images)
    
    if shuffle:
        random.shuffle(images)
    
    if W == 0:
        W = numImages
    
    if numImages < H * W:
        raise ValueError(f"Number of images ({len(images)}) is smaller than given grid size!")

    # Determine the aspect ratio for each image
    aspect_ratios = [img.shape[1] / img.shape[0] for img in images]

    # Create figure with specified figsize
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        # Determine a reasonable figure size based on the aspect ratios
        avg_ratio = np.mean(aspect_ratios)
        fig_width, fig_height = W * avg_ratio * 4, H * 4
        fig = plt.figure(figsize=(fig_width, fig_height))

    # Creating subplots for each image
    for i in range(min(H * W, len(images))):
        img = images[i]
        ax = fig.add_subplot(H, W, i+1)
        ax.imshow(img)
        ax.axis('off')

        # Adjust subplot aspect ratio based on the image's aspect ratio
        ax.set_aspect(aspect_ratios[i])

    plt.subplots_adjust(wspace=0.1, hspace=0)  # Adjust spacing between images
    plt.show()