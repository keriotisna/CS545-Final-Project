import numpy as np
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
import random



def getPCA(X: np.ndarray, k: int):
    # TODO: Make this normalize and return normalization data for the transforms because I keep forgetting to do it myself
    """
    Returns PCA using SVD from a dataset X with dimensions as columns and samples as rows. 
    
    X: An array of data with shape (DIMENSIONS, SAMPLES)
    
    k: How many components to produce
    
    Returns W, Z
    
    W: The feature matrix computed from PCA slide 18
    
    Z: The weights matrix computed from W @ X
    """

    covariance = np.cov(X)
    covariance = 1/2 * (covariance + covariance.T)

    eigenvalues, eigenvectors = eigs(covariance, k=k)

    # Convert to real numbers
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # W is the feature matrix
    W = np.dot(np.linalg.inv(np.diag(np.sqrt(eigenvalues))), eigenvectors.T)
    # W = np.abs(W)

    # Z is the weights matrix
    Z = W @ X

    # identityMatrix = W @ covariance @ W.T

    U = eigenvectors

    # reconstruction = np.linalg.pinv(W) @ Z

    return W, Z



def plotSubplots(data, superTitle, subplotTitle, subplotShape, plotType, figsize=(5, 5), rows=3, customSubplotNames=False):
    
    PLOT_COUNT = rows**2
    
    plt.figure(figsize=figsize)
    plt.subplots_adjust(hspace=0.2)
    plt.suptitle(superTitle)
    for i in range(PLOT_COUNT):
        axes = plt.subplot(rows, rows, i + 1)
        
        reshaped = np.reshape(data[:,i], subplotShape, 'F')
        
        if plotType == 'imshow':
            axes.imshow(reshaped)
        if plotType == 'plot':
            axes.plot(range(data.shape[0]), reshaped)
            
        if customSubplotNames:
            axes.set_title(f'{subplotTitle[i]}'), axes.set_xticks([]), axes.set_yticks([])
        else:
            axes.set_title(f'{subplotTitle}[{i}]'), axes.set_xticks([]), axes.set_yticks([])
    plt.show()
    



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

