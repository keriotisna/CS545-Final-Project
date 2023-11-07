import numpy as np


def getNMFParameters(X, R):
    
    """
    A helper function that returns the randomly generated weights W, and H based on the number of features you want and the data.
    Also returns other parameters for the NMF process such as iterations and a non-zero value.
    
    X: The data with columns of dimensions and rows of samples
    
    R: The number of parameters you want back
    
    Returns:
    
    W: The vertical information about X
    
    H: The horizontal information about X
        
    NZ: A small non-zero value that exists to prevent division by 0.
    """
    
    M = X.shape[0]
    N = X.shape[1]

    # W contains synthesis features and has shape (M x R) where R is the low rank dimensionality
    W = np.random.rand(M, R)

    # H contains the activations of the synthesis features and has dimensions (R x N)
    H = np.random.rand(R, N)

    # Look at ICA/NMF slides 41-43 for the "algorithm"

    # Not zero, a small value to prevent divide by 0
    NZ = 1e-5
    
    return W, H, NZ

def NMFUpdateKLDiv(X, W, H, NZ):
    
    """
    Performs a single update of W and H parameters using the KL divergence based update rule
    
    Returns: W, H
    """
    
    wNumerator = (X / (W @ H + NZ)) @ H.T
    wDenominator = np.sum(H, axis=1) + NZ
    
    W = np.multiply(W, wNumerator/wDenominator)
    W[W < 0] = 0

    
    
    hNumerator = W.T @ (X / (W @ H + NZ))
    hDenominator = np.sum(W, axis=0, keepdims=True).T + NZ
    
    H = np.multiply(H, hNumerator/hDenominator)
    H[H < 0] = 0

    return W, H

def getNMF(X: np.ndarray, R: int, ITERATIONS=1000, optimizationMethod='KL'):

    """
    Returns the NMF factored matrices W and H from a given dataset X and features R. Note that X should be unscaled beforehand. 
    
    X: An unscaled data matrix
    
    R: The number of features to be decomposed through NMF
    
    optimizationMethod: 'KL' or 'EU'. What optimization formula to use
    
    Returns: W, H. W contains vertical information about the data. H contains horizontal information about the data.
    """

    W, H, NZ = getNMFParameters(X, R)
    # differences = []
    for i in range(ITERATIONS):
        if optimizationMethod == 'KL':
            W, H = NMFUpdateKLDiv(X, W, H, NZ)

    return W, H