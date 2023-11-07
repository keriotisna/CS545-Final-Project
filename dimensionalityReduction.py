import numpy as np
from scipy.sparse.linalg import eigs



def normalizeData(data: np.ndarray):

    """
    Normalize data for PCA by subtracting the mean and dividing by the standard deviation
    
    Arguments
    data: An array of data in the shape (DIMENSIONS, SAMPLES)
    
    Return
    centeredData, currentMean, currentStd
    
    centeredData: The normalized centered data
    currentMean: The mean of the data if needed to transform test data
    currentStd: The standard deviation of the data if needed to transform test data
    """

    currentMean = np.mean(data, axis=1, keepdims=True)
    currentStd = np.std(data)

    centeredData = (data - currentMean)/currentStd
    
    return centeredData, currentMean, currentStd




def getPCA(X: np.ndarray, k: int):
    
    """
    Returns PCA using SVD from a dataset X with dimensions as columns and samples as rows. 
    
    X: An array of data with shape (DIMENSIONS, SAMPLES)
    
    k: How many components to produce
    
    Returns W, Z, xMean, xStd
    
    W: The feature matrix computed from PCA slide 18
    Z: The weights matrix computed from W @ X
    xMean: The extracted mean of the data if needed for reconstruction
    xStd: The extracted std of the data if needed for reconstruction
    """
    
    
    normalizedX, xMean, xStd = normalizeData(X)

    covariance = np.cov(normalizedX)
    covariance = 1/2 * (covariance + covariance.T)

    eigenvalues, eigenvectors = eigs(covariance, k=k)

    # Convert to real numbers
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # W is the feature matrix
    W = np.dot(np.linalg.inv(np.diag(np.sqrt(eigenvalues))), eigenvectors.T)
    # W = np.abs(W)

    # Z is the weights matrix
    Z = W @ normalizedX

    # U = eigenvectors
    # reconstruction = np.linalg.pinv(W) @ Z

    return W, Z, xMean, xStd



# def getNMFParameters(X, R):
    
#     # TODO: Rewrite the whole NMF process to not even need this function, instead default arguments should be passed into getNMF()
#     """
#     A helper function that returns the randomly generated weights W, and H based on the number of features you want and the data.
#     Also returns other parameters for the NMF process such as iterations and a non-zero value.
    
#     X: The data with columns of dimensions and rows of samples
    
#     R: The number of parameters you want back
    
#     Returns:
    
#     W: The vertical information about X
    
#     H: The horizontal information about X
        
#     NZ: A small non-zero value that exists to prevent division by 0.
#     """
    
#     M = X.shape[0]
#     N = X.shape[1]

#     # W contains synthesis features and has shape (M x R) where R is the low rank dimensionality
#     W = np.random.rand(M, R)

#     # H contains the activations of the synthesis features and has dimensions (R x N)
#     H = np.random.rand(R, N)

#     # Look at ICA/NMF slides 41-43 for the "algorithm"

#     # Not zero, a small value to prevent divide by 0
#     NZ = 1e-5
    
#     return W, H, NZ



def getNMF(X: np.ndarray, R: int, ITERATIONS=200, optimizationMethod='KL', eps=1e-5):

    """
    Returns the NMF factored matrices W and H from a given dataset X and features R. Note that X doesn't need to be normalized for NMF to work.

    Arguments:
        X: An unscaled data matrix of shape (DIMENSIONS, SAMPLES)
        R: The number of features to be decomposed through NMF
        optimizationMethod: 'KL' or 'EU'. What optimization formula to use


    Returns:
        W: W contains vertical information about the data.
        H: H contains horizontal information about the data.

    """


    def NMFUpdateKLDiv(X, W, H, eps):
        
        """
        Performs a single update of W and H parameters using the KL divergence based update rule
        
        Arguments:
            X: The data matrix of shape (DIMENSIONS, SAMPLES)
            W: The current vertical information matrix of shape (DIMENSIONS, R)
            H: The current horizontal information matrix of shape (R, SAMPLES)
            eps: A non-zero epsilon value to prevent div/0
        
        Returns:
            W: An updated estimate of W
            H: An updated estimate of H
        """
        
        wNumerator = (X / (W @ H + eps)) @ H.T
        wDenominator = np.sum(H, axis=1) + eps
        
        W = np.multiply(W, wNumerator/wDenominator)
        W[W < 0] = 0

        
        
        hNumerator = W.T @ (X / (W @ H + eps))
        hDenominator = np.sum(W, axis=0, keepdims=True).T + eps
        
        H = np.multiply(H, hNumerator/hDenominator)
        H[H < 0] = 0

        return W, H

    def NMFUpdateEuclidean(X, W, H, eps):
        
        """
        Performs a single update of W and H parameters using the matrix-based update rule based on Euclidean distance
        
        Arguments:
            X: The data matrix of shape (DIMENSIONS, SAMPLES)
            W: The current vertical information matrix of shape (DIMENSIONS, R)
            H: The current horizontal information matrix of shape (R, SAMPLES)
            eps: A non-zero epsilon value to prevent div/0
        
        Returns:
            W: An updated estimate of W
            H: An updated estimate of H
        """
        
        # An alternate form of the update rule that still works
        wNumerator = X @ H.T
        wDenominator = (W @ H @ H.T) + eps
        W = np.multiply(W, wNumerator/wDenominator)
        W[W < 0] = 0
        
        hNumerator = W.T @ X
        hDenominator = (W.T @ W @ H) + eps
        H = np.multiply(H, hNumerator/hDenominator)
        H[H < 0] = 0
        
        return W, H



    dims, samples = X.shape
    # W contains synthesis features and has shape (dims x R) where R is the low rank dimensionality
    W = np.random.rand(dims, R)
    # H contains the activations of the synthesis features and has dimensions (R x samples)
    H = np.random.rand(R, samples)
    

    for i in range(ITERATIONS):
        if optimizationMethod == 'KL':
            W, H = NMFUpdateKLDiv(X, W, H, eps)
        elif optimizationMethod == 'EU':
            W, H = NMFUpdateEuclidean(X, W, H, eps)
            
    return W, H