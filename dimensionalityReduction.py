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




def getPCA(X: np.ndarray, dims: int):
    
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

    eigenvalues, eigenvectors = eigs(covariance, k=dims)

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



def getNMF(X: np.ndarray, R: int, iterations:int=200, optimizationMethod='KL', eps:float=1e-5):

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


    def NMFUpdateKLDiv(X:np.ndarray, W:np.ndarray, H:np.ndarray, eps:float):
        
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

    def NMFUpdateEuclidean(X:np.ndarray, W:np.ndarray, H:np.ndarray, eps:float):
        
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
    

    for i in range(iterations):
        if optimizationMethod == 'KL':
            W, H = NMFUpdateKLDiv(X, W, H, eps)
        elif optimizationMethod == 'EU':
            W, H = NMFUpdateEuclidean(X, W, H, eps)
            
    return W, H



def getICA(X:np.ndarray, lr:float=1e-3, iterations:int=400):
    
    """
    Perform Independent Component Analysis on a data matrix X
    
    Arguments:
        X: A data matrix in the shape (DIMENSIONS, SAMPLES)
        lr: The learning rate for the gradient descent algorithm for ICA
        iterations: How many iterations should be performed
        
    Returns:
        W_ICA: The learned weights used to transform new data to the same space
        Z_ICA: The data transformed by the ICA matrix which is just W_ICA @ X
    """

    # Initialize random weights as identity matrix since we don't want to do another linear transform on top of ICA
    W_ICA = np.eye(X.shape[0])

    # LW = np.random.rand(X.shape[0], X.shape[0])

    # differences = []
    for i in range(iterations):
        
        # Adaptive learning rate and a lot of training to ensure convergence
        lr = lr/(i+1)
        
        y = W_ICA @ X

        gradW = (X.shape[0]*np.eye(W_ICA.shape[0]) - 2*np.tanh(y) @ y.T) @ W_ICA
        W_ICA += lr * gradW
        
        # Save the SSD between y and weights @ Z for each iteration to make sure we are converging
        # This is a bit of a hack, we really need to plot the loss of the infomax function and see if that actually converges or decreases. 
        # differences.append((np.sum(y - LW @ X))**2)

    # plt.plot(differences)
    # plt.title('ICA "loss" convergence')
    # plt.show()

    Z_ICA = W_ICA @ X
    
    return W_ICA, Z_ICA