import numpy as np
from scipy.sparse.linalg import eigs
from scipy import sparse
from numba import njit
from sklearn.decomposition import non_negative_factorization


def getDifference(prev, curr):
    return np.sum((prev - curr)**2)

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



def getNMF(X: np.ndarray, R: int, iterations:int=4000, optimizationMethod='KL', eps:float=1e-5):

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


    def NMFUpdateRegularized(X:np.ndarray, W:np.ndarray, H:np.ndarray, eps:float, regularization:float, lr:float):
        # Calculate the gradients
        deltaW = (W @ H - X) @ H.T
        deltaH = W.T @ (W @ H - X) + regularization * H

        # Update W and H with the learning rate
        W += lr * deltaW
        H += lr * deltaH

        # Enforce non-negativity
        W[W < 0] = 0
        H[H < 0] = 0

        return W, H
        

    dims, samples = X.shape
    # W contains synthesis features and has shape (dims x R) where R is the low rank dimensionality
    W = np.random.rand(dims, R).astype(np.float32)
    # H contains the activations of the synthesis features and has dimensions (R x samples)
    H = np.random.rand(R, samples).astype(np.float32)
    eps = np.float32(eps)
    
    prevW = W.copy()
    prevH = H.copy()
    
    for i in range(iterations):
        if optimizationMethod == 'KL':
            W, H = NMFUpdateKLDiv(X, W, H, eps)
        elif optimizationMethod == 'EU':
            W, H = NMFUpdateEuclidean(X, W, H, eps)
        elif optimizationMethod == 'RG':
            W, H = NMFUpdateRegularized(X, W, H, eps, regularization=0.00, lr=1e-7/(i+1))
            
        if getDifference(prevW, W) < 1e-5 and getDifference(prevH, H) < 1e-5:
            print(f'Break on iteration {i}')
            break
            
        prevW = W.copy()
        prevH = H.copy()
        
    return W, H





@njit(parallel=True)
def _NMFUpdateKLDiv(X, W, H, eps):
    # Calculate the numerator and the denominator for the H update
    # TODO: Maybe downweight updates to W so we can adapt better?
    hNumerator = W.T @ (X / (W @ H + eps))
    hDenominator = np.sum(W, axis=0) + eps  # Sum over axis 0 to get the correct shape
    
    # Explicitly broadcast hDenominator to match the shape of hNumerator
    hDenominator_reshaped = hDenominator.reshape(-1, 1)  # Reshape to (n_components, 1)

    # Perform the element-wise division
    H *= hNumerator / hDenominator_reshaped
    
    # Ensure non-negativity
    H = np.maximum(H, 0)
    
    return H


@njit
def decomposeAudio(X, soundArrayW, iterations=1000, eps=1e-5):

    
    X = X.astype(np.float32)
    W = soundArrayW.copy().astype(np.float32)
    n = W.shape[1]
    samples = X.shape[1]
    
    eps = np.float32(eps)
    
    H = np.random.rand(n, samples).astype(np.float32)
    prevH = H.copy()
    
    for i in range(iterations):
        H = _NMFUpdateKLDiv(X, W, H, eps)
        if np.linalg.norm(H - prevH) < eps:
            print(f'Convergence reached at iteration {i}.')
            break
        
        prevH = H.copy()
    
    return W, H



def _NMFUpdateKLDivSlow(X, W, H, eps):

    # We assume W is fixed since they represent our basis functions so we only update H
    # TODO: Maybe downweight updates to W so we can adapt better?
    
    hNumerator = W.T @ (X / (W @ H + eps))
    # Need to use expand_dims for numba
    hDenominator = np.expand_dims(np.sum(W, axis=0), 0).T + eps
    H *= hNumerator / hDenominator
    H[H < 0] = 0
    return H

def decomposeAudioSlow(X, soundArrayW, iterations=1000, eps=1e-5):

    
    X = X.astype(np.float32)
    W = soundArrayW.copy().astype(np.float32)
    n = W.shape[1]
    samples = X.shape[1]
    
    eps = np.float32(eps)
    
    H = np.random.rand(n, samples).astype(np.float32)
    prevH = H.copy()
    
    for i in range(iterations):
        H = _NMFUpdateKLDivSlow(X, W, H, eps)
        if np.linalg.norm(H - prevH) < eps:
            print(f'Convergence reached at iteration {i}.')
            break
        
        prevH = H.copy()
    
    return W, H

# TODO: Try with sparse matrices? Most of the basis functions are zero after all
def decomposeAudioSKLearn(X:np.ndarray, W:np.ndarray, H:np.ndarray=None, regularization=0.1) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Decompose audio from a given spectrogram X, a basis function matrix W, and an optional prior matrix H.
    This is pretty much identical to custom implementations, just much faster
    
    Arguments:
        X: A (DIMS, SAMPLES) numpy matrix which holds spectrogram data
        W: A (DIMS, R) matrix where R is the number of instrument samples
        H: A (R, SAMPLES) matrix which is an optional prior to start decomposition from
            This will maybe be time activations if we find a way to get them. 

    Returns:
        W_NMF, H_NMF
        The decomposed NMF features found from sklearn. These will be (basisFunctions, activations)
    """
    
    # sklearn only lets us freeze H, so we need to swap all the matrices to align with what it wants
    # Transpose X to fit sklearn's expected dimensionality (samples, features)
    XTransposed = X.T + 1e-5

    # Transpose H to fit sklearn's expected dimensionality (components, features)
    HTransposed = W.T

    # Initialize W randomly with the expected shape (samples, components)
    if H is None:
        WTransposed = np.random.rand(XTransposed.shape[0], HTransposed.shape[0])
    else:
        WTransposed = H.copy()

    # Perform NMF
    W_NMF, H_NMF, niter = non_negative_factorization(
        X=XTransposed.astype(np.float32),
        W=WTransposed.astype(np.float32),
        H=HTransposed.astype(np.float32),
        n_components=HTransposed.shape[0], # Need to define n_components or it will break, probably a bug
        update_H=False,  # Keep our actual weights W fixed
        init='custom',  # Use custom initialization
        max_iter=2000,
        solver='mu',
        beta_loss='frobenius',
        l1_ratio=0.5,
        alpha_W=regularization,
        tol=1e-6
    )
    
    print(f'NMF terminated after {niter} iterations')
    
    # Again, these are swapped and transposed since sklearn won't let us freeze W
    return H_NMF.T, W_NMF.T


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

    prevW_ICA = W_ICA.copy()

    # differences = []
    for i in range(iterations):
        
        # Adaptive learning rate and a lot of training to ensure convergence
        lr = lr/(i+1)
        
        y = W_ICA @ X

        gradW = (X.shape[0]*np.eye(W_ICA.shape[0]) - 2*np.tanh(y) @ y.T) @ W_ICA
        W_ICA += lr * gradW
        
        if getDifference(W_ICA, prevW_ICA) < 1e-10:
            print(f'Break on iteration {i}')
            break
        
        prevW_ICA = W_ICA.copy()

    # plt.plot(differences)
    # plt.title('ICA "loss" convergence')
    # plt.show()

    Z_ICA = W_ICA @ X
    
    return W_ICA, Z_ICA