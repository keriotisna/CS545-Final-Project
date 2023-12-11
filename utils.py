import matplotlib.pyplot as plt
import random
import numpy as np
from pydub import AudioSegment
import os
import struct


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
    
    
def findMinSampleRateFast(directory) -> int:
    
    """
    Finds the minimum sample rate of all .wav files in a directory by reading file header information
    
    Returns:
        minSampleRate: The minimum found rate or None if nothing was found
    """
    
    minSampleRate = None

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                try:
                    with open(os.path.join(root, file), 'rb') as wavFile:
                        wavFile.seek(24)  # Jump to the sample rate in the header
                        sampleRate = struct.unpack('<I', wavFile.read(4))[0]
                        if minSampleRate is None or sampleRate < minSampleRate:
                            minSampleRate = sampleRate
                            print(f'Current min rate: {minSampleRate}')

                except Exception as e:
                    print(f"Error reading {file}: {e}")

    return minSampleRate

# TODO: Check if we are supposed to clip like this, maybe ditch the actual clip function until the NMF needs it?
# I think the log really messes things up when converting back and forth from spec to audio. It seems to introduce a lot of artifacts for some reason, maybe the +1e-5 caused it?
def clipSpectrogram(spec):
    # return np.clip(np.log(np.abs(spec)+1e-5), a_min=0, a_max=np.inf).astype(np.float32)
    return np.clip(np.abs(spec), a_min=0, a_max=np.inf).astype(np.float32)



def getLowEnergyIndices(spec:np.ndarray, threshold=1) -> np.ndarray:

    """
    Gets a boolean mask represnting all indices where the total energy was below some threshold
    
    Arguments:
        spec: An input magnitude (or raw) spectrogram
        threshold: The threshold to consider a frame to have low energy
        
    Returns
        removable: A boolean numpy array of the same length as spec which denotes which frames are of low energy 
    """

    def getSpectrogramEnergy(spec:np.ndarray):
        return np.sum(np.square(spec), axis=0)
            
    energy = getSpectrogramEnergy(spec)
    
    removable = (energy < threshold)
    
    return removable

# TODO: Write this to prune the lowest 30% of frames?
def removeLowEnergyFrames(spec:np.ndarray, threshold=0, printPrunedStats=False) -> np.ndarray:
    
    """
    Removes low energy frames from a spectrogram and its corresponding labels if the energy is below some threshold
    
    Arguments:
        spec: The spectrogram
        threshold: How low energy should be for removal
        
    Returns:
        newSpec
    """
    
    removable = getLowEnergyIndices(spec, threshold=threshold)
    
    startingFrames = spec.shape[-1]
    newSpec = np.delete(spec, removable, axis=1)

    if printPrunedStats:
        print(f'Starting spectrogram has {startingFrames} frames')
        print(f'Pruned spectrogram has {newSpec.shape[-1]} frames. Pruned {startingFrames-newSpec.shape[-1]} frames')

    return newSpec

        
def normalizeWAV(arr:np.ndarray) -> np.ndarray:
    
    """
    Normalize the volume of a .wav file as a numpy array
    
    Arguments:
        arr: The audio to normalize as a 1D array
        
    Returns:
        normalized: A normalized version of the audio in the float32 format
    """
    
    maxVal = np.max(np.abs(arr))

    normalized = arr * (1.0 / maxVal)
    return normalized.astype(np.float32)


def convertFloat32toInt16(arr: np.ndarray) -> np.ndarray:
    """
    Convert a float32 normalized array to a 16-bit integer format.

    Arguments:
        arr: The input float32 array.
    
    Returns:
        int16Array: An int16 version of the float32 array
    """
    int16Array = np.clip(arr, -1, 1)  # Ensure values are within [-1, 1]
    int16Array = (int16Array * 32767).astype(np.int16)
    return int16Array

def concatenateSpectrograms(spectrograms:list) -> np.ndarray:
    """Concatenates a list of spectrograms along axis 1"""
    return np.concatenate(spectrograms, axis=1)



def displayConcatenatedSpectrograms(spectrograms:list, title='Concatenated spectrograms'):
    
    """
    Displays all spectrograms concatenated together when given a list of spectrograms
    
    Arguments:
        spectrograms: A list of spectrpgrams
    """
        
    concatenated = concatenateSpectrograms(spectrograms)
    plt.figure(figsize=(30, 3)), plt.pcolormesh(concatenated), plt.title(title), plt.show()