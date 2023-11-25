import numpy as np
import os
from scipy.io import wavfile
from collections import defaultdict

SAMPLE_PATH = 'data\\vl1.wav'

# TODO: Add an instrument argument to only read from certain instrument folders, as the memory requirements for good-sounds is a lot
def getDataset(directory:str, datasetName, instruments, toMonoAudio=True) -> tuple[dict, dict]:

    """
    Reads an entire directory for .wav files and reads them into a usable format. Also includes information about sample rates if downsampling is needed.
    
    Arguments:
        directory: str: The directory of the top level folder which holds the current dataset.
        toMonoAudio=True: Whether or not read wav audio should be converted to mono in the result (Usually, yes)
    

    Returns:
        audioDict, sampleRateDict
        
        audioDict: A dictionary with keys being the same name as the folders the audio data was read from. Each value is a list of numpy arrays which represents the read data.
        sampleRateDict: A dictionary with the same structure as audioDict, but contains sample rates of read files instead.
    """

    def addDataToDictionary(data):
        
        # Identify the instrument directory name (assuming it's the second level directory)
        directoryParts = root.split(os.sep)
        if datasetName == 'good-sounds':
            instrumentName = directoryParts[-2]
            audioDict[instrumentName].append(data)
            sampleRateDict[instrumentName].append(rate)
            return
        elif datasetName == 'IRMAS':
            instrumentName = directoryParts[-1]
            audioDict[instrumentName].append(data)
            sampleRateDict[instrumentName].append(rate)
            return

    audioDict = defaultdict(list)
    sampleRateDict = defaultdict(list)

    # Walk through all directories and files in the top directory
    for root, dirs, files in os.walk(directory):
        directoryParts = root.split(os.sep)
        # Ignore directories not related to valid instruments
        if len(set(directoryParts).intersection(set(instruments))) == 0:
            continue
        for file in files:
            if file.endswith(".wav"):
                # Construct the full file path
                file_path = os.path.join(root, file)

                # Read the .wav file
                rate, data = wavfile.read(file_path)
                
                data = data.astype(np.float32)
                
                if toMonoAudio and data.ndim == 2:
                    data = np.mean(data, axis=1)
                
                addDataToDictionary(data)


    return audioDict, sampleRateDict



