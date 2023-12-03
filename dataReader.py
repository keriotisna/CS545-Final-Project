import numpy as np
import os
from scipy.io import wavfile
from collections import defaultdict

SAMPLE_PATH = 'vl1.wav'

# TODO: Add an instrument argument to only read from certain instrument folders, as the memory requirements for good-sounds is a lot
def getDataset(directory:str, datasetName, instruments, toMonoAudio=True, **kwargs) -> tuple[dict, dict, dict]:

    """
    Reads an entire directory for .wav files and reads them into a usable format. Also includes information about sample rates if downsampling is needed.
    Data directories should be structured as directory > instrument_name(s) > wavfiles
    
    Arguments:
        directory: str: The directory of the top level folder which holds the current dataset.
        toMonoAudio=True: Whether or not read wav audio should be converted to mono in the result (Usually, yes)
        kwargs: Key word arguments that can be passed if anything fancy needs to happen here without needing to re-write everything
    

    Returns:
        (audioDict, sampleRateDict, filenamesDict)
        
        audioDict: A dictionary with keys being the same name as the folders the audio data was read from. Each value is a list of numpy arrays which represents the read data.
        sampleRateDict: A dictionary with the same structure as audioDict, but contains sample rates of read files instead.
        filenamesDict: A dictionary of format {instrumentName: [filenames]} which corresponds to the filenames of data in audioDict
    """

    def addDataToDictionary(data, file):
        
        # Identify the instrument directory name (assuming it's the second level directory)
        directoryParts = root.split(os.sep)
        
        # Add more datasets as needed
        match datasetName:
            case 'IRMAS':
                instrumentName = directoryParts[-1]
                audioDict[instrumentName].append(data)
                sampleRateDict[instrumentName].append(rate)
                filenamesDict[instrumentName].append(file)
                return
            case 'good-sounds':
                instrumentName = directoryParts[-2]
                audioDict[instrumentName].append(data)
                sampleRateDict[instrumentName].append(rate)
                filenamesDict[instrumentName].append(file)
                return
            case 'nsynth-valid':
                instrumentName = directoryParts[-1]
                audioDict[instrumentName].append(data)
                sampleRateDict[instrumentName].append(rate)
                filenamesDict[instrumentName].append(file)
                return


    def shouldReadWav(file:str):
        
        """
        Determines whether or not to read the current wavfile based on some condition usually defined in kwargs
        """
        
        # We always only want .wav files, so if it isn't .wav, short circuit
        if not file.endswith('.wav'):
            return False
        
        # Add more conditions per dataset as needed
        # Conditions should be short circuit, so they return False immediately and if all conditions pass, we return True at the end
        match datasetName:
            case 'nsynth-valid':
                if '_acoustic_' not in file and kwargs['kwargs'].get('nsynth_getAcousticOnly', False):
                    return False
                    
        # Return True if we haven't returned false already
        return True


    audioDict = defaultdict(list)
    sampleRateDict = defaultdict(list)
    filenamesDict = defaultdict(list)

    # Walk through all directories and files in the top directory
    for root, dirs, files in os.walk(directory):
        directoryParts = root.split(os.sep)
        # Ignore directories not related to valid instruments
        if len(set(directoryParts).intersection(set(instruments))) == 0:
            continue
        for file in files:
            if shouldReadWav(file):
                # Construct the full file path
                file_path = os.path.join(root, file)

                # Read the .wav file
                rate, data = wavfile.read(file_path)
                data = data.astype(np.float32)
                
                if toMonoAudio and data.ndim >= 2:
                    data = np.mean(data, axis=1)
                
                addDataToDictionary(data, file)


    return audioDict, sampleRateDict, filenamesDict



