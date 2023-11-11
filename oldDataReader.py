import numpy as np
import os
from scipy.io import wavfile

DATA_PATH = 'data'
IRMAS_TRAINING_DATA_PATH = os.path.join(DATA_PATH, 'IRMAS-TrainingData')
SAMPLE_PATH = os.path.join(DATA_PATH, 'vl1.wav')

# cello, clarinet, flute, acoustic guitar, electric guitar, organ, piano, saxophone, trumpet, violin, and human singing voice
IRMAS_TRAINING_DATA_INSTRUMENTS = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
IRMAS_INSTRUMENT_PATHS = {}
for instName in IRMAS_TRAINING_DATA_INSTRUMENTS:        
    IRMAS_INSTRUMENT_PATHS[instName] = os.path.join(IRMAS_TRAINING_DATA_PATH, instName)




def getIRMASTrainingInstruments(instruments:list=None) -> list:
    
    """
    Returns a list of instrument names, returning all if None is passed in.
    
    Arguments:
        instruments:list=None: A list of instrument names or None which will return all valid instrument names
        
    Returns:
        names: A list of valid instrument names
    """
    
    if instruments is None:
        return IRMAS_TRAINING_DATA_INSTRUMENTS
    else:
        return instruments


def getIRMASInstrumentPaths(instruments:list=None) -> dict:
    
    """
    Returns a dictionary of paths that point to respective instruments in the IRMAS training data set
    
    Arguments:
        instruments:list=None: Takes a list which contains strings of individual instrument types desired, or can be left as None to return all paths

    Returns:
        paths: A dictionary with keys as instrument names and values of paths in the IRMAS training set
    """
    
    if instruments is None:
        return IRMAS_INSTRUMENT_PATHS
    else:
        currentPaths = {}
        for instName in instruments:
            currentPaths[instName] = IRMAS_INSTRUMENT_PATHS[instName]
            
        return currentPaths



# def getSampleFilenames(instruments:list=None) -> dict:
    
#     """
#     Returns a dictionary keyed at instruments valued as a list of filenames for that instrument. These are the .wav files, not the 
    
#     Arguments:
#         instruments:list=None: A list of instrument names to retrieve filenames for
    
#     Returns:
#         filenames: A dictionary containing the sample filenames for each instrument
#     """
            
#     filenames = {}
#     filePaths = getIRMASInstrumentPaths(instruments)
    
#     assert len(instruments) == len(filePaths.values())
    
#     for instrumentName, filePath in zip(instruments, list(filePaths.values())):
        
        
#         rawFilenames = os.listdir(filePath)
#         fullFilenames = [os.path.join(filePath, x) for x in rawFilenames]
        
#         # Filter only filenames that are .wav files
#         filenames[instrumentName] = list(filter(lambda x: x.endswith('.wav'), fullFilenames))
        
    
#     return filenames



def readWavFiles(directory: str, toMonoAudio=True) -> dict:
    
    """
    Reads all the .wav files from a given directory and returns a dict with keys 'filenames', 'sampleRates' and 'audioData'

    Returns:
        readData: A dict containing information about the data read from the specified directory
            'filenames' refers to the names of the read files if they happen to contain important information
            'sampleRates' are the sample rates of the data as retrieved from wavfile.read()
            'audioData' contains a list of actual read data
    """
    
    readData = {}
    
    rawFilenames = os.listdir(directory)
    rawFilenames = [os.path.join(directory, x) for x in rawFilenames]
    
    # Filter only filenames that are .wav files
    wavFilenames = list(filter(lambda x: x.endswith('.wav'), rawFilenames))
    wavReadData = [wavfile.read(x) for x in wavFilenames]
    
    sampleRates = [pair[0] for pair in wavReadData]
    
    if toMonoAudio:
        audioData = [np.mean(pair[1], axis=1) for pair in wavReadData]
    else:
        audioData = [pair[1] for pair in wavReadData]
    
    readData['filenames'] = wavFilenames
    readData['sampleRates'] = sampleRates
    readData['audioData'] = audioData
    
    return readData



def getIRMASTrainingInstrumentData(instruments:list=None, toMonoAudio=True) -> dict:
    
    """
    Returns a dictionary with keys of instrument names and values of more dictionaries with keys 'sampleRates' and 'audioData'
        'sampleRates' holds the sample rate of the read file and 'audioData' contains the actual read data from the wav file
    
    Arguments:
        instruments:list=None: A list of strings representing the instruments that should be returned
        toMonoAudio=True: A flag that turns the stereo audio samples to single channel reads
        
    Returns:
        sampleData: A dictionary keyed at instrument names and valued with lists containing sample information
    """
    

    instruments = getIRMASTrainingInstruments(instruments)
    
    directories = getIRMASInstrumentPaths(instruments)
    
    sampleData = {}
    
    for instrumentName, directory in zip(instruments, directories.values()):
        
        readData = readWavFiles(directory)
        
        sampleData[instrumentName] = readData
    

    return sampleData