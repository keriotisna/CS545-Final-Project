from dataReader import getDataset
import numpy as np
import os
from scipy.signal import stft # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html

# TODO: Allow for args to read only certain instruments
# TODO: How to handle different sample rates between datasets? need to downsample to lowest sample rate. liberosa lets us do this I think?
# TODO: Create full spectrogram function which may be more efficient. createSpectrogramsIndependent makes a spectrogram for each point individually, 
#   but it may be worthwhile to make a single large wav file, then spectrogram that.

class AudioDataset():
    

    VALID_DATASETS = ['IRMAS', 'good-sounds']
    DATA_PATH = 'data'
    IRMAS_TRAINING_DATA_PATH = os.path.join(DATA_PATH, 'IRMAS-TrainingData')
    
    GOOD_SOUNDS_TRAINING_DATA_PATH = os.path.join(DATA_PATH, 'good-sounds\\sound_files')
    
    
    def __init__(self, datasetName:str):
        
        """
        Arguments:
            datasetName: ['IRMAS', 'good-sounds']
        
        """
        
        assert datasetName in self.VALID_DATASETS
        
        self.datasetName = datasetName
        
        match datasetName:
            case 'IRMAS':
                self.audioData, self.sampleRateDict = getDataset(self.IRMAS_TRAINING_DATA_PATH, toMonoAudio=True, datasetName=datasetName)
            case 'good-sounds':
                self.audioData, self.sampleRateDict = getDataset(self.GOOD_SOUNDS_TRAINING_DATA_PATH, toMonoAudio=True, datasetName=datasetName)

    
    
    def getAudioData(self) -> dict:
        return self.audioData
    
    def getSampleRateDict(self) -> dict:
        return self.sampleRateDict
    
    def getDatasetName(self) -> str:
        return self.datasetName
    
    def getSpectrograms(self) -> dict:
        if hasattr(self, 'spectrograms'):
            return self.spectrograms
        else:
            raise AttributeError('Spectrogram attribute not initialized, please call a createSpectrograms() function first')
        
    
    def createSpectrogramsIndependent(self, window='hann', nperseg=1024, noverlap=3/4):
        """
        Set the class variable spectrogram to a similarly structured dictionary with values as the spectrograms produced by each individual sample
        """

        def clipSpectrogram(spec):
            return np.clip(np.log(np.abs(spec)), a_min=0, a_max=np.inf)

        self.spectrograms = {}

        for instrument in self.audioData.keys():
            currentInstrumentData = self.audioData[instrument]
            currentSampleRates = self.sampleRateDict[instrument]
            
            spectrograms = []
            
            for data, sampleRate in zip(currentInstrumentData, currentSampleRates):
                spectrograms.append(clipSpectrogram(stft(data, fs=sampleRate, window=window, nperseg=nperseg, noverlap=int(noverlap*nperseg))[-1]))
                
            # List comprehension isn't much faster here
            # spectrograms = [clipSpectrogram(stft(data, fs=sampleRate, window=window, nperseg=nperseg, noverlap=int(noverlap*nperseg))[-1]) for data, sampleRate in zip(currentInstrumentData, currentSampleRates)]

            self.spectrograms[instrument] = spectrograms
            
        self.audioData = None
        self.sampleRateDict = None
            


    
    



















