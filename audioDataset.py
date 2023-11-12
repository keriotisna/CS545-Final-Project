from dataReader import getDataset
import numpy as np
import os
from scipy.signal import stft # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html
from collections import defaultdict
from itertools import chain

# TODO: How to handle different sample rates between datasets? need to downsample to lowest sample rate. liberosa lets us do this I think?
# TODO: Create full spectrogram function which may be more efficient. createSpectrogramsIndependent makes a spectrogram for each point individually, 
#   but it may be worthwhile to make a single large wav file, then spectrogram that.
# TODO: Pickle the AudioDataset objects to files for easier reading and to save memory if needed
# TODO: Write functions to pickle a dataset and/or combine multiple datasets assuming they're compatible

class AudioDataset():


    # This stores the mappings between general and specific instrument names if we decide certain samples should be included together
    GENERAL_INSTRUMENT_DICTIONARIES = {
            'good-sounds': {
            'bas': ['bass_alejandro_recordings'],
            'cel': ['cello_margarita_attack', 'cello_margarita_dynamics_stability', 'cello_margarita_open_strings', 'cello_margarita_pitch_stability', 'cello_margarita_reference', 'cello_margarita_timbre_richness', 'cello_margarita_timbre_stability', 'cello_nico_improvement_recordings'],
            'cla': ['clarinet_gener_evaluation_recordings', 'clarinet_gener_improvement_recordings', 'clarinet_marti_evaluation_recordings', 'clarinet_pablo_air', 'clarinet_pablo_attack', 'clarinet_pablo_dynamics_stability', 'clarinet_pablo_pitch_stability', 'clarinet_pablo_reference', 'clarinet_pablo_richness', 'clarinet_pablo_timbre_stability', 'clarinet_scale_gener_recordings'],
            'flu': ['flute_almudena_air', 'flute_almudena_attack', 'flute_almudena_dynamics_change', 'flute_almudena_evaluation_recordings', 'flute_almudena_reference', 'flute_almudena_reference_piano', 'flute_almudena_stability', 'flute_almudena_timbre', 'flute_josep_evaluation_recordings', 'flute_josep_improvement_recordings', 'flute_scale_irene_recordings'],
            'obo': ['oboe_marta_recordings'],
            'pic': ['piccolo_irene_recordings'],
            'sax': ['saxo_bariton_raul_recordings', 'saxo_raul_recordings', 'saxo_soprane_raul_recordings', 'saxo_tenor_iphone_raul_recordings', 'saxo_tenor_raul_recordings', 'sax_alto_scale_2_raul_recordings', 'sax_alto_scale_raul_recordings', 'sax_tenor_tenor_scales_2_raul_recordings', 'sax_tenor_tenor_scales_raul_recordings'],
            'tru': ['trumpet_jesus_evaluation_recordings', 'trumpet_jesus_improvement_recordings', 'trumpet_ramon_air', 'trumpet_ramon_attack_stability', 'trumpet_ramon_dynamics_stability', 'trumpet_ramon_evaluation_recordings', 'trumpet_ramon_pitch_stability', 'trumpet_ramon_reference', 'trumpet_ramon_timbre_stability', 'trumpet_scale_jesus_recordings'],
            'vio': ['violin_laia_improvement_recordings', 'violin_laia_improvement_recordings_2', 'violin_raquel_attack', 'violin_raquel_dynamics_stability', 'violin_raquel_pitch_stability', 'violin_raquel_reference', 'violin_raquel_richness', 'violin_raquel_timbre_stability', 'violin_violin_scales_laia_recordings']
        }
    }

    # A list of valid dataset names
    VALID_DATASETS = ['IRMAS', 'good-sounds']
    
    # Static path references for reading datasets
    DATA_PATH = 'data'
    IRMAS_TRAINING_DATA_PATH = os.path.join(DATA_PATH, 'IRMAS-TrainingData')
    GOOD_SOUNDS_TRAINING_DATA_PATH = os.path.join(DATA_PATH, 'good-sounds\\sound_files')
    
    # Valid instruments for each instrument
    VALID_IRMAS_INSTRUMENTS = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
    VALID_GOOD_SOUNDS_INSTRUMENTS = ['bass_alejandro_recordings', 'cello_margarita_attack', 'cello_margarita_dynamics_stability', 'cello_margarita_open_strings', 'cello_margarita_pitch_stability', 'cello_margarita_reference', 'cello_margarita_timbre_richness', 'cello_margarita_timbre_stability', 'cello_nico_improvement_recordings', 'clarinet_gener_evaluation_recordings', 'clarinet_gener_improvement_recordings', 'clarinet_marti_evaluation_recordings', 'clarinet_pablo_air', 'clarinet_pablo_attack', 'clarinet_pablo_dynamics_stability', 'clarinet_pablo_pitch_stability', 'clarinet_pablo_reference', 'clarinet_pablo_richness', 'clarinet_pablo_timbre_stability', 'clarinet_scale_gener_recordings', 'flute_almudena_air', 'flute_almudena_attack', 'flute_almudena_dynamics_change', 'flute_almudena_evaluation_recordings', 'flute_almudena_reference', 'flute_almudena_reference_piano', 'flute_almudena_stability', 'flute_almudena_timbre', 'flute_josep_evaluation_recordings', 'flute_josep_improvement_recordings', 'flute_scale_irene_recordings', 'oboe_marta_recordings', 'piccolo_irene_recordings', 'saxo_bariton_raul_recordings', 'saxo_raul_recordings', 'saxo_soprane_raul_recordings', 'saxo_tenor_iphone_raul_recordings', 'saxo_tenor_raul_recordings', 'sax_alto_scale_2_raul_recordings', 'sax_alto_scale_raul_recordings', 'sax_tenor_tenor_scales_2_raul_recordings', 'sax_tenor_tenor_scales_raul_recordings', 'trumpet_jesus_evaluation_recordings', 'trumpet_jesus_improvement_recordings', 'trumpet_ramon_air', 'trumpet_ramon_attack_stability', 'trumpet_ramon_dynamics_stability', 'trumpet_ramon_evaluation_recordings', 'trumpet_ramon_pitch_stability', 'trumpet_ramon_reference', 'trumpet_ramon_timbre_stability', 'trumpet_scale_jesus_recordings', 'violin_laia_improvement_recordings', 'violin_laia_improvement_recordings_2', 'violin_raquel_attack', 'violin_raquel_dynamics_stability', 'violin_raquel_pitch_stability', 'violin_raquel_reference', 'violin_raquel_richness', 'violin_raquel_timbre_stability', 'violin_violin_scales_laia_recordings']
    VALID_GENERAL_GOOD_SOUNDS_INSTRUMENTS = list(GENERAL_INSTRUMENT_DICTIONARIES['good-sounds'].keys())
    
    def __init__(self, datasetName:str, instruments:list=None, useGeneralInstruments=False):
        
        """
        Arguments:
            datasetName: ['IRMAS', 'good-sounds']
            instruments: A list representing the specific instruments that should be read during initialization. See the class itself for valid instruments
            useGeneralInstruments=False: Whether or not certain instruments should be grouped together and treated as a single key in self.audioData or self.spectrograms. 
                Instrument mappings are defined as class variables, so instruments for different datasets can be grouped together if deemed similar enough.
                
                VALID_IRMAS_INSTRUMENTS = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
                
                VALID_GENERAL_GOOD_SOUNDS_INSTRUMENTS = ['bas', 'cel', 'cla', 'flu', 'obo', 'pic', 'sax', 'tru', 'vio']
        """
        
        assert datasetName in self.VALID_DATASETS
        
        self.datasetName = datasetName        
        
        match self.datasetName:
            case 'IRMAS':
                self.setIRMASInstruments(instruments)
                self.audioData, self.sampleRateDict = getDataset(self.IRMAS_TRAINING_DATA_PATH, self.datasetName, self.instruments, toMonoAudio=True)

            case 'good-sounds':
                self.setGoodSoundsInstruments(instruments, useGeneralInstruments)
                self.audioData, self.sampleRateDict = getDataset(self.GOOD_SOUNDS_TRAINING_DATA_PATH, self.datasetName, self.specificInstruments, toMonoAudio=True)
                if useGeneralInstruments:
                    # TODO: Merge common instruments here
                    self.mergeInstruments()
                    pass



    def mergeInstruments(self):
        
        """
        Merges keys in the audioData and sampleRateDict according to the GENERAL_INSTRUMENT_DICTIONARIES for the current dataset.
        """

        def mergeDicts(mappingDict:dict, subDict:dict) -> dict:
            
            """
            Merges a given subDict into a new dictionary with fewer keys. These keys are merged based on a given mappingDict
            
            Arguments: 
                mappingDict: A dictionary which has a key containing the name of a general key/instrument and a value of a list containing all subkeys that should be merged under the new key.
                subDict: A dictionary that will have its keys merged according to the data specified in the mappingDict
                
            Returns:
                mergedData: A dictionary with the same data as subDict, but with keys merged according to the mappingDict argument
            """
            
            mergedData = defaultdict(list)

            for general_key, subkeys in mappingDict.items():
                # Flatten the list of values for existing subkeys
                flattenedValues = list(chain.from_iterable(subDict.get(k, []) for k in subkeys))

                # Add the flattened list to mergedData only if it is not empty
                if len(flattenedValues) > 0:
                    mergedData[general_key].extend(flattenedValues)

            return mergedData


        if self.datasetName == 'good-sounds':
            
            mappingDict = self.GENERAL_INSTRUMENT_DICTIONARIES['good-sounds']
            audioData = self.getAudioData()
            sampleRateDict = self.sampleRateDict
                        
            mergedAudio = mergeDicts(mappingDict, audioData)
            mergedSampleRates = mergeDicts(mappingDict, sampleRateDict)
        
            self.audioData = mergedAudio
            self.sampleRateDict = mergedSampleRates

        
        pass


    def setIRMASInstruments(self, instruments):

        """
        Sets current instruments during initialization. Should only be called during initialization
        
        Arguments:
            instruments: A list of strings containing instrument names for the given dataset. These should match the keys/instrument names read into the dataset when getDataset() is called
        """

        assert self.datasetName == 'IRMAS'
        
        if instruments is None:
            self.instruments = self.VALID_IRMAS_INSTRUMENTS
        else:
            assert set(instruments).issubset(set(self.VALID_IRMAS_INSTRUMENTS)), f'Invalid instrument specified, valid instruments are {self.VALID_IRMAS_INSTRUMENTS}, given instruments were {instruments}'
            self.instruments = instruments
                    
        
        
        
    def setGoodSoundsInstruments(self, instruments, useGeneralInstruments):
        
        """
        Sets current instruments during initialization. Should only be called during initialization
        
        Arguments:
            instruments: A list of strings containing instrument names for the given dataset. These should match the keys/instrument names read into the dataset when getDataset() is called
            useGeneralInstruments: A boolean flag which determines whether or not the returned data will be combined under general instrument names
        """
        
        assert self.datasetName == 'good-sounds'
        
        if useGeneralInstruments:
        
            if instruments is None:
                self.instruments = self.VALID_GENERAL_GOOD_SOUNDS_INSTRUMENTS
            else:
                assert set(instruments).issubset(set(self.VALID_GENERAL_GOOD_SOUNDS_INSTRUMENTS)), f'Invalid instrument specified, valid instruments are {self.VALID_GENERAL_GOOD_SOUNDS_INSTRUMENTS}, given instruments were {instruments}'
                
                specificInstruments = self.getInstrumentsFromGeneralInstruments(instruments)
                self.instruments = instruments
                self.specificInstruments = specificInstruments
                
        else:
            if instruments is None:
                self.instruments = self.VALID_GOOD_SOUNDS_INSTRUMENTS
            else:
                assert set(instruments).issubset(set(self.VALID_GOOD_SOUNDS_INSTRUMENTS)), f'Invalid instrument specified, valid instruments are {self.VALID_GOOD_SOUNDS_INSTRUMENTS}, given instruments were {instruments}'
                self.instruments = instruments

        
        
    def getInstrumentsFromGeneralInstruments(self, instruments) -> list:
        
        """
        Translates a list of general instruments to specific instruments that can be used to read the correct folders in getDataset()

        Returns:
            specificInstruments: A list containing all the folders that should be read from as defined in dictionaries in this function. 
        """
                
        specificInstruments = []
        if self.datasetName == 'good-sounds':
            mappingDict = self.GENERAL_INSTRUMENT_DICTIONARIES['good-sounds']
            for genInstrument in instruments:
                specificInstruments.extend(mappingDict[genInstrument])
            
        return specificInstruments
        
        
        
        
    def getInstruments(self) -> list:
        """
        Returns a list of instrument names contained in the dictionary
        """
        return self.instruments
    
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
        
    def deleteAudioData(self) -> None:
        """
        Sets self.audioData to None to clear it from memory
        """
        self.audioData = None
        del self.audioData
        
    
    def createSpectrogramsIndependent(self, window='hann', nperseg=1024, noverlap=3/4, deleteAudioData=False):
        """
        Set the class variable spectrogram to a similarly structured dictionary with values as the spectrograms produced by each individual sample
        IMPORTANT: This will delete the audioData attribute to try and save memory as data is read
        
        Arguments:
            window='hann': The window to be used for spectrogram creation
            nperseg=1024: The nperseg argument for the spectrogram. Equivalent to window size
            noverlap=3/4: The fraction of overlap there should be between each window
            deleteAudioData=False: Whether or not audioData should be deleted as spectrograms are created to save memory
        """

        def clipSpectrogram(spec):
            return np.clip(np.log(np.abs(spec)), a_min=0, a_max=np.inf)

        def getSpectrogram(data, sampleRate):
            return clipSpectrogram(stft(data, fs=sampleRate, window=window, nperseg=nperseg, noverlap=int(noverlap*nperseg))[-1])
        
        self.spectrograms = {}

        for instrument, currentInstrumentData in self.audioData.items():
            
            currentSampleRates = self.sampleRateDict[instrument]
            assert len(currentInstrumentData) == len(currentSampleRates)
            spectrograms = []
        
            if deleteAudioData:
                for idx, (data, sampleRate) in enumerate(zip(currentInstrumentData, currentSampleRates)):
                    spectrogram = getSpectrogram(data, sampleRate)
                    self.audioData[instrument][idx] = spectrogram
                    
                self.spectrograms = self.audioData
                self.audioData = None

            else:
                for idx, (data, sampleRate) in enumerate(zip(currentInstrumentData, currentSampleRates)):
                    spectrogram = getSpectrogram(data, sampleRate)
                    spectrograms.append(spectrogram)
                    
                self.spectrograms[instrument] = spectrograms

            # List comprehension isn't much faster here
            # spectrograms = [clipSpectrogram(stft(data, fs=sampleRate, window=window, nperseg=nperseg, noverlap=int(noverlap*nperseg))[-1]) for data, sampleRate in zip(currentInstrumentData, currentSampleRates)]

            

            


    
    



















