from dataReader import getDataset
import numpy as np
import os
from scipy.signal import stft # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html
from collections import defaultdict
from itertools import chain
from dimensionalityReduction import decomposeAudioSKLearn

# TODO: Add support for the nsynth dataset
# TODO: Add support for the UIOWA dataset? It's pretty small though
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
    VALID_DATASETS = ['IRMAS', 'good-sounds', 'nsynth-valid']
    
    # Valid instruments for each instrument, used for identifying the unique instruments in a dataset
    VALID_IRMAS_INSTRUMENTS = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
    VALID_GOOD_SOUNDS_INSTRUMENTS = ['bass_alejandro_recordings', 'cello_margarita_attack', 'cello_margarita_dynamics_stability', 'cello_margarita_open_strings', 'cello_margarita_pitch_stability', 'cello_margarita_reference', 'cello_margarita_timbre_richness', 'cello_margarita_timbre_stability', 'cello_nico_improvement_recordings', 'clarinet_gener_evaluation_recordings', 'clarinet_gener_improvement_recordings', 'clarinet_marti_evaluation_recordings', 'clarinet_pablo_air', 'clarinet_pablo_attack', 'clarinet_pablo_dynamics_stability', 'clarinet_pablo_pitch_stability', 'clarinet_pablo_reference', 'clarinet_pablo_richness', 'clarinet_pablo_timbre_stability', 'clarinet_scale_gener_recordings', 'flute_almudena_air', 'flute_almudena_attack', 'flute_almudena_dynamics_change', 'flute_almudena_evaluation_recordings', 'flute_almudena_reference', 'flute_almudena_reference_piano', 'flute_almudena_stability', 'flute_almudena_timbre', 'flute_josep_evaluation_recordings', 'flute_josep_improvement_recordings', 'flute_scale_irene_recordings', 'oboe_marta_recordings', 'piccolo_irene_recordings', 'saxo_bariton_raul_recordings', 'saxo_raul_recordings', 'saxo_soprane_raul_recordings', 'saxo_tenor_iphone_raul_recordings', 'saxo_tenor_raul_recordings', 'sax_alto_scale_2_raul_recordings', 'sax_alto_scale_raul_recordings', 'sax_tenor_tenor_scales_2_raul_recordings', 'sax_tenor_tenor_scales_raul_recordings', 'trumpet_jesus_evaluation_recordings', 'trumpet_jesus_improvement_recordings', 'trumpet_ramon_air', 'trumpet_ramon_attack_stability', 'trumpet_ramon_dynamics_stability', 'trumpet_ramon_evaluation_recordings', 'trumpet_ramon_pitch_stability', 'trumpet_ramon_reference', 'trumpet_ramon_timbre_stability', 'trumpet_scale_jesus_recordings', 'violin_laia_improvement_recordings', 'violin_laia_improvement_recordings_2', 'violin_raquel_attack', 'violin_raquel_dynamics_stability', 'violin_raquel_pitch_stability', 'violin_raquel_reference', 'violin_raquel_richness', 'violin_raquel_timbre_stability', 'violin_violin_scales_laia_recordings']
    VALID_GENERAL_GOOD_SOUNDS_INSTRUMENTS = list(GENERAL_INSTRUMENT_DICTIONARIES['good-sounds'].keys())
    VALID_NSYNTH_VALID_INSTRUMENTS = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']
    
    def __init__(self, datasetName:str, instruments:list=None, useGeneralInstruments=False,
                 spectrogramKwargs:dict={
                    'window': 'hann',
                    'nperseg': 1024,
                    'noverlap': 768, 
                 },
                 DATA_PATH='data',
                 **kwargs):
        
        """
        Arguments:
            datasetName: ['IRMAS', 'good-sounds']
            instruments: A list representing the specific instruments that should be read during initialization. See the class itself for valid instruments
            useGeneralInstruments=False: Whether or not certain instruments should be grouped together and treated as a single key in self.audioData or self.spectrograms. 
                Instrument mappings are defined as class variables, so instruments for different datasets can be grouped together if deemed similar enough.
                
                VALID_IRMAS_INSTRUMENTS = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
                
                VALID_GENERAL_GOOD_SOUNDS_INSTRUMENTS = ['bas', 'cel', 'cla', 'flu', 'obo', 'pic', 'sax', 'tru', 'vio']
                
            spectrogramKwargs: A kwarg dictionary for the scipy stft function. This will ensure every spectrogram is consistent across datasets and instances
            DATA_PATH: Where to look for the data. Can specify to look in a uniform sample rate directory
            kwargs: Miscellaneous arguments
        """
        
        assert datasetName in self.VALID_DATASETS
        self.datasetName = datasetName
        
        self.spectrogramKwargs = spectrogramKwargs
        self.DATA_PATH = DATA_PATH
        
        # Static path references for reading datasets
        self.IRMAS_TRAINING_DATA_PATH = os.path.join(self.DATA_PATH, 'IRMAS-TrainingData')
        self.GOOD_SOUNDS_TRAINING_DATA_PATH = os.path.join(self.DATA_PATH, 'good-sounds\\sound_files')
        self.NSYNTH_VALID_TRAINING_DATA_PATH = os.path.join(self.DATA_PATH, 'nsynth-valid\\audio')
        
        instruments.sort()
        
        match self.datasetName:
            case 'IRMAS':
                # self._setIRMASInstruments(instruments)
                self._setDatasetInstruments(instruments=instruments)
                self.audioData, self.sampleRateDict = getDataset(self.IRMAS_TRAINING_DATA_PATH, self.datasetName, self.instruments, toMonoAudio=True)

            case 'good-sounds':
                # self._setGoodSoundsInstruments(instruments, useGeneralInstruments)
                self._setDatasetInstruments(instruments=instruments, useGeneralInstruments=useGeneralInstruments)
                self.audioData, self.sampleRateDict = getDataset(self.GOOD_SOUNDS_TRAINING_DATA_PATH, self.datasetName, self.specificInstruments, toMonoAudio=True)
                if useGeneralInstruments:
                    self._mergeInstruments()
            case 'nsynth-valid':
                # self._setNsynthValidInstruments(instruments)
                self._setDatasetInstruments(instruments=instruments)
                self.audioData, self.sampleRateDict = getDataset(self.NSYNTH_VALID_TRAINING_DATA_PATH, self.datasetName, self.instruments, toMonoAudio=True, **kwargs)



    def _mergeInstruments(self):
        
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

        datasetName = self.getDatasetName()

        # Add more datasets as needed
        match datasetName:
            case 'good-sounds':
                mappingDict = self.GENERAL_INSTRUMENT_DICTIONARIES['good-sounds']
                audioData = self.getAudioData()
                sampleRateDict = self.sampleRateDict
                            
                mergedAudio = mergeDicts(mappingDict, audioData)
                mergedSampleRates = mergeDicts(mappingDict, sampleRateDict)
            
                self.audioData = mergedAudio
                self.sampleRateDict = mergedSampleRates



    def _setDatasetInstruments(self, instruments, useGeneralInstruments=False, **kwargs):
        
        """
        Sets current instruments during initialization. Should only be called during initialization
        
        Arguments:
            instruments: A list of strings containing instrument names for the given dataset. These should match the keys/instrument names read into the dataset when getDataset() is called
            useGeneralInstruments: Whether or not to group certain instruments together if they are similar enough, things like tenor sax and alto sax can be just considered saxophone
            kwargs: Keyword arguments that can be specified if anything special needs to happen
        """
        
        datasetName = self.getDatasetName()
        
        # Add more datasets as needed
        match datasetName:
            case 'IRMAS':
                if instruments is None:
                    self.instruments = self.VALID_IRMAS_INSTRUMENTS
                else:
                    assert set(instruments).issubset(set(self.VALID_IRMAS_INSTRUMENTS)), f'Invalid instrument specified, valid instruments are {self.VALID_IRMAS_INSTRUMENTS}, given instruments were {instruments}'
                    self.instruments = instruments
                    
            case 'good-sounds':
                if useGeneralInstruments:
                    if instruments is None:
                        self.instruments = self.VALID_GENERAL_GOOD_SOUNDS_INSTRUMENTS
                    else:
                        assert set(instruments).issubset(set(self.VALID_GENERAL_GOOD_SOUNDS_INSTRUMENTS)), f'Invalid instrument specified, valid instruments are {self.VALID_GENERAL_GOOD_SOUNDS_INSTRUMENTS}, given instruments were {instruments}'
                        
                        specificInstruments = self._getInstrumentsFromGeneralInstruments(instruments)
                        self.instruments = instruments
                        self.specificInstruments = specificInstruments
                else:
                    if instruments is None:
                        self.instruments = self.VALID_GOOD_SOUNDS_INSTRUMENTS
                    else:
                        assert set(instruments).issubset(set(self.VALID_GOOD_SOUNDS_INSTRUMENTS)), f'Invalid instrument specified, valid instruments are {self.VALID_GOOD_SOUNDS_INSTRUMENTS}, given instruments were {instruments}'
                        self.instruments = instruments
        
            case 'nsynth-valid':
                if instruments is None:
                    self.instruments = self.VALID_NSYNTH_VALID_INSTRUMENTS
                else:
                    assert set(instruments).issubset(set(self.VALID_NSYNTH_VALID_INSTRUMENTS)), f'Invalid instrument specified, valid instruments are {self.VALID_NSYNTH_VALID_INSTRUMENTS}, given instruments were {instruments}'
                    self.instruments = instruments
        

    # def _setNsynthValidInstruments(self, instruments):
        
    #     """
    #     Sets current instruments during initialization. Should only be called during initialization
        
    #     Arguments:
    #         instruments: A list of strings containing instrument names for the given dataset. These should match the keys/instrument names read into the dataset when getDataset() is called
    #     """
        
    #     assert self.datasetName == 'nsynth-valid'

    #     if instruments is None:
    #         self.instruments = self.VALID_NSYNTH_VALID_INSTRUMENTS
    #     else:
    #         assert set(instruments).issubset(set(self.VALID_NSYNTH_VALID_INSTRUMENTS)), f'Invalid instrument specified, valid instruments are {self.VALID_NSYNTH_VALID_INSTRUMENTS}, given instruments were {instruments}'
    #         self.instruments = instruments
        

    # def _setIRMASInstruments(self, instruments):

    #     """
    #     Sets current instruments during initialization. Should only be called during initialization
        
    #     Arguments:
    #         instruments: A list of strings containing instrument names for the given dataset. These should match the keys/instrument names read into the dataset when getDataset() is called
    #     """

    #     assert self.datasetName == 'IRMAS'
        
    #     if instruments is None:
    #         self.instruments = self.VALID_IRMAS_INSTRUMENTS
    #     else:
    #         assert set(instruments).issubset(set(self.VALID_IRMAS_INSTRUMENTS)), f'Invalid instrument specified, valid instruments are {self.VALID_IRMAS_INSTRUMENTS}, given instruments were {instruments}'
    #         self.instruments = instruments
                    
        
        
        
    # def _setGoodSoundsInstruments(self, instruments, useGeneralInstruments):
        
    #     """
    #     Sets current instruments during initialization. Should only be called during initialization
        
    #     Arguments:
    #         instruments: A list of strings containing instrument names for the given dataset. These should match the keys/instrument names read into the dataset when getDataset() is called
    #         useGeneralInstruments: A boolean flag which determines whether or not the returned data will be combined under general instrument names
    #     """
        
    #     assert self.datasetName == 'good-sounds'
        
    #     if useGeneralInstruments:
        
    #         if instruments is None:
    #             self.instruments = self.VALID_GENERAL_GOOD_SOUNDS_INSTRUMENTS
    #         else:
    #             assert set(instruments).issubset(set(self.VALID_GENERAL_GOOD_SOUNDS_INSTRUMENTS)), f'Invalid instrument specified, valid instruments are {self.VALID_GENERAL_GOOD_SOUNDS_INSTRUMENTS}, given instruments were {instruments}'
                
    #             specificInstruments = self._getInstrumentsFromGeneralInstruments(instruments)
    #             self.instruments = instruments
    #             self.specificInstruments = specificInstruments
                
    #     else:
    #         if instruments is None:
    #             self.instruments = self.VALID_GOOD_SOUNDS_INSTRUMENTS
    #         else:
    #             assert set(instruments).issubset(set(self.VALID_GOOD_SOUNDS_INSTRUMENTS)), f'Invalid instrument specified, valid instruments are {self.VALID_GOOD_SOUNDS_INSTRUMENTS}, given instruments were {instruments}'
    #             self.instruments = instruments

        
        
    def _getInstrumentsFromGeneralInstruments(self, instruments) -> list:
        
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
        
    def getPhases(self) -> dict:
        if hasattr(self, 'phases'):
            return self.phases
        else:
            raise AttributeError('Phase attribute not initialized, please call a createSpectrograms() function first')
        
    def getSpectrogramKwargs(self):
        return self.spectrogramKwargs
        
    def deleteAudioData(self) -> None:
        """
        Sets self.audioData to None to clear it from memory
        """
        self.audioData = None
        del self.audioData
        
    
    


    def getMagnitudeSpectrogram(self, data:np.ndarray, fs:int) -> np.ndarray:
        
        """
        Gets a magnitude spectrogram of wav file data given a sample rate and kwargs set in self.spectrogramKwargs
        
        Arguments:
            data: A numpy array that represents the raw wavfile read data
            fs: The sample rate of the wav data
            
        Returns:
            magnitude: A magnitude spectrogram of the given data using kwargs stored in self.spectrogramKwargs
        """
        
        def clipSpectrogram(spec):
            return np.clip(np.log(np.abs(spec)), a_min=0, a_max=np.inf).astype(np.float32)
        
        return clipSpectrogram(stft(data, fs=fs, **self.spectrogramKwargs)[-1])
    
    
    def getMagnitudePhaseSpectrogram(self, data:np.ndarray, fs:int) -> tuple[np.ndarray, np.ndarray]:
        
        """
        Gets a magnitude and phase spectrogram of wav file data given a sample rate and kwargs set in self.spectrogramKwargs
        
        Arguments:
            data: A numpy array that represents the raw wavfile read data
            fs: The sample rate of the wav data
            
        Returns:
        (magnitude, phase)
            magnitude: A magnitude spectrogram of the given data using kwargs stored in self.spectrogramKwargs
            phase: A phase spectrogram of the given data using the kwargs stored in self.spectrogramKwargs
        """
        
        def clipSpectrogram(spec):
            return np.clip(np.log(np.abs(spec)), a_min=0, a_max=np.inf).astype(np.float32)
        
        spec = stft(data, fs=fs, **self.spectrogramKwargs)[-1]
        
        # (magnitude, phase)
        return clipSpectrogram(spec), np.angle(spec)
    
        
    def createSpectrogramsIndependent(self, deleteAudioData=False):
        """
        Set the class variable spectrogram to a similarly structured dictionary with values as the spectrograms produced by each individual sample
        
        Sets class variables self.spectrograms and self.phases
        
        IMPORTANT: This will delete the audioData attribute to try and save memory as data is read
        
        Arguments:
            deleteAudioData=False: Whether or not audioData should be deleted as spectrograms are created to save memory
        """

        self.spectrograms = {}
        self.phases = {}

        for instrument, currentInstrumentData in self.audioData.items():
            
            currentSampleRates = self.sampleRateDict[instrument]
            assert len(currentInstrumentData) == len(currentSampleRates)
            spectrograms = []
            phases = []
            # Replace audiodata in place with spectrograms
            if deleteAudioData:
                for idx, (data, sampleRate) in enumerate(zip(currentInstrumentData, currentSampleRates)):
                    mag, phase = self.getMagnitudePhaseSpectrogram(data, sampleRate)
                    self.audioData[instrument][idx] = mag
                    phases.append(phase)
                    

            # Create spectrograms without replacing audioData
            else:
                for idx, (data, sampleRate) in enumerate(zip(currentInstrumentData, currentSampleRates)):
                    mag, phase = self.getMagnitudePhaseSpectrogram(data, sampleRate)
                    spectrograms.append(mag)
                    phases.append(phase)
                    
                self.spectrograms[instrument] = spectrograms
                
            # Phases will be completely new for each instrument, so we always do this regardless of deleting audio data
            self.phases[instrument] = phases

        if deleteAudioData:
            self.spectrograms = self.audioData
            self.deleteAudioData()



    # TODO: Delete old spectrograms?
    # TODO: Do dimensionality reduction on basis functione beforehand? But would this make sense and how would we go back and forth?
    def getBasisFunctions(self) -> tuple[np.ndarray, dict]:
        
        """
        Gets a set of all basis functions from created spectrograms by concatenating them all together. Additionally
        returns a dictionary containing the length of each instrument's spectrograms so they can be identified later

        Returns:
            (basisFunctions, indexDict)
            basisFunctions: A numpy array of shape (DIMS, SAMPLES) which holds the concatenated spectrograms for all instruments in self.spectograms
            indexDict: A dictionary with keys of instrument names and values of ints which represents how many spectrogram frames are contained by each instrument
        """
        
        indexDict = {}
        allData = None
        for instrumentName, data in self.spectrograms.items():
            
            concatenatedData = np.concatenate(data, axis=1)
            
            if allData is None:
                allData = concatenatedData
            else:
                allData = np.concatenate((allData, concatenatedData), axis=1)
            
            # TODO: Remove columns with low energy to reduce size?
            
            # Store how long the current data array was so we can navigate it
            indexDict[instrumentName] = concatenatedData.shape[-1]
                    
        return allData, indexDict
    
    
    def runNMFAudioDecomposition(self, X:np.ndarray):
        
        # Combine all spectrogram data into a large array that can be used for NMF as a basis function array. 
        basisFunctions, indexDict = self.getBasisFunctions()
        
        W_NMF, H_NMF = decomposeAudioSKLearn(X=X, W=basisFunctions, H=None)
        
        
        
        pass
            


    
    



















