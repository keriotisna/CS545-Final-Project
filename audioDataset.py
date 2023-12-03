from dataReader import getDataset
import numpy as np
import os
from scipy.signal import stft, istft, check_NOLA # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html
from collections import defaultdict
from itertools import chain
from dimensionalityReduction import decomposeAudioSKLearn
from scipy.io import wavfile
import re
from utils import *

# NOTE: Refactor to pull out functions not directly tied to the dataset itself and place them in utils.py


# TODO: Write function to trim out intemediate values in samples? Like delete a few frames from the middle? Maybe not?
# TODO: Pickle the AudioDataset objects to files for easier reading and to save memory if needed
# TODO: Write functions to pickle a dataset and/or combine multiple datasets assuming they're compatible

"""
Dataset Notes:

NSYNTH DATASET:
Each filename is formatted as follows:
    <instrumentName>_<instrumentSource>_<instrumentID>-<pitch>-<velocity>
"""

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
    
    
    # Add more regex expressions as needed as bad files are found
    # A list of bad Nsynth files that are just white noise for some reason. These can be removed in memory without touching the real dataset
    # brass_acoustic_046-084-***.wav to brass_acoustic_046-108-***.wav
    badFilenameRegex = re.compile(r'brass_acoustic_046-(08[4-9]|09[0-9]|10[0-8])-\d{3}\.wav$')
    BAD_NSYNTH_FILENAMES = [badFilenameRegex]
    
    def __init__(self, datasetName:str, instruments:list=None, useGeneralInstruments=False,
                 spectrogramKwargs:dict={
                    'window': 'hann',
                    'nperseg': 1024,
                    'noverlap': 256, 
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
        
        self.setSpectrogramKwargs(spectrogramKwargs)
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
                self.audioData, self.sampleRateDict, self.filenamesDict = getDataset(self.IRMAS_TRAINING_DATA_PATH, self.datasetName, self.instruments, toMonoAudio=True)

            case 'good-sounds':
                # self._setGoodSoundsInstruments(instruments, useGeneralInstruments)
                self._setDatasetInstruments(instruments=instruments, useGeneralInstruments=useGeneralInstruments)
                self.audioData, self.sampleRateDict, self.filenamesDict = getDataset(self.GOOD_SOUNDS_TRAINING_DATA_PATH, self.datasetName, self.specificInstruments, toMonoAudio=True)
                if useGeneralInstruments:
                    self._mergeInstruments()
            case 'nsynth-valid':
                # self._setNsynthValidInstruments(instruments)
                self._setDatasetInstruments(instruments=instruments)
                self.audioData, self.sampleRateDict, self.filenamesDict = getDataset(self.NSYNTH_VALID_TRAINING_DATA_PATH, self.datasetName, self.instruments, toMonoAudio=True, **kwargs)



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
        
        """
        Returns a dictionary containing raw audio data read from a dataset
        
        Returns:
            audioData: A dictionary of format {'instrumentName': [dataArray]}
        """
        return self.audioData
    
    def getSampleRateDict(self) -> dict:
        return self.sampleRateDict
    
    def getMinSampleRate(self) -> int:
        
        """
        Returns the minimum sample rate of all files read into the dataset. 
        """
        
        sampleRates = self.getSampleRateDict()
        return min([item for sublist in sampleRates.values() for item in sublist])
    
    def getDatasetName(self) -> str:
        return self.datasetName
    
    def getSpectrograms(self) -> dict:
        
        """
        Gets a spectrogram dictionary containing the magnitude spectrograms for read instruments in the dataset
        
        Returns:
            spectrograms: A dictionary of format {'instrumentName': [spectrograms]}
        """
        
        if hasattr(self, 'spectrograms'):
            return self.spectrograms
        else:
            raise AttributeError('Spectrogram attribute not initialized, please call a createSpectrograms() function first')
        
    def getPhases(self) -> dict:
        
        """
        Gets a phase dictionary containing the phases for read instruments in the dataset
        
        Returns:
            phases: A dictionary of format {'instrumentName': [phaseArrays]}
        """
        
        if hasattr(self, 'phases'):
            return self.phases
        else:
            raise AttributeError('Phase attribute not initialized, please call a createSpectrograms() function first')
        
    def getSpectrogramKwargs(self) -> dict:
        
        """
        Gets the spectrogram kwargs used in the scipy stft and istft
        """
        
        return self.spectrogramKwargs
        
    def setSpectrogramKwargs(self, kwargs:dict):
        
        """
        Sets the spectrogram kwargs used in the scipy stft and istft
        """
        assert check_NOLA(**kwargs), f"ERROR: Current spectrogram kwargs {kwargs} will not satisfy NOLA condition making inversion impossible!\nSee https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.check_NOLA.html for more info"
        self.spectrogramKwargs = kwargs
        
    def deleteAudioData(self):
        
        """
        Sets self.audioData to None to clear it from memory
        """
        
        self.audioData = None
        del self.audioData

    
    def deleteRedundantAudioData(self, pitchInterval=2, writeDebugFiles=False):
        
        """
        Deletes a fraction of data samples from each instrument under the assumption most samples are very similar and can be pruned.
        This works best when each sample is a single note of an instrument so we can prune notes that are close together.
        
        Arguments:
            pitchInterval: Get every Nth pitch
            writeDebugFiles: Whether or not to write all the remaining .wav files to a debug folder to examine manually
        """
        
        def getFilenameData(fn:str):
            return fn.rsplit('_', 1)[-1].split('.')[0]
        
        def getInstId(fn:str):
            return getFilenameDataAttr(fn, 0)
        
        def getInstPitch(fn:str):
            return getFilenameDataAttr(fn, 1)
        
        def getInstVel(fn:str):
            return getFilenameDataAttr(fn, 2)
        
        def getFilenameDataAttr(fn, idx):
            return getFilenameData(fn).split('-')[idx]
        
        def filterPitches(indices, pitches):
            """
            Filter out pitches that are very close to each other.
            Keeping every third pitch as an example.
            """
            filteredIndices = []
            previousPitch = None
            for i, pitch in enumerate(pitches):
                if previousPitch is None or abs(int(pitch) - int(previousPitch)) >= pitchInterval:
                    filteredIndices.append(indices[i])
                    previousPitch = pitch
            return filteredIndices
        
        def isValidFilename(filename):
            """
            Check if the filename does not match any bad filename patterns.
            """
            return not any(pattern.match(filename) for pattern in self.BAD_NSYNTH_FILENAMES)

        audioData = self.audioData
        filenamesDict = self.filenamesDict
        srDict = self.sampleRateDict
        
        for instrumentName in self.getInstruments():
            
            dataList = audioData[instrumentName]
            filenameList = filenamesDict[instrumentName]
            srList = srDict[instrumentName]
            assert len(dataList) == len(filenameList), f'Length of dataList {len(dataList)} does not equal length of filenameList {len(filenameList)}! self.audioData or self.filenamesDict was altered inconsistently!'
            
            # Filter out bad filenames
            validIndices = [i for i in range(len(filenameList)) if isValidFilename(filenameList[i])]
            
            # First, filter based on velocity
            velocityFilteredIndices = [i for i in validIndices if getInstVel(filenameList[i]) == '100']

            # Group indices by instrument ID for velocity-filtered data
            idToIndices = {}
            for i in velocityFilteredIndices:
                instID = getInstId(filenameList[i])
                if instID not in idToIndices:
                    idToIndices[instID] = []
                idToIndices[instID].append(i)

            # Filter pitches for each instrument ID to the specified pitchInterval
            allFilteredIndices = set()
            for indices in idToIndices.values():
                pitches = [getInstPitch(filenameList[i]) for i in indices]
                filteredIndices = filterPitches(indices, pitches)
                allFilteredIndices.update(filteredIndices)

            # Remove unfiltered data from current built lists
            for i in range(len(filenameList) - 1, -1, -1):
                if i not in allFilteredIndices:
                    del dataList[i]
                    del filenameList[i]
                    del srList[i]

            # Update all dictionaries in place after removing items
            audioData[instrumentName] = dataList
            filenamesDict[instrumentName] = filenameList
            srDict[instrumentName] = srList            
            
            if writeDebugFiles:
                for data, fn in zip(dataList, filenameList):
                    x = normalizeWAV(data)
                    self._writeDebugSample(x, fn=fn)
            

    def _writeDebugSample(self, data:np.ndarray, fn:str=''):
        """
        Writes raw audio data from a .wav file to a debug directory for manual analysis.

        Args:
            data: The raw read audio data in a 1D numpy array. Normalization is highly recommended to avoid blowing out your eardrums
            fn: An optional suffix for the saved file 
        """
        wavfile.write(f'debug-samples\\debugSample{fn}.wav', rate=self.getMinSampleRate(), data=data)

    def normalizeAudioData_(self):
        
        """
        Normalize the raw audio data stored in self.audioData in-place
        THIS SEEMS TO BREAK STUFF SO DON'T DO IT UNLESS YOU HAVE A GOOD REASON
        """
        
        audioData = self.audioData
        
        for instrumentName in self.getInstruments():
            instList = audioData[instrumentName]
            for idx in range(len(instList)):
                instList[idx] = normalizeWAV(instList[idx])

    # TODO: This produces TERRIBLE reconstructions for some reason. Lots of artifacting compared to original samples, even at the 16kHz SR
    def writeBasisFunctionAudioFiles(self):
        
        """
        Writes all the basis functions separated by instrument to audio files for your listening pleasure (or displeasure if something went wrong)
        """
        
        spectrograms = self.getSpectrograms()
        phases = self.getPhases()
        
        for instrumentName in spectrograms.keys():
            
            spec = spectrograms[instrumentName]
            phase = phases[instrumentName]
            
            concatenatedSpec = np.concatenate(spec, axis=1)
            concatenatedPhase = np.concatenate(phase, axis=1)

            x = self.reconstructWAVData(concatenatedSpec, concatenatedPhase)
            x = normalizeWAV(x)
            
            x = convertFloat32toInt16(x)
            
            self._writeDebugSample(x, f'_{instrumentName}')
            
            
    def reconstructWAVData(self, mag, phase) -> np.ndarray:
        
        """
        Converts from a magnitude and phase back to a .wav format using the istft.
        
        Don't forget to normalize and convert from float32 to int16 using convertFloat32toInt16()
        
        Arguments:
            mag: A magnitude spectrogram
            phase: Phase information captured from the initial spectrogram generation
            
        Returns:
            x: A 1D numpy array which represents the raw reconstruction without post-processing like normalization or type conversion.
        """
        
        complexSpec = np.exp(mag) * np.exp(1j * phase)
        t, x = istft(complexSpec, fs=self.getMinSampleRate(), **self.getSpectrogramKwargs())
        
        return x


    def _cycleSTFT(self, data, fs=16000) -> np.ndarray:
        
        """
        Cycles given audio data using an STFT and then an ISTFT to ensure the process is reversible
        
        Arguments:
            data: The raw audio data read from a .wav file
            fs: The sample rate
            
        Returns:
            reconstruction: A reconstructed audio data
        """
                
        if data.ndim > 1:
            data = np.mean(data, axis=1).astype(np.float32)
        
        print(f'NOLA is: {check_NOLA(**self.getSpectrogramKwargs())}')
        
        spec = stft(data, fs=fs, **self.getSpectrogramKwargs())[-1]
        
        # (magnitude, phase)
        phase = np.angle(spec)
        spec = np.clip(np.log(np.abs(spec)+1e-5), a_min=0, a_max=np.inf).astype(np.float32)

        _, reconstruction = istft(np.exp(spec)*np.exp(1j * phase), fs=fs, **self.getSpectrogramKwargs())
        
        return reconstruction
    
    def displayInstrumentSpectrograms(self):
        
        """
        Displays all spectrograms concatenated together based on instrument
        """
            
        instruments = self.getInstruments()
        spectrograms = self.getSpectrograms()
        
        for instName in instruments:
            
            specs = spectrograms[instName]
            
            concatenated = concatenateSpectrograms(specs)
            plt.figure(figsize=(30, 3)), plt.pcolormesh(concatenated), plt.title(instName), plt.show()

    def getMagnitudeSpectrogram(self, data:np.ndarray, fs:int) -> np.ndarray:
        
        """
        Gets a magnitude spectrogram of wav file data given a sample rate and kwargs set in self.spectrogramKwargs
        
        Arguments:
            data: A numpy array that represents the raw wavfile read data
            fs: The sample rate of the wav data
            
        Returns:
            magnitude: A magnitude spectrogram of the given data using kwargs stored in self.spectrogramKwargs
        """
        
        if data.ndim > 1:
            data = np.mean(data, axis=1).astype(np.float32)
        
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
        
        if data.ndim > 1:
            data = np.mean(data, axis=1).astype(np.float32)
        
        spec = stft(data, fs=fs, **self.spectrogramKwargs)[-1]
        phase = np.angle(spec)
        # (magnitude, phase)
        return clipSpectrogram(spec), phase
    
    def getAudioFromMagnitudePhaseSpectrogram(self, mag, phase, fs) -> tuple[np.ndarray, np.ndarray]:
        
        """
        Gets an inverse stft of a magnitude and phase spectrogram
        
        Arguments:
            mag: The magnitude spectrogram
            phase: The phase spectrogram
            fs: The sample rate
            
        Returns:
            (outputTimes, istft)
            outputTimes: An array representing the output times for the spectrogram, usually tossed since we don't need it
            istft: The actual result of the inverse stft process
        """
        
        complexSpec = mag * np.exp(1j * phase)
        
        return istft(complexSpec, fs=fs, **self.getSpectrogramKwargs())

    
        
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


    def removeLowEnergyFrames_(self, threshold=1):
        
        """
        Remove all spectrogram frames below some threshold for all instruments in the dataset.
        
        Arguments:
            threshold: The threshold below which spectrogram frames will be removed for low energy.
        """
        
        instruments = self.getInstruments()
        spectrograms = self.getSpectrograms()
        phasesDict = self.getPhases()
        
        for instName in instruments:
        
            specs = spectrograms[instName]
            phases = phasesDict[instName]
            
        
            startingSize = np.sum([spec.shape[-1] for spec in specs])
        
            removableIndices = [getLowEnergyIndices(spec=spec, threshold=threshold) for spec in specs]
            
            prunedSpecs = [np.delete(spec, removable, axis=1) for spec, removable in zip(specs, removableIndices)]
            prunedPhases = [np.delete(phase, removable, axis=1) for phase, removable in zip(phases, removableIndices)]
            
            endingSize = np.sum([spec.shape[-1] for spec in prunedSpecs])
            
            prunedCount = startingSize - endingSize
            print(f'{instName} has {startingSize} frames')
            print(f'Pruned {prunedCount} frames from {instName}')
            
            # Replace old spectrograms in place
            spectrograms[instName] = prunedSpecs
            phasesDict[instName] = prunedPhases


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
                        
            # Store how long the current data array was so we can navigate it
            indexDict[instrumentName] = concatenatedData.shape[-1]
                    
        return allData, indexDict
    
    
    # TODO: Try doing several smaller decompositions on segments of the testAudio
    def runNMFAudioDecomposition(self, testAudio:np.ndarray, showPlots=True, nmfRegularization=0.001):
    
        """
        Runs the full NMF audio decomposition process on given test audio using the stored spectrograms as basis functions
        
        Arguments:
            testAudio: A 1D array which represents the raw .wav file to be separated
            showPlots: Whether or not to show intermediate plots of reuslts like individual decomposition activations
            nmfRegularization: The level of regularization to use in the NMF decomposition process
        """

        
        def _reconstructDecompositions(W_NMF:np.ndarray, H_NMF:np.ndarray, phase:np.ndarray, indexDict:dict) -> dict:
            
            """
            Reconstructs instruments from basis functions and activations and writes decompositions to the decompositions folder

            Arguments:
                W_NMF: A numpy array of shape (DIMS, SAMPLES) which represents the basis functions from NMF
                H_NMF: A numpy array of shape (SAMPLES, SPECTROGRAM_LENGTH) which represents the decomposed activations of each instrument from NMF
                indexDict: A dictionary of format ('instrumentName': instrumentLength) which notes how long each basis function is

            Returns:
                isolations: A dictionary of format ('instrumentName': isolation) where isolation is a complex spectrogram which can be converted back to a wav file
            """
            
            isolations = []
            
            currentBase = 0
            
            for instrumentName, indexOffset in indexDict.items():
                currentBasisFunctions = W_NMF[:, currentBase:currentBase+indexOffset]
                currentActivations = H_NMF[currentBase:currentBase+indexOffset, :]
                
                currentReconstruction = np.exp(currentBasisFunctions @ currentActivations) * np.exp(1j * phase)
                isolations.append(currentReconstruction)
                currentBase += indexOffset
                
                if showPlots:
                    plt.pcolormesh(np.abs(currentReconstruction)), plt.title('Decomposed Recon'), plt.show()
            
            return isolations
        
        if testAudio.ndim > 1:
            testAudio = np.mean(testAudio, axis=1).astype(np.float32)

        fs = self.getMinSampleRate()

        # Combine all spectrogram data into a large array that can be used for NMF as a basis function array. 
        basisFunctions, indexDict = self.getBasisFunctions() # basisFunctions are float32

        # Extract magnitude and phase spectrograms of the test audio for channel separation
        magnitude, phase = self.getMagnitudePhaseSpectrogram(data=testAudio, fs=fs)
        
        if showPlots:
            plt.figure(figsize=(30, 3)), plt.pcolormesh(magnitude), plt.title('Original data'), plt.show()
            plt.figure(figsize=(30, 3)), plt.pcolormesh(basisFunctions), plt.title('W_NMF'), plt.show()

        # TODO: If we get time priors for H, initialize them here
        # NOTE: REGULARIZATION HYPERPARAMETER IS VERY IMPORTANT. RESULTS ARE SENSITIVE
        W_NMF, H_NMF = decomposeAudioSKLearn(X=magnitude, W=basisFunctions, H=None, regularization=nmfRegularization)

        # print(f'Min: {np.min(H_NMF)} Max: {np.max(H_NMF)}')
        if showPlots:
            plt.figure(figsize=(30, 3)), plt.pcolormesh(H_NMF), plt.title('H_NMF'), plt.show()
                    
        # Isolate the instruments via NMF reconstruction
        isolations = _reconstructDecompositions(W_NMF, H_NMF, phase, indexDict)

        # Write each isolation back to a int16 .wav file using the istft and normalization
        for idx, isolation in enumerate(isolations):
            assert check_NOLA(**self.getSpectrogramKwargs()), f"ERROR: Current spectrogram kwargs {self.getSpectrogramKwargs()} will not satisfy NOLA condition making inversion impossible!\nSee https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.check_NOLA.html for more info"
            # print(f'minSpec: {np.min(isolation)} maxSpec: {np.max(isolation)}')
            t, x = istft(isolation, fs=fs, **self.getSpectrogramKwargs())

            x = normalizeWAV(x)
            x = convertFloat32toInt16(x)
            # print(f'minRaw: {np.min(x)} maxRaw: {np.max(x)}')

            wavfile.write(f'decompositions\\decomposition{idx}.wav', rate=fs, data=x)



















