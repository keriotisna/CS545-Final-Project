import os
import soundfile

# Converts all the .aif files in a directory to .wav files (creating a new directory for the
# output .wav files).
def convertAIFToWav(inputDirectory, outputDirectory):
    # Recursively create the output directory
    for root, dirs, files in os.walk(inputDirectory):
        if root == inputDirectory:
            continue
        instrumentType = os.path.basename(root)

        # Create corresponding subfolder for the instrument in the output folder
        outputSubfolder = os.path.join(outputDirectory, instrumentType)
        os.makedirs(outputSubfolder, exist_ok=True)

        for file in files:
            if file.endswith(".aif"):
                inputPath = os.path.join(root, file)
                outputFile = os.path.join(outputSubfolder, os.path.splitext(file)[0] + ".wav")

                data, sampleRate = soundfile.read(inputPath)
                soundfile.write(outputFile, data, sampleRate, format='WAV', subtype='PCM_16')

iowaAif = "data/University-of-Iowa-Aif-Files"
iowaWav = "data/University-of-Iowa-Wav-Files"
convertAIFToWav(iowaAif, iowaWav)