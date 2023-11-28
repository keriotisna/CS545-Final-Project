import os

prefixFolderDict = {
    'bass': 'bass',
    'brass': 'brass',
    'flute': 'flute',
    'guitar': 'guitar',
    'keyboard': 'keyboard',
    'mallet': 'mallet',
    'organ': 'organ',
    'reed': 'reed',
    'string': 'string',
    'vocal_synthetic': 'synth_lead',
    'vocal_acoustic': 'vocal'
}

DATA_PATH = os.path.normpath(r'C:\Users\Nicholas\Documents\Desktop\SCHOOL\GRADUATE Offline\CS 545\Final Project\CS545-Final-Project\data\nsynth-valid\audio')

for root, dirs, files in os.walk(DATA_PATH):
    
    for file in files:
        
        for key, value in prefixFolderDict.items():
            if file.startswith(key):
                dirParts = root.split(os.sep)
                targetFolder = value
                
                dirParts.append(targetFolder)
                dirParts.append(file)

                oldPath = os.path.normpath(os.path.join(root, file))
                newPath = os.path.normpath(os.path.join(root, targetFolder, file))

                os.rename(oldPath, newPath)







