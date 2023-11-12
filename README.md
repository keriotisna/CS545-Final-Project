# CS545-Final-Project

### Decomposition considerations:
- Instead of using individual spectrogram frames, use 3-10 frames at a time and count them as a single frame by doing vec() on them. This provides a bit more time context
- Ensure consistent sample rates between all audio samples

### Reconstruction considerations:
- How do we detect distortions in reconstructed samples??
- Maybe use VAEs to reconstruct holes in data
- We could use a CNN on the spectrogram to classify a given portion of audio, but we may want to try GMMs or other simple classifiers first
