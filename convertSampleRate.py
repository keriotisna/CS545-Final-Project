import os
from pydub import AudioSegment
import argparse
import struct

def find_min_sample_rate(folder_path):
    min_rate = None
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.aiff', '.wav', '.aif')):
                audio = AudioSegment.from_file(os.path.join(root, file))
                if min_rate is None or audio.frame_rate < min_rate:
                    min_rate = audio.frame_rate
                    print(f'Current min rate: {min_rate}')
    return min_rate

def find_min_sample_rate_fast(directory):
    
    """
    Find the min sample rate of .wav files in a directory by examining file header data instead of reading the whole .wav
    """
    
    min_sample_rate = None

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                try:
                    with open(os.path.join(root, file), 'rb') as wav_file:
                        wav_file.seek(24)  # Jump to the sample rate in the header
                        sample_rate = struct.unpack('<I', wav_file.read(4))[0]
                        if min_sample_rate is None or sample_rate < min_sample_rate:
                            min_sample_rate = sample_rate
                            print(f'Current min rate: {min_sample_rate}')

                except Exception as e:
                    print(f"Error reading {file}: {e}")

    return min_sample_rate

def convert_audio_files(source_folder, target_folder, target_sample_rate):
    for root, dirs, files in os.walk(source_folder):
        relative_path = os.path.relpath(root, source_folder)
        target_path = os.path.join(target_folder, relative_path)
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        for file in files:
            if file.endswith(('.aiff', '.wav', '.aif')):
                audio = AudioSegment.from_file(os.path.join(root, file))
                audio = audio.set_frame_rate(target_sample_rate)
                target_file = os.path.splitext(file)[0] + '.wav'
                audio.export(os.path.join(target_path, target_file), format='wav')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert audio files to a minimum sample rate.')
    parser.add_argument('source_folder', type=str, help='Path to the source folder containing original audio files.')
    parser.add_argument('target_folder', type=str, help='Path to the target folder containing converted audio files.')
    parser.add_argument('sample_rate_override', type=int, help='The override sample rate to be used for converted audio files. Set to 0 for normal usage')
    return parser.parse_args()

if __name__ == "__main__":
    # Usage: python convertSampleRate.py "source-folder" "target-folder" 0
    args = parse_arguments()
    source_folder = os.path.normpath(args.source_folder)
    target_folder = os.path.normpath(args.target_folder)
    sample_rate_override = args.sample_rate_override
    
    print(f'Source folder: {source_folder}')
    print(f'Target folder: {target_folder}')
    
    min_rate = None
    if sample_rate_override == 0:
        min_rate = find_min_sample_rate_fast(source_folder)
        print(f'Min sample rate is {min_rate}')
    else:
        min_rate = sample_rate_override
        print(f'Overridden sample rate is {min_rate}')
    
    print(f'Starting conversion...')
    convert_audio_files(source_folder, target_folder, min_rate)
    print('Done!')
