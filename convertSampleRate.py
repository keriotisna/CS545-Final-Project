import os
from pydub import AudioSegment
import argparse

def find_min_sample_rate(folder_path):
    min_rate = None
    for file in os.listdir(folder_path):
        if file.endswith('.aiff') or file.endswith('.wav') or file.endswith('.aif'):
            audio = AudioSegment.from_file(os.path.join(folder_path, file))
            if min_rate is None or audio.frame_rate < min_rate:
                min_rate = audio.frame_rate
    return min_rate

def convert_audio_files(source_folder, target_folder, target_sample_rate):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for file in os.listdir(source_folder):
        if file.endswith('.aiff') or file.endswith('.wav') or file.endswith('.aif'):
            audio = AudioSegment.from_file(os.path.join(source_folder, file))
            audio = audio.set_frame_rate(target_sample_rate)
            target_file = os.path.splitext(file)[0] + '.wav'
            audio.export(os.path.join(target_folder, target_file), format='wav')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Check the success of audio file conversion.')
    parser.add_argument('source_folder', type=str, help='Path to the source folder containing original audio files.')
    parser.add_argument('target_folder', type=str, help='Path to the target folder containing converted audio files.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    source_folder = args.source_folder
    target_folder = args.target_folder
    min_rate = find_min_sample_rate(source_folder)

    convert_audio_files(source_folder, target_folder, min_rate)
