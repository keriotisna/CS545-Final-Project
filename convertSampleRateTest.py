import os
from pydub import AudioSegment
import argparse
from convertSampleRate import find_min_sample_rate

def is_conversion_successful(source_folder, target_folder, expected_sample_rate):
    success = True

    # Check for each file in the source folder
    for file in os.listdir(source_folder):
        if file.endswith('.aiff') or file.endswith('.wav') or file.endswith('.aif') :
            # Construct the expected output file path
            target_file_path = os.path.join(target_folder, os.path.splitext(file)[0] + '.wav')

            # Check if the file exists in the target folder
            if not os.path.exists(target_file_path):
                print(f"File not found in target folder: {target_file_path}")
                success = False
                continue

            # Check the sample rate of the converted file
            audio = AudioSegment.from_file(target_file_path)
            if audio.frame_rate != expected_sample_rate:
                print(f"Sample rate mismatch for {target_file_path}. Expected: {expected_sample_rate}, Found: {audio.frame_rate}")
                success = False

    return success

def parse_arguments():
    parser = argparse.ArgumentParser(description='Check the success of audio file conversion.')
    parser.add_argument('source_folder', type=str, help='Path to the source folder containing original audio files.')
    parser.add_argument('target_folder', type=str, help='Path to the target folder containing converted audio files.')
    return parser.parse_args()


if __name__ == "__main__":
# Usage: python3 convertSampleRate.py input_dir output_dir
    args = parse_arguments()
    source_folder = args.source_folder
    target_folder = args.target_folder
    expected_sample_rate = find_min_sample_rate(source_folder)  

    if is_conversion_successful(source_folder, target_folder, expected_sample_rate):
        print("Conversion successful for all files.")
    else:
        print("Conversion failed for one or more files.")
