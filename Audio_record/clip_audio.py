import pandas as pd
from pydub import AudioSegment, silence
import numpy as np
import os
import csv
import random

def detect_non_silent(audio, silence_thresh=-40, min_silence_len=1000):
    non_silent_ranges = silence.detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    return non_silent_ranges

def trim_silence(audio, silence_thresh=-40, min_silence_len=1000):
    non_silent_ranges = detect_non_silent(audio, silence_thresh, min_silence_len)
    if not non_silent_ranges:
        return None
    start_trim = non_silent_ranges[0][0]
    end_trim = non_silent_ranges[-1][1]
    trimmed_audio = audio[start_trim:end_trim]
    return trimmed_audio

def segment_audio(audio, segment_length=10000):
    audio_length = len(audio)
    segments = []
    start = 0
    while start < audio_length:
        end = start + segment_length
        if end > audio_length:
            end = audio_length
        segments.append(audio[start:end])
        start = end
    return segments

def save_segments(segments, base_name, output_dir='output_segments'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    metadata = []
    for i, segment in enumerate(segments):
        segment_name = f'{base_name}_segment_{i + 1}.wav'
        segment_path = os.path.join(output_dir, segment_name)
        segment.export(segment_path, format="wav")
        metadata.append(segment_name)
    return metadata

def process_audio(input_file, metadata_file, output_dir='no_audio2', silence_thresh=-40, min_silence_len=1000):
    audio = AudioSegment.from_wav(input_file)
    trimmed_audio = trim_silence(audio, silence_thresh, min_silence_len)
    if not trimmed_audio:
        print("No non-silent audio detected.")
        return

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    segments = segment_audio(trimmed_audio, segment_length=10000)
    segment_names = save_segments(segments, base_name, output_dir)

    # Read the existing metadata file
    df = pd.read_csv(metadata_file)

    # Define additional metadata for the new segments
    fsID = random.randint(100000, 400000)
    fold = 'no_audio2'
    classID = 4
    class_label = 'No_sound'
    comment = 'No sound Audio data from downloaded youtude'

    new_metadata = []
    start_time = 0
    for segment_name in segment_names:
        audio_duration = len(AudioSegment.from_wav(os.path.join(output_dir, segment_name))) / 1000  # duration in seconds
        end_time = start_time + audio_duration
        new_metadata.append([segment_name, fsID, start_time, end_time, 0, fold, classID, class_label, comment])
        start_time = end_time

    # Convert to DataFrame and append to existing metadata
    new_metadata_df = pd.DataFrame(new_metadata,
                                   columns=['slice_file_name', 'fsID', 'start', 'end', 'salience', 'fold', 'classID',
                                            'class', 'comment'])
    df = pd.concat([df, new_metadata_df], ignore_index=True)

    # Save the updated metadata back to the CSV file
    df.to_csv(metadata_file, index=False)
    print(f"Processed {len(segments)} segments and saved in '{output_dir}'.")

# Example usage
input_file = "/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/to_record/untitled folder/2 HOUR STUDY WITH ME at the LIBRARY Background noise no breaks real time countdown timer [TubeRipper.com].wav"  # Replace with the actual path to your input WAV file
metadata_file = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/to_record/metadata_test.csv'  # Replace with the actual path to the metadata CSV file
process_audio(input_file, metadata_file)
