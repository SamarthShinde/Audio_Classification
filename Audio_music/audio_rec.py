import sounddevice as sd
from scipy.io.wavfile import write, read
import numpy as np
import os
from datetime import datetime
import csv

metadata_file = "audio_metadata.csv"

class_id_map = {
    'human_voice': 1,
    'music': 2,
    'human_with_music': 3
}

def get_valid_input(prompt, valid_options, option_map=None):
    while True:
        user_input = input(prompt).lower()
        if user_input in valid_options:
            if option_map:
                return option_map[user_input]
            return user_input
        print('Please enter a valid option.\n')

def get_user_configuration():
    audio_name = input('Enter audio file name:  \n').lower()

    environment = get_valid_input(
        'Is this recording in-car or in-room? \n 1: In-Car \n 2: In-Room \n',
        ['1', '2'],
        {'1': 'in_car', '2': 'in_room'}
    )

    num_people = int(input('How many people are present in the room/car? \n'))

    is_music_playing = get_valid_input('Is music being played? (y/n): \n', ['y', 'n'])

    if is_music_playing == 'y':
        music_volume = get_valid_input('What is the music volume level? \n 1: Low \n 2: Medium \n 3: High \n',
                                       ['1', '2', '3'],
                                       {'1': 'low', '2': 'medium', '3': 'high'})
    else:
        music_volume = 'no_music'

    if environment == 'in_car':
        windows = get_valid_input('Are the windows open or closed? \n 1: Open \n 2: Closed \n',
                                  ['1', '2'],
                                  {'1': 'open', '2': 'closed'})

        num_windows_open = 0
        if windows == 'open':
            num_windows_open = int(input('How many windows are open? \n'))

        noise_types = []
        print('Select the types of noise present (you can choose multiple types): ')
        noise_count = 0
        for noise in ['engine', 'traffic', 'horn', 'wind']:
            include_noise = get_valid_input(f'Is there {noise} noise? (y/n): ', ['y', 'n'])
            if include_noise == 'y':
                noise_count += 1

        class_type = get_valid_input('What is the class of the audio? \n 1: Human Voice \n 2: Music \n 3: Human with Music \n',
                                     ['1', '2', '3'],
                                     {'1': 'human_voice', '2': 'music', '3': 'human_with_music'})

        silence_class = 1

        user_configuration = {
            'audio_name': audio_name,
            'environment': environment,
            'windows': windows,
            'num_windows_open': num_windows_open,
            'music_volume': music_volume,
            'noise_count': noise_count,
            'class_type': class_type,
            'silence_class': silence_class,
            'num_people': num_people
        }

    elif environment == 'in_room':
        class_type = get_valid_input('What is the class of the audio? \n 1: Human Voice \n 2: Music \n 3: Human with Music \n',
                                     ['1', '2', '3'],
                                     {'1': 'human_voice', '2': 'music', '3': 'human_with_music'})

        silence_class = 0

        noise_count = 0  # Assume no noise types for room recordings unless otherwise specified

        user_configuration = {
            'audio_name': audio_name,
            'environment': environment,
            'class_type': class_type,
            'silence_class': silence_class,
            'music_volume': music_volume,
            'noise_count': noise_count,
            'num_people': num_people
        }

    return user_configuration

def record_audio(output_directory, user_configuration, duration, sample_rate=44100, channels=1):
    if user_configuration is None:
        print("No configuration provided.")
        return None

    date_folder = datetime.now().strftime("%Y-%m-%d")
    environment = user_configuration['environment']
    fold = f"{environment}_{date_folder}"

    output_directory = os.path.join(output_directory, fold)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_file = os.path.join(output_directory,
                               f"{user_configuration['audio_name']}_{datetime.now().strftime('%H%M%S')}.wav")

    print(f"Recording for {duration} seconds... in process")
    recorded_voice = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)
    sd.wait()
    write(output_file, sample_rate, recorded_voice)
    print(f"Recording finished...\nSaved as: {output_file}")

    # Segment audio if duration is greater than 30 seconds
    segment_files = []
    if duration > 30:
        segment_files = segment_audio(output_file, duration, sample_rate)
        os.remove(output_file)  # Delete the main audio file after segmentation
    else:
        segment_files.append(output_file)  # Add the whole file as one segment if <= 30 sec

    return segment_files, date_folder, fold

def segment_audio(file_path, duration, sample_rate):
    _, data = read(file_path)
    num_segments = duration // 10
    base_name = os.path.splitext(file_path)[0]
    segment_files = []
    for i in range(num_segments):
        segment_data = data[i * 10 * sample_rate:(i + 1) * 10 * sample_rate]
        segment_file = f"{base_name}_segment_{i + 1}.wav"
        write(segment_file, sample_rate, segment_data)
        segment_files.append(segment_file)
        print(f"Segment saved as: {segment_file}")
    return segment_files

def update_metadata(metadata_file, user_configuration, segment_files, folder_name, fold, duration):
    metadata_dir = os.path.dirname(metadata_file)
    if metadata_dir and not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)

    if not os.path.isfile(metadata_file):
        with open(metadata_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=[
                'slice_file_name', 'fold', 'start', 'end', 'silence_class', 'music_volume', 'noise_count', 'class_type', 'num_people', 'comment'
            ])
            writer.writeheader()

    comment = input("Enter any comments for this recording: \n")

    for i, segment_file in enumerate(segment_files):
        metadata = {
            'slice_file_name': os.path.basename(segment_file),
            'fold': fold,
            'start': i * 10,
            'end': (i + 1) * 10,
            'silence_class': user_configuration['silence_class'],
            'music_volume': user_configuration['music_volume'],
            'noise_count': user_configuration['noise_count'],
            'class_type': user_configuration['class_type'],
            'num_people': user_configuration['num_people'],
            'comment': comment
        }
        with open(metadata_file, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=metadata.keys())
            writer.writerow(metadata)

def main():
    output_directory = "audio_data"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    while True:
        user_configuration = get_user_configuration()
        duration = int(input("Enter the duration of the recording (in seconds): \n"))
        segment_files, folder_name, fold = record_audio(output_directory, user_configuration, duration)
        if segment_files:
            update_metadata(metadata_file, user_configuration, segment_files, folder_name, fold, duration)
            print("Audio recording and metadata update completed.")

        # Ask if the user wants to record again
        record_again = get_valid_input('Do you want to record another audio? (y/n): ', ['y', 'n'])
        if record_again == 'n':
            break

if __name__ == "__main__":
    main()