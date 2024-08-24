import os
import sounddevice as sd
from scipy.io.wavfile import write, read
import numpy as np
import csv
from datetime import datetime
import librosa

metadata_file = "audio_metadata.csv"

class_id_map = {
    'human_voice': 1,
    'music': 2,
    'human_with_music': 3
}

valid_audio_extensions = ['.wav', '.mp3', '.flac']  # List of valid audio file extensions

def get_valid_input(prompt, valid_options, option_map=None):
    while True:
        user_input = input(prompt).lower()
        if user_input in valid_options:
            if option_map:
                return option_map[user_input]
            return user_input
        print('Please enter a valid option.\n')

def extract_config_from_filename(filename):
    # Extract the part after "0000-"
    code = filename.split("0000-")[-1][:4]

    if len(code) != 4:
        return None  # Invalid code length

    # Extract configuration based on the code
    engine_status = 'on' if code[0] == '1' else 'off'
    windows_status = 'open' if code[1] == '1' else 'closed'
    num_windows_open = 2 if windows_status == 'open' else 0
    music_status = 'on' if code[2] == '1' else 'off'
    num_people = int(code[3])

    return {
        'engine_status': engine_status,
        'windows_status': windows_status,
        'num_windows_open': num_windows_open,
        'music_status': music_status,
        'num_people': num_people
    }

def get_user_configuration(defaults=None):
    if defaults:
        # If configuration extracted from filename, fill them automatically
        engine_status = defaults['engine_status']
        windows_status = defaults['windows_status']
        num_windows_open = defaults['num_windows_open']
        music_status = defaults['music_status']
        num_people = defaults['num_people']

        print(f"Extracted configurations from filename:")
        print(f"Engine Status: {engine_status}")
        print(f"Windows Status: {windows_status} (Number of windows open: {num_windows_open})")
        print(f"Music Status: {music_status}")
        print(f"Number of People: {num_people}")

        # By default, engine noise is present if the engine is on
        noise_count = 1 if engine_status == 'on' else 0

        if windows_status == 'open':
            noise_count += 1  # Assuming wind noise when windows are open

        # Ask only for additional information
        if music_status == 'on':
            music_volume = get_valid_input('What is the music volume level? \n 1: Low \n 2: Medium \n 3: High \n',
                                           ['1', '2', '3'],
                                           {'1': 'low', '2': 'medium', '3': 'high'})
        else:
            music_volume = 'no_music'

        # Check for additional noise types
        print('Select additional types of noise present (you can choose multiple types): ')
        for noise in ['traffic', 'horn']:
            include_noise = get_valid_input(f'Is there {noise} noise? (y/n): ', ['y', 'n'])
            if include_noise == 'y':
                noise_count += 1

        class_type = get_valid_input(
            'What is the class of the audio? \n 1: Human Voice \n 2: Music \n 3: Human with Music \n',
            ['1', '2', '3'],
            {'1': 'human_voice', '2': 'music', '3': 'human_with_music'})

        silence_class = 1

        user_configuration = {
            'environment': 'in_car',
            'engine_status': engine_status,
            'windows': windows_status,
            'num_windows_open': num_windows_open,
            'music_volume': music_volume,
            'noise_count': noise_count,
            'class_type': class_type,
            'silence_class': silence_class,
            'num_people': num_people
        }

    else:
        # Prompt for full configuration when defaults are not provided
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

            num_windows_open = 2 if windows == 'open' else 0

            noise_count = 0
            print('Select the types of noise present (you can choose multiple types): ')
            for noise in ['engine', 'traffic', 'horn', 'wind']:
                include_noise = get_valid_input(f'Is there {noise} noise? (y/n): ', ['y', 'n'])
                if include_noise == 'y':
                    noise_count += 1

            class_type = get_valid_input(
                'What is the class of the audio? \n 1: Human Voice \n 2: Music \n 3: Human with Music \n',
                ['1', '2', '3'],
                {'1': 'human_voice', '2': 'music', '3': 'human_with_music'})

            silence_class = 1

            user_configuration = {
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
            class_type = get_valid_input(
                'What is the class of the audio? \n 1: Human Voice \n 2: Music \n 3: Human with Music \n',
                ['1', '2', '3'],
                {'1': 'human_voice', '2': 'music', '3': 'human_with_music'})

            silence_class = 0
            noise_count = 0  # Assume no noise types for room recordings

            user_configuration = {
                'environment': environment,
                'class_type': class_type,
                'silence_class': silence_class,
                'music_volume': music_volume,
                'noise_count': noise_count,
                'num_people': num_people
            }

    return user_configuration

def segment_audio(file_path, sample_rate):
    _, data = read(file_path)
    duration = len(data) // sample_rate
    segment_files = []

    if duration <= 30:
        segment_files.append(file_path)
    else:
        num_segments = (duration // 10) + 1
        base_name = os.path.splitext(file_path)[0]
        for i in range(num_segments):
            start_sample = i * 10 * sample_rate
            end_sample = min((i + 1) * 10 * sample_rate, len(data))
            segment_data = data[start_sample:end_sample]

            # Check if the segment has audio
            if np.abs(segment_data).mean() < 0.001:
                print(f"Segment {i + 1} has no significant audio, skipping...")
                continue

            segment_file = f"{base_name}_segment_{i + 1}.wav"
            write(segment_file, sample_rate, segment_data)
            segment_files.append(segment_file)
            print(f"Segment saved as: {segment_file}")
    return segment_files

def update_metadata(metadata_file, user_configuration, segment_file, fold, start_time, end_time):
    metadata_dir = os.path.dirname(metadata_file)
    if metadata_dir and not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)

    if not os.path.isfile(metadata_file):
        with open(metadata_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=[
                'slice_file_name', 'fold', 'start', 'end', 'silence_class', 'music_volume', 'noise_count', 'class_type',
                'num_people', 'comment'
            ])
            writer.writeheader()

    comment = input("Enter any comments for this recording: \n")

    metadata = {
        'slice_file_name': os.path.basename(segment_file),
        'fold': fold,
        'start': start_time,
        'end': end_time,
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

def annotate_existing_audio_files():
    dataset_location = input('Enter the location of the audio dataset: \n')
    audio_directory = dataset_location
    if not os.path.exists(audio_directory):
        print("Invalid directory.")
        return

    audio_files = [f for f in os.listdir(audio_directory) if
                   os.path.isfile(os.path.join(audio_directory, f)) and os.path.splitext(f)[
                       -1].lower() in valid_audio_extensions]

    previous_configuration = None

    for audio_file in audio_files:
        file_path = os.path.join(audio_directory, audio_file)
        print(f"Processing {audio_file}...")

        sample_rate, _ = read(file_path)
        duration = librosa.get_duration(path=file_path)
        print(f"Duration of the audio: {duration:.2f} seconds")

        segments = segment_audio(file_path, sample_rate)

        fold = datetime.now().strftime("%Y-%m-%d")

        if '0000-' in audio_file:
            config_from_filename = extract_config_from_filename(audio_file)
        else:
            config_from_filename = None

        for i, segment_file in enumerate(segments):
            replay_segment = 'y'
            while replay_segment == 'y':
                print(f"Playing segment {i + 1}/{len(segments)}...")
                y_segment, sr_segment = librosa.load(segment_file, sr=None)
                sd.play(y_segment, sr_segment)
                sd.wait()
                replay_segment = get_valid_input("Would you like to listen to the audio again? (y/n): \n", ['y', 'n'])

            delete_segment = get_valid_input("Do you want to delete this segment? (y/n): \n", ['y', 'n'])
            if delete_segment == 'y':
                os.remove(segment_file)
                print(f"Segment {segment_file} deleted.")
                continue

            if previous_configuration:
                print("\nPrevious Configuration:")
                for key, value in previous_configuration.items():
                    print(f"{key.replace('_', ' ').title()}: {value}")

            # Ask if the user wants to use the previous configuration
            use_previous_config = get_valid_input("Do you want to use the previous configuration? (y/n): \n",
                                                  ['y', 'n'])

            if use_previous_config == 'y' and previous_configuration:
                user_configuration = previous_configuration
            else:
                user_configuration = get_user_configuration(config_from_filename)
                previous_configuration = user_configuration  # Save the current configuration for the next segment

            start_time = i * 10
            end_time = start_time + 10 if start_time + 10 <= duration else duration

            # Move the segment file to the appropriate folder
            new_segment_path = os.path.join(audio_directory, fold, os.path.basename(segment_file))
            if not os.path.exists(os.path.join(audio_directory, fold)):
                os.makedirs(os.path.join(audio_directory, fold))
            os.rename(segment_file, new_segment_path)

            # Update metadata
            update_metadata(metadata_file, user_configuration, new_segment_path, fold, start_time, end_time)
if __name__ == "__main__":
    annotate_existing_audio_files()