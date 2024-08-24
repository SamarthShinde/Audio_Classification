import sounddevice as sd
from scipy.io.wavfile import write
import os
from datetime import datetime
import csv
import random

# Enter duration to record the audio in seconds
duration = 10
metadata_file = "metadata_test.csv"

class_id_map = {
    'male': 1,
    'female': 2,
    'engine_rev': 3,
    'traffic': 4,
    'air_conditioner': 5
}


def get_valid_input(prompt, valid_options, option_map=None):
    """
    Prompt the user for input and validate against a list of valid options.

    Args:
        prompt (str): The prompt to display to the user.
        valid_options (list): List of valid options.
        option_map (dict, optional): Map of user input to corresponding values (if needed).

    Returns:
        str: User-selected option.
    """
    while True:
        user_input = input(prompt).lower()
        if user_input in valid_options:
            if option_map:
                return option_map[user_input]
            return user_input
        print('Please enter a valid option.\n')


def get_user_configuration():
    """
    Prompt the user to enter configuration details.

    Returns:
        dict: User configuration details.
    """
    previous_config = None
    try:
        with open("previous_config.txt", "r") as file:
            previous_config = eval(file.read())
            print("Previous Configuration:")
            for key in ["audio_name", "type_of_audio", "condition", "music", "window", "people_in_car"]:
                if key in previous_config:
                    print(f"    {key.replace('_', ' ').title()}: {previous_config[key]}")
    except FileNotFoundError:
        pass

    use_previous = get_valid_input('Do you want to use the previous configuration? (y/n): ',
                                   ['y', 'n'])

    if use_previous == 'y' and previous_config:
        print("Using previous configuration.")
        return previous_config
    else:
        def get_audio_details():
            audio_name = input('Enter audio file name:  \n').lower()

            type_of_audio = get_valid_input(
                'Enter type of audio: \n 1: male, \n 2: female, \n 3: engine rev, \n 4: traffic, \n 5: air conditioner \n',
                ['1', '2', '3', '4', '5'],
                {'1': 'male', '2': 'female', '3': 'engine_rev', '4': 'traffic', '5': 'air_conditioner'})

            condition, music, window, people_in_car, traffic_type, salience = None, None, None, None, None, 1
            if type_of_audio in ['male', 'female']:
                condition = get_valid_input('Enter condition: \n 1: driving , 2: parking , 3: traffic, 4: lab \n',
                                            ['1', '2', '3', '4'],
                                            {'1': 'driving', '2': 'parking', '3': 'traffic', '4': 'lab'})
                if condition == 'lab':
                    salience = 0
                music = get_valid_input('Enter music status: \n 1: on , 2: off \n',
                                        ['1', '2'],
                                        {'1': 'on', '2': 'off'})
                window = get_valid_input('Enter window status: \n 1: open , 2: close \n',
                                         ['1', '2'],
                                         {'1': 'open', '2': 'close'})
                people_in_car = input('Enter number of people in the car: \n')
            elif type_of_audio == 'traffic':
                condition = get_valid_input('Enter condition of traffic: \n 1: high , 2: moderate , 3: low \n',
                                            ['1', '2', '3'],
                                            {'1': 'high', '2': 'moderate', '3': 'low'})
                traffic_type = get_valid_input('Enter type of traffic: \n 1: while driving , 2: on signal \n',
                                               ['1', '2'],
                                               {'1': 'while_driving', '2': 'on_signal'})
            elif type_of_audio == 'air_conditioner':
                print('No one in the car should talk while recording.')
                window = get_valid_input('Enter window status: \n 1: open , 2: close \n',
                                         ['1', '2'],
                                         {'1': 'open', '2': 'close'})

            return audio_name, type_of_audio, condition, music, window, people_in_car, traffic_type, salience

        audio_name, type_of_audio, condition, music, window, people_in_car, traffic_type, salience = get_audio_details()

        user_configuration = {
            'audio_name': audio_name,
            'type_of_audio': type_of_audio,
            'condition': condition,
            'music': music,
            'window': window,
            'people_in_car': people_in_car,
            'traffic_type': traffic_type,
            'salience': salience
        }

        # Save current configuration to file
        with open("previous_config.txt", "w") as file:
            file.write(str(user_configuration))

        return user_configuration


def record_audio(output_directory, user_configuration, duration, sample_rate=44100, channels=1):
    """
    Record audio and save to a WAV file.

    Args:
        output_directory (str): Output directory to save the audio file.
        user_configuration (dict): User configuration details.
        duration (int): Duration of the recording in seconds.
        sample_rate (int): Sampling rate of the recording.
        channels (int): Number of audio channels.
    """
    if user_configuration is None:
        print("Using previous configuration.")
        return None

    # Create folder based on the current date
    date_folder = datetime.now().strftime("%Y-%m-%d")
    type_of_audio = user_configuration['type_of_audio']
    output_directory = os.path.join(output_directory, date_folder, type_of_audio)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if type_of_audio in ['male', 'female']:
        output_file = os.path.join(output_directory,
                                   f"{user_configuration['audio_name']}_{user_configuration['condition']}_{user_configuration['music']}_{user_configuration['window']}_{user_configuration['people_in_car']}_{datetime.now().strftime('%H%M%S')}.wav")
    elif type_of_audio == 'traffic':
        output_file = os.path.join(output_directory,
                                   f"{user_configuration['audio_name']}_{user_configuration['condition']}_{user_configuration['traffic_type']}_{datetime.now().strftime('%H%M%S')}.wav")
    elif type_of_audio == 'air_conditioner':
        output_file = os.path.join(output_directory,
                                   f"{user_configuration['audio_name']}_{user_configuration['window']}_{datetime.now().strftime('%H%M%S')}.wav")

    print(f"Recording for {duration} seconds... in process")
    recorded_voice = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)
    sd.wait()
    write(output_file, sample_rate, recorded_voice)
    print(f"Recording finished...\nSaved as: {output_file}")
    return output_file, date_folder


def update_metadata(metadata_file, user_configuration, output_file, folder_name):
    """
    Update the metadata CSV file with the new recording details.

    Args:
        metadata_file (str): Path to the metadata CSV file.
        user_configuration (dict): User configuration details.
        output_file (str): Path to the output audio file.
        folder_name (str): Name of the folder where the audio file is saved.
    """
    # Ensure the metadata directory exists
    metadata_dir = os.path.dirname(metadata_file)
    if metadata_dir and not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)

    # Ensure the metadata file is initialized with headers if it doesn't exist
    if not os.path.isfile(metadata_file):
        with open(metadata_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=[
                'slice_file_name', 'fsID', 'start', 'end', 'salience', 'fold', 'classID', 'class'
            ])
            writer.writeheader()

    fsid = generate_unique_fsid(metadata_file)
    metadata = {
        'slice_file_name': os.path.basename(output_file),
        'fsID': fsid,
        'start': 0,
        'end': duration,
        'salience': user_configuration['salience'],
        'fold': folder_name,
        'classID': class_id_map[user_configuration['type_of_audio']],
        'class': user_configuration['type_of_audio'].capitalize()
    }

    with open(metadata_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metadata.keys())
        writer.writerow(metadata)


def generate_unique_fsid(metadata_file):
    """
    Generate a unique FSID that is not already present in the metadata file.

    Args:
        metadata_file (str): Path to the metadata CSV file.

    Returns:
        int: A unique FSID.
    """
    existing_fsids = set()

    if os.path.isfile(metadata_file):
        with open(metadata_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    existing_fsids.add(int(row['fsID']))
                except KeyError:
                    continue  # Handle missing 'fsID' in existing records

    while True:
        fsid = random.randint(100000, 999999)
        if fsid not in existing_fsids:
            return fsid


def main():
    output_directory = "audiorec"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    while True:
        user_configuration = get_user_configuration()
        output_file, folder_name = record_audio(output_directory, user_configuration, duration)
        if output_file:
            update_metadata(metadata_file, user_configuration, output_file, folder_name)
            print("Audio recording and metadata update completed.")

        # Ask if the user wants to record again
        record_again = get_valid_input('\n \n Do you want to record another audio? (y/n): ', ['y', 'n'])
        if record_again == 'n':
            break


if __name__ == "__main__":
    main()
