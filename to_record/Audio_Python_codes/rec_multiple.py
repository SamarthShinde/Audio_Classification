import sounddevice as sd
from scipy.io.wavfile import write
import os
from datetime import datetime

# Enter duration to record the audio in seconds
duration = 300

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
            print("    Audio Name:    ",    previous_config['audio_name'])
            print("    Type of Audio: ", previous_config['type_of_audio'])
            print("    Emotion:       ",       previous_config['emotion'])
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

            type_of_audio = get_valid_input('Enter type of audio: \n 1: male, \n 2: female, \n 3: engine, \n 4: traffic: \n',
                                            ['1', '2', '3', '4'],
                                            {'1': 'male', '2': 'female', '3': 'engine', '4': 'traffic'})

            if type_of_audio in ['male', 'female']:
                emotion = get_valid_input('Enter emotions if any \n 1: angry , 2: loud: , 3: normal  \n',
                                          ['1', '2', '3'],
                                          {'1': 'angry', '2': 'loud', '3': 'normal'})
            else:
                emotion = None

            return audio_name, type_of_audio, emotion

        audio_name, type_of_audio, emotion = get_audio_details()

        user_configuration = {'audio_name': audio_name, 'type_of_audio': type_of_audio, 'emotion': emotion}

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

    if type_of_audio in ['male', 'female'] and user_configuration['emotion']:
        output_file = os.path.join(output_directory, f"{user_configuration['audio_name']}_{user_configuration['emotion']}_{datetime.now().strftime('%H%M%S')}.wav")
    else:
        output_file = os.path.join(output_directory, f"{user_configuration['audio_name']}_{datetime.now().strftime('%H%M%S')}.wav")

    print(f"Recording for {duration} seconds... in process")
    recorded_voice = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)
    sd.wait()
    write(output_file, sample_rate, recorded_voice)
    print(f"Recording finished...\nSaved as: {output_file}")
    return output_file

def main():
    output_directory = "audiorec"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    while True:
        user_configuration = get_user_configuration()
        output_file = record_audio(output_directory, user_configuration, duration)
        print("Audio recording completed.")

        # Ask if the user wants to record again
        record_again = get_valid_input('\n \n Do you want to record another audio? (y/n): ', ['y', 'n'])
        if record_again == 'n':
            break

if __name__ == "__main__":
    main()
