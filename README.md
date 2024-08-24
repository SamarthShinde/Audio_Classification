# Audio Classification Project

[![GitHub license](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/samarth-shinde/your-repository-name/blob/master/LICENSE)

This repository contains the codebase for an audio classification project aimed at detecting and classifying various sounds inside a car. The project is structured in stages, with the ultimate goal of identifying the emotions of a driver or passenger based on their speech while other sounds (like music) are playing in the background.

## Project Overview

### Stages of the Project

1. **Stage 1: Basic Audio Classification**
   - Classifies five audio classes: male, female, engine, traffic, and music.
   - This stage is fully implemented and includes a CNN model for classification.

2. **Stage 2: Speech Detection in Noisy Environment**
   - Focuses on detecting human speech inside a car while music is playing.
   - This stage is currently in progress.

3. **Stage 3: Emotion Detection**
   - Aims to detect the emotional state (angry, sad, happy, frustrated) of a speaker inside a car.
   - This stage has not yet been started.

### Classes Included

- **Male Voice**
- **Female Voice**
- **Engine Noise**
- **Traffic Noise**
- **Music**
- **No Sound**

## Project Structure

The project is organized into the following folders:

- **`audio_datasets/`**: Contains modified CSV files derived from UrbanSound8K and Urban Sounds, used to identify various environmental sounds like traffic, children playing, etc.

- **`audio_model/`**: Includes Python programs (`cnn-model.py`, `inf.py`, `label.py`) for training and inference using a CNN model. The code here is responsible for the classification of audio in Stage 1.

- **`audio_music/`**: Contains code for annotating audio, particularly for Stage 2 of the project. The `audio_annotation.py` script is used for categorizing audio files into the necessary classes for speech detection.

- **`audio_record/`**: Houses scripts for audio recording and processing:
  - `audio_rec.py`: Records audio.
  - `clip_audio.py`: Clips audio files.
  - `convert_video_to_wav.py`: Extracts audio from video files and converts them to WAV format.

- **`combine_logic/`**: Implements logic for combining class detections:
  - Level 1 detects whether the engine is on or off.
  - Level 2 detects whether a person is talking inside the car and categorizes the audio (e.g., male, female, music, no sound).

- **`H_file/`**: Stores all models used in the project.

- **`label_encoder/`**: Contains all the label encodings used across the project.

- **`MetaData codes/`**: Scripts for modifying metadata. Each script is designed to handle specific modifications to the metadata used in the project.

- **`Real-time Prediction/`**: Contains scripts for real-time data prediction. These scripts are designed to run on devices like Jetson Nano, displaying results in various formats such as pie charts and tables.

- **`wav_files_codes/`**: Handles WAV file operations, including conversion from MP3 to WAV, MP4 to WAV, etc.

## Requirements

The project was developed on a MacBook with an M1 chip. The `requirements.txt` file in the repository lists all the necessary Python dependencies. Some dependencies may vary depending on the operating system or architecture.

### Install Dependencies

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Getting Started

### Switch to the `master` Branch

All the code for this project is located in the `master` branch. To access the code, switch to the `master` branch:

```bash
git checkout master
```

### Clone the Repository

To get started, clone the repository:

```bash
git clone https://github.com/samarth-shinde/your-repository-name.git
cd your-repository-name
git checkout master
```

### Download the Dataset

The datasets required for this project are hosted on Google Drive. Use the provided script to download and extract the datasets:

```bash
bash download_data.sh
```

The datasets include audio files used for training and testing the models.

### Running the Models

To train or test the models, navigate to the `audio_model/` directory and run the appropriate scripts. Detailed instructions for each stage can be found in the respective folders.

## Future Work

- **Stage 2 Completion**: Finalizing the detection of human speech inside a car while music is playing.
- **Stage 3 Implementation**: Beginning work on detecting the emotional states of speakers.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](https://github.com/samarth-shinde/your-repository-name/blob/master/LICENSE) file for details.

## Contact

If you have any questions, feel free to reach out:

- **Name**: Samarth Shinde
- **Email**: samarth.shinde505@gmail.com
- **GitHub**: [SamarthShinde](https://github.com/SamarthShinde)
```

### Key Adjustments:
- **Removed Unnecessary Text**: Removed the text that was showing the file as a template instead of as the actual README content.
- **Proper Markdown Formatting**: Ensured that headers (`#`, `##`, `###`), lists (`-`, `*`), and inline code blocks (`` ` ``) are used correctly.

This should render correctly on GitHub. You can update your `README.md` file with this content, and it should look properly formatted in your repository. Let me know if you need any more help!
