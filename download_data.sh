#!/bin/bash

echo "Downloading dataset from Google Drive..."
curl -L -o dataset.zip "https://drive.google.com/uc?export=download&id=https://drive.google.com/file/d/14HrRYkE3XE5MNxXt2WTptIamagR8SnGy/view?usp=share_link"

echo "Unzipping dataset..."
unzip dataset.zip -d ./data

echo "Cleaning up..."
rm dataset.zip

echo "Dataset downloaded and extracted to the 'data' directory."