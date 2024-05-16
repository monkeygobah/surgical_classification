import os
import csv
import random

def extract_info_and_save_csv(directory_paths, train_csv_path, test_csv_path, split_ratio=0.8):
    data = []

    for directory_path in directory_paths:
        for filename in os.listdir(directory_path):
            if filename.endswith(".mp4"):
                file_path = os.path.join(directory_path, filename)
                absolute_file_path = os.path.abspath(file_path)

                parts = filename.split('_')
                last_number = parts[-1].split('.')[0]  # Split on dot and take the first part

                data.append([absolute_file_path, int(last_number)-1])

    # Shuffle the data randomly
    random.shuffle(data)

    # Split the data into train and test sets
    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]

    # Save train data to CSV file
    with open(train_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(train_data)

    # Save test data to CSV file
    with open(test_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(test_data)

base_directory = 'data'
directory_paths = [os.path.join(base_directory, f'Phase_{i}') for i in range(1, 11)]
train_csv_path = 'data/train_all_phases.csv'
test_csv_path = 'data/test_all_phases.csv'

extract_info_and_save_csv(directory_paths, train_csv_path, test_csv_path)