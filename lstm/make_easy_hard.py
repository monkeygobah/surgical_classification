import os
import shutil

def prepare_folders(root_dir, easy_phases, easy_folder, hard_folder):
    # Create the "easy" and "hard" folders if they don't exist
    os.makedirs(easy_folder, exist_ok=True)
    os.makedirs(hard_folder, exist_ok=True)



    # Iterate over the phase folders in the root directory
    for phase_folder in os.listdir(root_dir):
        if phase_folder.startswith("Phase_"):
            phase = int(phase_folder.split("_")[1])  # Extract the phase number from the folder name
            phase_folder_path = os.path.join(root_dir, phase_folder)

            # Iterate over the video files in the phase folder
            for file_name in os.listdir(phase_folder_path):
                if file_name.endswith(".mp4"):
                    source_path = os.path.join(phase_folder_path, file_name)
                    new_file_name = f"(phase_{phase}){file_name}"  # Modify the file name

                    if phase in easy_phases:
                        destination_path = os.path.join(easy_folder, new_file_name)
                    else:
                        destination_path = os.path.join(hard_folder, new_file_name)

                    shutil.copy2(source_path, destination_path)
if __name__ == '__main__':
    root_dir = "data/"
    easy_phases = [2, 3, 4, 5, 10]  # Replace with your list of easy phases
    easy_folder = "easy_phases/"  #
    hard_folder = "hard_phases/"  

    prepare_folders(root_dir, easy_phases, easy_folder, hard_folder)