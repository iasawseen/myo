import os
from auto_testing.processing.preprocess import launch_pre_process


if __name__ == '__main__':

    def generate_file_paths():
        raw_data_folder = '..\\raw_data'
        for directory in os.listdir(raw_data_folder):
            dir_path = os.path.join(raw_data_folder, directory)
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                yield file_path

    launch_pre_process(generate_file_paths())
