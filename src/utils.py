import os

def get_label_from_filename(file_path):
    filename = os.path.basename(file_path)
    label = filename.split('__')[1][0]
    return 0 if label == "F" else 1