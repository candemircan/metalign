import os

from metalign.utils import set_pycortex_filestore_path

if __name__ == "__main__":
    # get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # go one level up to the project root
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    # append the desired filestore path
    desired_path = os.path.join(project_root, "data", "external", "brain_data", "surfaces")
    set_pycortex_filestore_path(desired_path)