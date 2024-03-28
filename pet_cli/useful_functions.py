"""
Module to handle abstracted functionalities
"""
import os

def make_path(paths: list[str]):
    """
    Creates a new path in local system by joining paths, and making any new directories, if
    necessary.

    Args:
        paths (list[str]): A list containing strings to be joined as a path in the system
            directory.

    Note:
        If the final string provided includes a period '.' (a proxy for checking if the path is a 
        file name) this method will result in creating the folder above the last provided string in
        the list.
    """
    end_dir = paths[-1]
    if end_dir.find('.') == -1:
        out_path = os.path.join(paths)
    else:
        out_path = os.path.join(paths[:-1])
    os.makedirs(out_path,exist_ok=True)
