import os


def get_file_name(file_path: str) -> str:
    """Extracts the file name from a file path.

    Args:
        file_path (str): The file path.

    Returns:
        str: The file name.
    """
    return os.path.basename(file_path).split(".")[0]
