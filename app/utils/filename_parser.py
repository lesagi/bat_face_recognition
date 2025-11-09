import os
import re
from collections import defaultdict


def parse_filename_class(filename):
    """
    Parse filename to extract class information.
    
    Expected pattern: (?<type>\w)--(?<class>\w+)--(?<id>\w+(\.\d+)?)(?<aug_suffix>--aug(?<aug_id>\d{3}))?
    
    Args:
        filename: The filename to parse
        
    Returns:
        tuple: (type, class_name, id, aug_id) or None if parsing fails
    """
    name_without_ext = os.path.splitext(filename)[0]
    
    pattern = r'^(?P<type>\w+)--(?P<class>\w+)--(?P<id>\w+(?:\.\d+)?)(?P<aug_suffix>--aug(?P<aug_id>\d{3}))?$'
    match = re.match(pattern, name_without_ext)

    if match:
        groups = match.groupdict()
        return groups.get('type'), groups.get('class'), groups.get('id'), groups.get('aug_id')
    else:
        return None


def group_files_by_class(file_paths):
    """
    Group files by their class based on filename parsing.
    
    Args:
        file_paths: List of file paths
        
    Returns:
        dict: Dictionary mapping class names to dict of ids to lists of file paths
              Structure: {class_name: {id: [file_paths]}}
    """
    class_files = defaultdict(lambda: defaultdict(list))
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        parsed = parse_filename_class(filename)
        
        if parsed:
            _, class_name, id, _ = parsed
            class_files[class_name][id].append(file_path)
        else:
            print(f"Warning: Skipping file with unexpected naming pattern: {filename}")
    
    return dict(class_files)


def is_augmented_file(filename):
    """
    Check if a file is an augmented version based on filename.
    
    Args:
        filename: The filename to check
        
    Returns:
        bool: True if the file is augmented (contains --aug###), False otherwise
    """
    parsed = parse_filename_class(filename)
    if parsed:
        _, _, _, aug_id = parsed
        return aug_id is not None
    return False

