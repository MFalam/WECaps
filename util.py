import os
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def navigate_parent_dirs(path: str, levels: int) -> str:

    if levels < 0:
        raise ValueError("levels must be >= 0, not {}".format(levels))

    result = os.path.abspath(os.path.join(path, os.path.sep.join(".." for _ in range(levels))))

    if os.path.isfile(result):
        return os.path.dirname(result)

    return result


def makedirs(path: str) -> None:

    if os.path.isfile(path):
        raise ValueError("path '{}' is an existing file; cannot create as a directory".format(path))

    os.makedirs(path, exist_ok=True)


def find_repo_root() -> str:

    return navigate_parent_dirs(os.path.dirname(__file__), 2)


def find_data_root() -> str:

    return os.path.join(find_repo_root(), DATA_ROOT_DIR)


def find_data_dir(relative_path: str, create=True) -> str:

    custom_dir_absolute = os.path.join(find_data_root(), relative_path)

    if create:
        makedirs(custom_dir_absolute)

    return custom_dir_absolute
