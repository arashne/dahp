import glob
import importlib


def append_class_to_path(path, cls):
    return f'{path}/{cls.__module__}.{cls.__name__}'


def load_class(path):
    # Find which child class is this
    files = glob.glob(f'{path}*')
    assert len(files) == 1, f'searching for {path} following files are found: ' + '\n'.join(files)
    child_load_path = files[0]

    class_specs = child_load_path.split('/')[-1].split('.')[-3:]
    module_name = '.'.join(class_specs[:-1])
    class_name = class_specs[-1]

    module = importlib.import_module(module_name)
    child_class = getattr(module, class_name)

    return child_class
