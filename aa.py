
import os
import sys
from pprint import pprint
import importlib.util

def load_module_from_path(module_name, path):
    abs_path = os.path.abspath(path)
    print(abs_path)
    sys.path.insert(0, abs_path)
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    print(spec)
    module = importlib.util.module_from_spec(spec)
    print(module)
    spec.loader.exec_module(module)
    print(module)
    sys.path.remove(abs_path)
    return module

print(os.path.abspath("."))
pprint(list(sys.modules.keys()))
spec = load_module_from_path('KonanXAI.XAI', './KonanXAI/xai.py')
pprint(list(sys.modules.keys()))