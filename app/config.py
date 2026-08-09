import os
from pathlib import Path
import yaml
from types import SimpleNamespace
from dotenv import load_dotenv

load_dotenv()
# def find_folder(root_dir,target_dir):
#     for dirpath,dirnames,filenames in os.walk(root_dir):
#         if target_dir in dirnames:
#             return os.path.join(dirpath, target_dir)
#     return None

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v)for k, v in d.items()})
    elif isinstance(d,list):
        return [dict_to_namespace(x) for x in d]
    else:
        return d
        
def load_config_yamls(path):
    config_path = {}
    for dirpath,dirnames,filenames in os.walk(path):
        for files in filenames:
            name = os.path.splitext(files)[0]
            with open(os.path.join(path,files),"r") as f:
                data=yaml.safe_load(f)
                config_path[name]= dict_to_namespace(data)
    return SimpleNamespace(**config_path)
    
    
ROOT_DIR = BASE_DIR = Path(__file__).resolve().parent.parent  # points to maintenance-query-agent/
TARGET_FOLDER = "config"
path = ROOT_DIR / TARGET_FOLDER
settings = load_config_yamls(path)

for section in vars(settings).values():
    if hasattr(section, "paths"):
        for key, value in vars(section.paths).items():
            resolved = Path(value)
            if not resolved.is_absolute():
                resolved = ROOT_DIR / resolved
            setattr(section.paths, key, resolved)
