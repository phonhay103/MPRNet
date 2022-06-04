# from natsort import natsorted
from pathlib import Path

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    Path(path).mkdir(exist_ok=True)

# def get_last_path(path, session):
# 	x = natsorted(glob(os.path.join(path,'*%s'%session)))[-1]
# 	return x