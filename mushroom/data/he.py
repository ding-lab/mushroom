import tifffile
from tifffile import TiffFile

def get_size(filepath):
    tif = TiffFile(filepath)
    p = next(iter(tif.pages))
    return p.shape

def read_he(filepath):
    ext = filepath.split('.')[-1]
    if ext == 'tif':
        return tifffile.imread(filepath)
    elif ext == 'svs':
        raise RuntimeError('reading .svs not implemented yet')
    else:
        raise RuntimeError(f'Extension {ext} not supported for H&E')
