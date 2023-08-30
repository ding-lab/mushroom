from tifffile import TiffFile

def get_size(filepath):
    tif = TiffFile(filepath)
    p = next(iter(tif.pages))
    return p.shape
