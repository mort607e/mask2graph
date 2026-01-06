import rasterio
from typing import Tuple, Any
import numpy as np
from affine import Affine
import rasterio.crs

def open_tif(tif_path: str) -> Tuple[np.ndarray, Any, Any, int, int]:
    """
    Opens a TIFF file and extracts its imagery data, CRS, transform, width, and height.

    Returns:
    - Tuple containing:
        - bands: numpy array representing the imagery data.
        - crs: coordinate reference system of the TIFF file.
        - transform: affine transform of the TIFF file.
        - width: int, the width of the TIFF file.
        - height: int, the height of the TIFF file.
    """
    with rasterio.open(tif_path) as src:
        bands = src.read()
        crs = src.crs
        transform = src.transform
        width = src.width
        height = src.height

    return (bands, crs, transform, width, height)


if __name__ == '__main__':
    
    tif_path = r"data\sample\Predicted_Mask_Bramaputra_2020-03-02.tif"
    
    result = open_tif(tif_path)
    bands: np.ndarray = result[0]
    crs: rasterio.crs.CRS = result[1]
    transform: Affine = result[2]
    width: int = result[3]
    height: int = result[4]

    print(f"bands shape {bands.shape} type: {type(bands)}")
    print(f"crs: {crs}, type: {type(crs)}")
    print(f"transform type: {type(transform)}")
    print(f"width: {width}, type: {type(width)}")
    print(f"height: {height}, type: {type(height)}")
    
    