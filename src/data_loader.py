''' Functions to load variables and default grids
'''

import numpy as np
import OpenVisus as ov


def get_geos_var(
    var: str,
    x: list[int],
    y: list[int],
    face: int,
    time: int,
    z: list[int],
    quality: int,
    to_print: bool,
) -> np.ndarray:
    """Loads and extracts from DYAMOND GEOS dataset.

    This function connects remotely via a formatted URL using OpenVisus,
    optionally prints the dataset metadata and reads a subset of the data.

    Args:
        var: The name of the climate variable to retrieve (e.g., 'u', 'v', 'temp').
        x: A list containing the bounding box range for the X-axis, typically [start, end].
        y: A list containing the bounding box range for the Y-axis, typically [start, end].
        face: The specific face of the cubed-sphere grid to fetch data from.
        time: The timestep index to retrieve.
        z: A list containing the bounding box range or indices for the Z-axis (depth/altitude).
        quality: The OpenVisus resolution quality level (lower means lower resolution/faster fetch).
        to_print: If True, prints metadata about the loaded dataset to the console.

    Returns:
        A NumPy array containing the extracted dataset slice with its single-entry
        first dimension removed.
    """    
    # load dataset with url
    url = f"https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/GEOS/GEOS_{var.upper()}/{var.lower()}_face_{face}_depth_52_time_0_10269.idx"
    db = ov.LoadDataset(url)

    if to_print==True:
        print(f'Variable: {var}')
        print(f'Dimensions: {db.getLogicBox()[1][0]}*{db.getLogicBox()[1][1]}*{db.getLogicBox()[1][2]}')
        print(f'Total Timesteps: {len(db.getTimesteps())}')
        print(f'Field: {db.getField().name}')
    print(f"variables conected successfully for face {face} and time {time}! max resolution: {db.getMaxResolution()}")

    # reading and preping dataset
    data = db.read(x=x, y=y, time=float(time), quality=quality, z=z)

    return data


def get_ocean_var(
    variable: str,
    time: int,
    quality: int,
    x: list[int],
    y: list[int],
    z: list[int],
    to_print: bool,
) -> np.ndarray:
    """Loads and extracts DYAMOND OCEAN from the MITgcm LLC2160 dataset.

    Connects to the dataset using OpenVisus, optionally prints metadata, and reads the
    requested slicing of the multi-dimensional dataset.

    Args:
        variable: The ocean variable name (e.g., 'theta', 'w', 'u').
        time: The timestep index to retrieve.
        quality: The OpenVisus resolution quality level.
        x: A list containing the range for the X-axis, typically [start, end].
        y: A list containing the range for the Y-axis, typically [start, end].
        z: A list containing the range for the Z-axis (depth), typically [start, end].
        to_print: If True, prints metadata about the loaded dataset to the console.

    Returns:
        A NumPy array containing the requested ocean simulation data slice.
    """
    base_url= "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/"

    if variable=="theta" or variable=="w":
        base_dir=f"mit_output/llc2160_{variable}/llc2160_{variable}.idx"
    elif variable=="u":
        base_dir= "mit_output/llc2160_arco/visus.idx"
    else:
        base_dir=f"mit_output/llc2160_{variable}/{variable}_llc2160_x_y_depth.idx"

    field= base_url+base_dir

    db=ov.LoadDataset(field)
    if to_print == True:
        print(f'Dimensions: {db.getLogicBox()[1][0]}*{db.getLogicBox()[1][1]}*{db.getLogicBox()[1][2]}')
        print(f'Total Timesteps: {len(db.getTimesteps())}')
        print(f'Field: {db.getField().name}')
        print('Data Type: float32')

    data = db.read(time=time, x=x, y=y, quality=quality, z=z)
    print(f'dataset shape: {data.shape}')
    return data


if __name__ == '__main__':
  u_data = get_geos_var('u', x=[0, 100], y=[0, 100], face=0, time=0, z=[0,2], quality=0, to_print=True)
  u_data = get_ocean_var('u', x=[0, 100], y=[0, 100], time=0, z=[0,2], quality=0, to_print=True)
