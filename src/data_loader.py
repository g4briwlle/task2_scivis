''' Functions to load variables and default grids
'''

import numpy as np
import pandas as pd
import OpenVisus as ov
import cartopy.feature as cfeature
#import cdms2
#import vcs
#import os
import xarray as xr


def get_geos_vars(
    var: str,
    x: list[int],
    y: list[int],
    face: int,
    time: int,
    z_level: list[int],
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
        z_level: A list containing the bounding box range or indices for the Z-axis (depth/altitude).
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
    data = db.read(x=x, y=y, time=float(time), quality=quality, z=z_level)
    # transform (I, Z, Y, X) in (Z, Y, X) - get rid of time dimension
    data = data.squeeze(axis=0)

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
    return data


def generate_grid_geos_face(face=0, Ny=1440, Nx=1440):
  """
    Generates a latitude and longitude grid for a specific face of a cubed-sphere projection.
    
    The function creates a 2D grid, projects it onto a 3D cube face using gnomonic 
    projection, rotates the coordinates to match the specified global region, and 
    projects them back onto a sphere to calculate the geographic coordinates.

    Args:
        face (int, optional): The cubed-sphere face index (0 to 5) representing 
            different global regions. Defaults to 0.
            - 0: Equator 4 (Americas)
            - 1: Equator 1 (Africa/Europe)
            - 2: North Pole
            - 3: Equator 2 (Asia/Indian Ocean)
            - 4: Equator 3 (Pacific Ocean)
            - 5: South Pole
        Ny (int, optional): Number of grid points along the Y-axis. Defaults to 1440.
        Nx (int, optional): Number of grid points along the X-axis. Defaults to 1440.

    Returns:
        tuple: A tuple containing:
            - lon (numpy.ndarray): 2D array of longitudes in degrees.
            - lat (numpy.ndarray): 2D array of latitudes in degrees.
            - center_lon (float): Longitude of the face center.
            - center_lat (float): Latitude of the face center.

    Raises:
        ValueError: If the `face` parameter is not an integer between 0 and 5.
    """

  # create face grid
  i = np.arange(Nx)
  j = np.arange(Ny)
  I, J = np.meshgrid(i, j)

  alpha = (np.pi / 4) * (2*I - Nx) / Nx
  beta = (np.pi / 4) * (2*J - Ny) / Ny

  X_base = np.ones_like(alpha)
  Y_base = np.tan(alpha)
  Z_base = np.tan(beta)

  # rotates cube according to wanted face
  if face==0: # Equator 4 (Américas)
    X, Y, Z = Y_base, -X_base, Z_base
    center_lon, center_lat = -90.0, 0.0
  elif face==1: # Equator 1 (África/Europa)
    X, Y, Z = X_base, Y_base, Z_base
    center_lon, center_lat = 0.0, 0.0
  elif face==2: # North Pole (Polo Norte)
    X, Y, Z = -Z_base, Y_base, X_base
    center_lon, center_lat = 0.0, 90.0
  elif face==3: # Equator 2 (Ásia/Índico)
    X, Y, Z = -Y_base, X_base, Z_base
    center_lon, center_lat = 90.0, 0.0
  elif face==4: # Equator 3 (Pacífico)
    X, Y, Z = -X_base, -Y_base, Z_base
    center_lon, center_lat = 180.0, 0.0
  elif face==5: # South Pole (Polo Sul)
    X, Y, Z = Z_base, Y_base, -X_base
    center_lon, center_lat = 0.0, -90.0
  else:
    raise ValueError("Face must be between 0 and 5")

  # project rotaded points into the sphere to get coordinates
  R = np.sqrt(X**2 + Y**2 + Z**2)
  lat = np.degrees(np.arcsin(Z/R))
  lon = np.degrees(np.arctan2(Y, X))

  return lon, lat, center_lon, center_lat


def create_3d_coastlines(R_sphere: int):
  """
    Extracts global coastlines and maps them onto a 3D spherical mesh.

    This function uses Cartopy to fetch 2D geographical coastline geometries 
    and transforms them into 3D Cartesian coordinates ($X, Y, Z$). It includes a 
    radial offset to prevent "z-fighting" (visual clipping) against the main 
    spherical mesh.

    Args:
        R_sphere (float): The base radius of the 3D sphere upon which the 
            coastlines will be projected.

    Returns:
        pyvista.PolyData: A PyVista 3D object containing the coordinates and 
            line connectivity of the global coastlines.
    """
  
  # Adds infinitesimal in radius so coastlines don't soak into the sphere face
  R = R_sphere + (0.001 * R_sphere)

  coastlines = cfeature.COASTLINE.geometries()

  points = []
  lines_connectivity = []
  offset = 0

  for geom in coastlines:
    if geom.geom_type == 'MultiLineString':
      segments = list(geom.geometries)
    else: # LineString
      segments = [geom]

      for seg in segments:
        coords = np.array(seg.coords)
        lat = coords[:, 1]
        lon = coords[:, 0]

        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        Z = R * np.sin(lat_rad)
        X = R * np.cos(lat_rad) * np.cos(lon_rad)
        Y = R * np.cos(lat_rad) * np.sin(lon_rad)

        seg_points = np.column_stack((X,Y,Z))
        points.append(seg_points)

        # connecting points in VTK format
        n_pts = len(seg_points)
        lines_connectivity.append([n_pts] + list(range(offset, offset + n_pts)))
        offset += n_pts

  points_array = np.vstack(points)
  lines_array = np.hstack(lines_connectivity)

  # retuns 3d object with the lines
  return pv.PolyData(points_array, lines=lines_array)
