
import numpy as np
import pandas as pd
import OpenVisus as ov
import cartopy.feature as cfeature
import pyvista as pv


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