""" Functions to create visualizations to html
"""

import numpy as np
import pandas as pd
import OpenVisus as ov
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyvista as pv
import trame
import xarray as xr
import cubedsphere as cs

from utils import *

def viz_qv_wind_face_grid(qv, u, v, face, time):

  plt.figure(figsize=(8, 6))

  # BACKGROUND - humidity
  # using imshow since it's faster than counterf for big matrices
  humidity_map = plt.imshow(qv, cmap='YlGnBu', origin='lower') # origin is the south
  # add color bar
  plt.colorbar(humidity_map, label='specific humidity (QV)', fraction=0.046, pad=0.04)

  # OVERLAPPING - winds
  # defining pixel step to draw arrows
  step = 120
  new_size = 1440 // step

  # mean of blocks
  u_mean = u.reshape(new_size, step, new_size, step).mean(axis=(1,3))
  v_mean = v.reshape(new_size, step, new_size, step).mean(axis=(1,3))

  # mesh grid to position arrows

  # Offset by step//2 places the arrow in the middle of the 120x120 block you averaged
  offset = step // 2
  x_indices = np.arange(offset, 1440, step)
  y_indices = np.arange(offset, 1440, step)

  # np.meshgrid is safer than mgrid for generating X, Y coordinate pairs for quiver
  x_arrows, y_arrows = np.meshgrid(x_indices, y_indices)

  # A purely eastward wind (u>0, v=0) should point RIGHT
  # A purely northward wind (u=0, v>0) should point UP
  plt.quiver([720], [720], [1], [0], color='red', scale=5)   # should point right
  plt.quiver([720], [720], [0], [1], color='blue', scale=5)  # should point up

  # draw arrows using quiver
  arrows = plt.quiver(x_arrows, y_arrows, u_mean, v_mean, color='black', alpha=0.6)

  plt.title(f'Humidity and wind map (Face {face} Time {time})', fontsize=14)

  plt.tight_layout()
  plt.show()


def viz_qv_wind_sphere_face(qv, u, v, face, time):
  lon, lat, center_lon, center_lat = generate_grid_geos_face(face)

  fig = plt.figure(figsize=(8,6))
  ax = plt.axes(projection=ccrs.Orthographic(
      central_longitude=center_lon,
      central_latitude=center_lat
  ))

  # pcolormesh understands transform= so it maps data coords → projection
  humidity_map = ax.pcolormesh(
      lon, lat, qv,
      cmap='YlGnBu',
      transform=ccrs.PlateCarree()   # tells Cartopy lon/lat are in PlateCarree
  )

  # defining pixel step to draw arrows
  step = 120
  new_size = 1440 // step

  # mean of blocks
  u_mean = u.reshape(new_size, step, new_size, step).mean(axis=(1,3))
  v_mean = v.reshape(new_size, step, new_size, step).mean(axis=(1,3))

  # mesh grid to position arrows

  # Offset by step//2 places the arrow in the middle of the 120x120 block you averaged
  offset = step // 2
  x_indices = np.arange(offset, 1440, step)
  y_indices = np.arange(offset, 1440, step)

  # Instead of a meshgrid of pixel numbers,
  # index into lon/lat at those exact pixels
  # np.ix_ creates the right indexing shape for 2D arrays
  arrow_lons = lon[np.ix_(y_indices, x_indices)]  # shape (12, 12)
  arrow_lats = lat[np.ix_(y_indices, x_indices)]  # shape (12, 12)

  ax.quiver(
      arrow_lons, arrow_lats,   # geographic position (replaces x_arrows, y_arrows)
      u_mean, v_mean,
      transform=ccrs.PlateCarree(),
      color='black', alpha=0.6
  )

  # coastlines now align automatically
  ax.coastlines(resolution='50m', color='black', linewidth=1, alpha=0.7)
  ax.gridlines(draw_labels=True)
  plt.colorbar(humidity_map, label='Specific Humidity (QV)', fraction=0.046, pad=0.04)
  plt.title(f'Humidity and wind map (Face {face} Time {time})', fontsize=14)
  plt.show()


def viz_set_qv_wind_sphere(datasets, n_cols, figsize_per_cell=(5,3)):
  """
    datasets: lista de dicts com chaves:
        - 'qv': array 2D
        - 'u':  array 2D
        - 'v':  array 2D
        - 'face': int
        - 'time': int (0 to 10000)
    ncols: número de colunas no grid
    figsize_per_cell: tamanho (largura, altura) de cada célula
    """

  n = len(datasets)
  n_rows = n // n_cols

  fig_w = figsize_per_cell[0] * n_cols
  fig_h = figsize_per_cell[1] * n_rows
  fig = plt.figure(figsize=(fig_w, fig_h))

  # Calcula vmin/vmax globais para colorbar consistente
  all_qv = np.concatenate([d['qv'].ravel() for d in datasets])
  vmin, vmax = np.nanmin(all_qv), np.nanmax(all_qv)

  axes = []
  for i, ds in enumerate(datasets):
    face = ds['face']
    time = ds['time']
    qv   = ds['qv'].copy()
    u    = ds['u'].copy()
    v    = ds['v'].copy()

    if face >= 3:
      qv = qv.T
      u = u.T
      v = v.T

    # Cria grid dedicado para a Umidade (qv)
    Ny_qv, Nx_qv = qv.shape
    lon_qv, lat_qv, center_lon, center_lat = generate_grid_geos_face(face, Ny=Ny_qv, Nx=Nx_qv)

    # Cria grid dedicado para os Ventos (u, v)
    Ny_w, Nx_w = u.shape
    lon_w, lat_w, _, _ = generate_grid_geos_face(face, Ny=Ny_w, Nx=Nx_w)

    ax = fig.add_subplot(
      n_rows, n_cols, i + 1,
      projection=ccrs.Orthographic(
        central_longitude=center_lon,
        central_latitude=center_lat
      )
    )
    axes.append(ax)

    humidity_map = ax.pcolormesh(
      lon_qv, lat_qv, qv,
      cmap='YlGnBu',
      vmin=vmin, vmax=vmax,
      transform=ccrs.PlateCarree()
    )

    # Define o pulo do fatiamento garantindo aprox. 12 setas por eixo, sem dar erro de reshape
    step_y = max(1, Ny_w // 12)
    step_x = max(1, Nx_w // 12)

    # Subamostra (fatia) os arrays para não poluir o gráfico
    u_sub = u[::step_y, ::step_x]
    v_sub = v[::step_y, ::step_x]
    arrow_lons = lon_w[::step_y, ::step_x]
    arrow_lats = lat_w[::step_y, ::step_x]

    # Plota os ventos usando o grid de Ventos
    ax.quiver(
      arrow_lons, arrow_lats,
      u_sub, v_sub,
      transform=ccrs.PlateCarree(),
      color='black', alpha=0.6
    )


    ax.coastlines(resolution='50m', color='black', linewidth=0.8, alpha=0.7)
    ax.gridlines()
    ax.set_title(f'Face {face} | Time {time}', fontsize=10)

  # Esconde subplots vazios (se n não for múltiplo de ncols)
  for j in range(i + 1, n_rows * n_cols):
      fig.add_subplot(n_rows, n_cols, j + 1).set_visible(False)

  # Colorbar única compartilhada
  cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
  sm = plt.cm.ScalarMappable(cmap='YlGnBu', norm=plt.Normalize(vmin=vmin, vmax=vmax))
  sm.set_array([])
  fig.colorbar(sm, cax=cbar_ax, label='Specific Humidity (QV)')

  fig.suptitle('Humidity and Wind Maps', fontsize=14, y=1.01)
  plt.tight_layout()
  plt.show()


def viz_qv_3d(qv, R_sphere, lon, lat, face, time):
  ### grid transformation: from (lat,lon) to (X,Y,Z)

  # turn degrees in radians
  lat_rad = np.radians(lat)
  lon_rad = np.radians(lon)

  # get coordinates
  r_xy = R_sphere * np.cos(lat_rad)
  Z = R_sphere * np.sin(lat_rad)
  X = r_xy * np.cos(lon_rad)
  Y = r_xy * np.sin(lon_rad)

  ### initialize VTK grid
  grid = pv.StructuredGrid(X, Y, Z)

  # attach data to the grid
  # vtk expects 1d flatten arrays for point data
  grid.point_data['qv'] = qv.T.flatten(order='C')

  # set up interactive plotter
  pv.set_jupyter_backend('html')
  plotter = pv.Plotter(notebook=True)
  plotter.add_mesh(
      grid,
      scalars= 'qv',
      cmap='YlGnBu',
      show_edges=False
  )

  coast_mesh = create_3d_coastlines(R_sphere)
  plotter.add_mesh(coast_mesh, color='black', line_width=2.5)

  plotter.add_text(f"DYAMOND QV - Face {face}, Time {time}", font_size=12)
  plotter.show()
