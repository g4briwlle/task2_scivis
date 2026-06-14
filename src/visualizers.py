""" Functions to create visualizations to html
"""
from data_loader import *
from utils import *

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import pyvista as pv
import xarray as xr

import sys
from unittest.mock import MagicMock

# Criando bibliotecas "fantasmas" para enganar o import
sys.modules['esmpy'] = MagicMock()
sys.modules['ESMF'] = MagicMock()
sys.modules['xesmf'] = MagicMock()
#sys.modules['xmitgcm'] = MagicMock()
#sys.modules['xgcm'] = MagicMock()

# cubedsphere vai passar direto pelo erro
import cubedsphere as cs


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


def viz_ocean_3d_slices(data, variable_name: str, x: list[int], y: list[int], z: list[int]):
  print('Opening ocean lat/lon file')
  latlon_grid = xr.open_dataset('llc2160_latlon.nc')

  lon_2d = latlon_grid['longitude'].values # matrix 2D
  lat_2d = latlon_grid['latitude'].values

  print('making cuts')
  # setting cutting points
  x_start, x_end = x[0], x[1]
  y_start, y_end = y[0], y[1]

  # getting wanted cut for lat/lon
  lon_cut = lon_2d[y_start:y_end, x_start:x_end]
  lat_cut = lat_2d[y_start:y_end, x_start:x_end]

  # getting data dimensions
  nz, ny, nx = data.shape

  print('creating rc file')
  # creating depth in center of cels, "RC" missing file
  n_levels = 90
  total_depth = 7000
  dz_min = 1.0                
  dz_max = 480.0
  k = np.linspace(-2.0, 2.5, n_levels)
  dz = dz_min + (dz_max - dz_min) * (0.5 * (1 + np.tanh(k)))

  fator_correcao = total_depth / np.sum(dz)
  dz = dz * fator_correcao

  z_interfaces = np.zeros(n_levels + 1)
  print("starting z interfaces")
  for i in range(n_levels):
      z_interfaces[i+1] = z_interfaces[i] - dz[i]

  Z_1d_real = z_interfaces[:-1] - (dz / 2)

  print(f"Espessura da 1ª camada (dz): {dz[0]:.2f} m")
  print(f"Profundidade do centro da 1ª camada: {Z_1d_real[0]:.2f} m")

  print(f"\nEspessura da última camada (dz): {dz[-1]:.2f} m")
  print(f"Profundidade do centro da última camada: {Z_1d_real[-1]:.2f} m")
  print(f"Fundo do oceano real (última interface): {z_interfaces[-1]:.2f} m")


  ### getting to 3d from 2d
  # building coordenate matrices
  X_3d = np.zeros((nz, ny, nx))
  Y_3d = np.zeros((nz, ny, nx))
  Z_3d = np.zeros((nz, ny, nx))

  # iter in depth layers
  for i in range(nz):
    X_3d[i, :, :] = lon_cut           
    Y_3d[i, :, :] = lat_cut
    Z_3d[i, :, :] = Z_1d_real[i]

  print("initializing pyvista grid")
  grid = pv.StructuredGrid(X_3d, Y_3d, Z_3d)
  grid[variable_name] = data.flatten(order='F')


  pv.set_jupyter_backend('trame')
  plotter = pv.Plotter(notebook=True)


  plotter.add_mesh_slice_orthogonal(
    grid, 
    scalars=variable_name, 
    cmap="coolwarm",
    show_edges=False
)
  plotter.add_axes()
  plotter.set_scale(zscale=0.1)
  plotter.show()

  plotter.export_html('visualizacao_3d.html')

if __name__ == '__main__':
  theta = get_ocean_var('theta', time=0, quality=0, x=[0, 400], y=[2000, 2300], z=[0, 9], to_print=True)
  viz_ocean_3d_slices(theta, 'theta', x=[0,400], y=[2000, 2300], z=[0,9])
