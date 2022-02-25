import plotly.graph_objects as go
import numpy as np

def cut_z (arr, z_min):
  for i in range (arr.shape[0]):
    z = np.array([arr[i][j][2] for j in range(arr[i].shape[0])])
    arr[i] = arr[i][z > z_min]
  return np.array(arr)

def cut_point_num (arr, point_num):
  new_arr = []
  for i in range (arr.shape[0]):
    if (arr[i].shape[0] >= point_num):
        new_arr.append(arr[i][0:point_num])
  return np.array(new_arr)

def draw_pc (pc_arr):

  x = np.array([pc_arr[i][0] for i in range(pc_arr.shape[0])])
  y = np.array([pc_arr[i][1] for i in range(pc_arr.shape[0])])
  z = np.array([pc_arr[i][2] for i in range(pc_arr.shape[0])])

  fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(
      size=2,
      color=z,                # set color to an array/list of desired values
      colorscale='Viridis',   # choose a colorscale
      opacity=0.8
      ))])
  
  fig.show()
