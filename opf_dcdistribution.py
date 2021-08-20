
"""
    OPF in DC distribution using SOC
    Alejandro Garces
    18/08/2021
    Version 1.0
"""

import numpy as np
import pandas as pd
import cvxpy as cvx

" Parametros de la red "
# Envio Recibo rkm d pmax
grid = [[1,  2,  0.0053, 0.70,0.0],
        [2,  3,  0.0054, 0.00,0.0],
        [3,  4,  0.0054, 0.36,0.0],
        [4,  5,  0.0063, 0.04,0.0],
        [4,  6,  0.0051, 0.36,0.0],
        [3,  7,  0.0037, 0.00,0.0],
        [7,  8,  0.0079, 0.32,0.0],
        [7,  9,  0.0072, 0.80,1.5],
        [3,  10, 0.0053, 0.00,0.0],
        [10, 11, 0.0038, 0.45,0.0],         
        [11, 12, 0.0079, 0.68,2.5],
        [11, 13, 0.0078, 0.10,0.0],
        [10, 14, 0.0083, 0.00,0.0],
        [14, 15, 0.0065, 0.22,0.0],
        [15, 16, 0.0064, 0.23,0.0],
        [16, 17, 0.0074, 0.43,0.0],
        [16, 18, 0.0081, 0.34,2.5],
        [14, 19, 0.0078, 0.09,0.0],
        [19, 20, 0.0084, 0.21,0.0],
        [19, 21, 0.0082, 0.21,2.0]]

"Crear diccionario con: (envio,recibo): gkm, demanda, pmax, id_gen(si pm>0)"
num_lines = len(grid)
num_nodes = num_lines + 1
num_gen = 0
feeder = {}
for k in range(num_lines):
  n1 = grid[k][0]-1
  n2 = grid[k][1]-1
  gkm = 1/grid[k][2]
  dm = grid[k][3]
  pmax = grid[k][4] 
  if pmax>0: 
     feeder[(n1,n2)] = (k,gkm,dm,(pmax,num_gen))
     num_gen = num_gen+1
  else: feeder[(n1,n2)] = (k,gkm,dm,'None')

"Modelo SOC"
u = cvx.Variable(num_nodes)       # v^2
s = cvx.Variable(num_gen)         # potencia generada
p_from = cvx.Variable(num_lines)  # potencia en las lineas direccion km
p_to   = cvx.Variable(num_lines)  # potencia en las lineas direccion mk
w = cvx.Variable(num_lines)       # variable auxiliar vk*vm
EqN = num_nodes*[0]
for (k,m) in feeder:
  line,gkm,dm,gen = feeder[(k,m)]
  EqN[m] = EqN[m] + p_to[line]
  EqN[k] = EqN[k] - p_from[line]

res = [u[0] == 1.0]
for (k,m) in feeder:
  line,gkm,dm,gen = feeder[(k,m)]
  res += [cvx.SOC(u[k]+u[m],cvx.vstack([2*w[line],u[k]-u[m]]))]
  res += [gkm*w[line] ==  p_to[line] + gkm*u[m]]
  res += [gkm*u[k]    ==  p_from[line] + gkm*w[line]]
  res += [u[m] >= 0.95**2]
  res += [u[m] <= 1.05**2]
  if gen=='None':
    res += [EqN[m] == dm]
  else:
    pmax,k_gen = gen
    res += [EqN[m] == dm - s[k_gen]]
    res += [s[k_gen] >= 0]
    res += [s[k_gen] <= pmax]    
  
obj = cvx.Minimize(cvx.sum(p_from)-cvx.sum(p_to))    
OPFSOC = cvx.Problem(obj,res)
OPFSOC.solve(solver=cvx.SCS)
print('Load-flow in DC grids using second-order cone optimization')
print('Power loss',obj.value,OPFSOC.status)
print('Generation', s.value)

"Flujo de carga"
g_bus = np.zeros((num_nodes,num_nodes))
d_bus = np.zeros(num_nodes)
id_gen = {}
for (k,m) in feeder:
  line,gkm,dm,gen = feeder[(k,m)]
  g_bus[k,k] = g_bus[k,k] + gkm
  g_bus[k,m] = g_bus[k,m] - gkm
  g_bus[m,k] = g_bus[m,k] - gkm
  g_bus[m,m] = g_bus[m,m] + gkm
  d_bus[m] = dm
  if not(gen=='None'):
    id_gen[m] = gen[1]
g_N0 = g_bus[1:num_nodes,0]
g_NN = g_bus[1:num_nodes,1:num_nodes]
z_NN = np.linalg.inv(g_NN)
def load_flow(p_gen):
  v0 = 1
  p_node = -d_bus
  for m in id_gen:
    p_node[m] = p_node[m] + p_gen[id_gen[m]]
  v_node = np.ones(num_nodes)*v0
  i_node = p_node[1:num_nodes]/v_node[1:num_nodes]
  err = np.linalg.norm(i_node - g_N0*v0 - g_NN@v_node[1:num_nodes])
  while err > 1E-10:    
    v_node[1:num_nodes] = z_NN@(i_node - g_N0*v0) 
    i_node = p_node[1:num_nodes]/v_node[1:num_nodes]    
    err = np.linalg.norm(i_node - g_N0*v0 - g_NN@v_node[1:num_nodes])
  p_loss = v_node.T@g_bus@v_node
  i_line = np.zeros(num_lines)
  for (k,m) in feeder:
     line,gkm,dm,gen = feeder[(k,m)]
     i_line[line] = gkm*(v_node[k]-v_node[m])
  return p_loss, err, v_node, i_line
p_loss, err, v_node, i_line = load_flow([1.5, 1.35, 1.08, 0.7])  
i_node = g_bus@v_node
resultados = pd.DataFrame()
resultados['Voltage'] = v_node
resultados['Load'] = d_bus
resultados['Generation'] = np.round(v_node*i_node,4)+d_bus
print(resultados)
