# importing libraries
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
import plotly.graph_objs as go
import plotly

def visualize(x, y, z):
    trace = go.Scatter3d(
       x = x, y = y, z = z,mode = 'markers', marker = dict(
          size = 2,
          color = z, # set color to an array/list of desired values
          colorscale = 'Viridis'
          )
       )
    layout = go.Layout(title = '3D Scatter plot')
    fig = go.Figure(data = [trace], layout = layout)
    plotly.offline.iplot(fig)
	
def keep_only_surface_points(x, y, z):
    xmin = min(x)
    ymin = min(y)
    xr = max(x)-min(x)+1
    yr = max(y)-min(y)+1
    relief = make_relief(x,y,z)

    xnew = np.ndarray(0, np.int16)
    ynew = np.ndarray(0, np.int16)
    znew = np.ndarray(0)
    
    for j in range(yr):
        for i in range(xr):
            xnew = np.append(xnew, i+xmin) #cm
            ynew = np.append(ynew, j+ymin) #cm
            znew = np.append(znew, relief[j,i])
    return xnew, ynew, znew 

def make_relief(x, y, z):
    xmin = min(x)
    ymin = min(y)
    xr = max(x)-min(x)+1
    yr = max(y)-min(y)+1
    relief = np.zeros((yr,xr)) - 32768
    for i in range(len(x)):
        if z[i] > relief[y[i]-ymin,x[i]-xmin]:
            relief[y[i]-ymin,x[i]-xmin] = z[i]
    relief[relief==-32768] = - 15
    return relief

def mprod(M, v):
    q = [0, 0, 0, 0]
    q[0] = M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2] + M[0][3]*v[3]
    q[1] = M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2] + M[1][3]*v[3]
    q[2] = M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2] + M[2][3]*v[3]
    q[3] = M[3][0]*v[0] + M[3][1]*v[1] + M[3][2]*v[2] + M[3][3]*v[3]
    return q

def update(xold, yold, zold, xnew, ynew, znew, qx, qy, qz, dx,dy,dz):
    qx = qx*math.pi/180
    qy = qy*math.pi/180
    qz = qz*math.pi/180
    # rotation matrices
    My = [[math.cos(qy), 0, math.sin(qy), 0],
          [0, 1, 0, 0],
          [-math.sin(qy), 0, math.cos(qy), 0],
          [0, 0, 0, 1]
         ]
    Mx = [[1, 0, 0, 0],
          [0, math.cos(qx), -math.sin(qx), 0],
          [0, math.sin(qx), math.cos(qx), 0],
          [0, 0, 0, 1]
         ]
    Mz = [[math.cos(qz), math.sin(qz), 0, 0],
          [-math.sin(qz), math.cos(qz), 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1]
         ]
    
    x = np.ndarray(0, np.int16)
    y = np.ndarray(0, np.int16)
    z = np.ndarray(0)
    
    # camera position in robot coordinate system
    x0, y0, z0 = 0, 15, 5
    
    #rotate new data by -qx and -qy
    for i in range(len(xnew)):
        v0 = [xnew[i]+x0, ynew[i]+y0, znew[i]+z0, 1]
        v1 = mprod(My, mprod(Mx, v0))
        x = np.append(x, math.trunc(round(v1[0])))
        y = np.append(y, math.trunc(round(v1[1])))
        z = np.append(z, v1[2])
        
    #rotate old data by -qz
    for i in range(len(xold)):
        v0 = [xold[i], yold[i], zold[i], 1]
        v1 = mprod(Mz, v0)
        x = np.append(x, math.trunc(round(v1[0]))-dx)
        y = np.append(y, math.trunc(round(v1[1]))-dy)
        z = np.append(z, v1[2]-dz)
        
    return x, y, z
	
def scan():
    # opening and downscaling left image
    l = cv.imread('left2.jpg',0)
    x = len(l[0])
    y = len(l)
    k = 16
    l = cv.resize(l, (x//k, y//k))

    # opening and downscaling right image
    r = cv.imread('right2.jpg',0)
    r = cv.resize(r, (x//k, y//k))

    # computing dispartity map
    stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(l,r)

    # visualization of disparity map
    plt.imshow(disparity,'gray')
    plt.show()

    # stereocam parameters
    w = 4.5 #mm (matrix width)
    h = 3.4 #mm (matrix height)
    fn = 5.6 #mm (focal length)
    T = 10 #mm (spacing)

    # middle point
    xmid = x/(2*k) #pixel
    ymid = y/(2*k) #pixel

    # pixel to mm factor
    Kpm = w/x

    # initializing point cloud
    xc = np.ndarray(0, np.int16)
    yc = np.ndarray(0, np.int16)
    zc = np.ndarray(0)

    xraw = np.ndarray(0, np.int16)
    yraw = np.ndarray(0, np.int16)
    zraw = np.ndarray(0)

    Krare = 10

    # computing point cloud
    for j in range(y//k):
        for i in range(x//k):
            d = (disparity[j][i])*Kpm #mm
            if d > 0:
                dx = (i-xmid)*Kpm #mm
                dy = (j-ymid)*Kpm #mm
                Z = T*fn/d #mm
                if Z < 1000:
                    xx = math.trunc(20*dx*Z/fn)
                    yy = math.trunc(20*dy*Z/fn)
                    zz = math.trunc(Z)

                    xc = np.append(xc, xx//Krare) #cm
                    yc = np.append(yc, zz//Krare) #cm
                    zc = np.append(zc, -yy/Krare) #cm
    return xc,yc,zc
	
# main
def main():
  x = np.ndarray(0, np.int16)
  y = np.ndarray(0, np.int16)
  z = np.ndarray(0)

  xn, yn, zn = scan()
  x, y, z = update(x,y,z, xn,yn,zn, 0,0,0, 0,0,0)
  x, y, z = keep_only_surface_points(x, y, z)

  x, y, z = update(x,y,z, xn,yn, zn, 10,20,-90, -70,0,0)
  x, y, z = keep_only_surface_points(x, y, z)

  visualize(x,y,z)

  # making relief for output
  size = 25
  relief = np.zeros((2*size+1,2*size+1))-32768
  for i in range(len(x)):
    if (abs(x[i])<=size) and (abs(y[i])<=size):
      relief[y[i]+size][x[i]+size] = z[i]
  relief[relief==-32768] = -15

  # smoothing
  kernel = np.ones((5,5),np.int16)/25
  relief = cv.filter2D(relief,-1,kernel)

if __name__ == "__main__":
  main()
