import math
import numpy as np
import re
import tkinter as tk
import matplotlib.pyplot as plt

pointList=[]
bodyCurveList=[]
tailCurveList=[]
root =tk.Tk()
screenWidth = root.winfo_screenwidth()/3
fig = plt.figure()
image= np.zeros((int(screenWidth),int(screenWidth),3),dtype=np.float32)
color=[0,0,0]
class Point:
    x=int
    y=int
    z=int
    def __str__(self):
        return ("x: "+str(self.x)+" y: "+str(self.y)+" z: "+str(self.z)+" ")
    def __init__(self,x,y,z):
        self.x=x
        self.y=y
        self.z=z
    def  GetX(self):
        return self.x
    def  GetY(self):
        return self.y
    def  GetZ(self):
        return self.z
    def GetCoords(self):
        return [self.x,self.y,self.z]
    def SetX(self, x):
        self.x=x
    def SetY(self, y):
        self.y = y
    def SetZ(self, z):
        self.z = z
    def SetCoords(self,coords):
        self.x=coords[0]
        self.y=coords[1]
        self.z=coords[2]
    def SetCoords(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z

class BezierCurve:
    point0=Point
    point1=Point
    point2=Point

    def __init__(self,point1,point2,point3):
        self.point1 = point1
        self.point2 = point2
        self.point3 = point3
    def GetPoints(self):
        return [self.point1,self.point2,self.point3]
def BresenhamAlt(x0,y0,x1,y1,color):
    x0 = int(x0)
    y0 = int(y0)
    x1 = int(x1)
    y1 = int(y1)

    deltaX=abs(x1-x0)
    if x0<x1:
        sx=1
    else:
        sx=-1
    deltaY=-abs(y1-y0)
    if y0<y1:
        sy=1
    else:
        sy=-1
    error=deltaX+deltaY
    while True:
        image[int(x0),int(y0)]=color
        if x0==x1 and y0==y1:
            break
        e2 = 2 * error
        if e2 >= deltaY:
            if x0 == x1: break
            error = error + deltaY
            x0 = x0 + sx
        if e2 <= deltaX:
            if y0 == y1: break
            error = error + deltaX
            y0 = y0 + sy
def AddCurve(point0,point1,point2):
    pointList.append(point0)
    pointList.append(point1)
    pointList.append(point2)
    bodyCurveList.append(BezierCurve(point0,point1,point2))
def DrawBezierCurve(i,color):
    x0=i.point0.x
    x1=i.point1.x
    x2 = i.point2.x
    y0 = i.point0.y
    y1 = i.point1.y
    y2 = i.point2.y
    sx = int(x2-x1)
    sy = int(y2-y1)
    xx = int(x0-x1)
    yy = int(y0-y1)
    dx,dy,err,cur =float(xx*sy-yy*sx)
    assert (xx * sx <= 0 and yy * sy <= 0);
    if(sx**2+sy**2>xx**2+yy**2):
        x2=x0
        x0=sx+x1
        y2=y0
        y0 = sy + y1
        cur = -cur
    if(cur!=0):
        xx+= sx
        if(x0 < x2):
            sx=1
            xx *= sx
        else:
            sx = -1
            xx *= sx
        yy += sy
        if (y0 < y2):
            sy = 1
            yy *= sy
        else:
            sy = -1
            yy *= sy
        xy=2*xx*yy
        xx*=xx
        yy*=yy
        if(cur*sx*sy<0):
            xx=-xx
            yy=-yy
            xy=-xy
            cur=-cur
        dx = 4.0 * sy * cur * (x1 - x0) + xx - xy
        dy = 4.0 * sx * cur * (y0 - y1) + yy - xy
        xx+= xx
        yy+= yy
        err=dx+dy+xy
        while True:
            image[int(x0),int(y0)]=color
            if(x0==x2 and y0==y2):
                return
            y1=2*err<dx
            if(2*err>dy):
                x0+=sx
                dx-=xy
                dy += yy
                err+=dy
            if(y1):
                y0+=sy
                dy_=xy
                dx += xx
                err+=dx
            if dy>=dx:
                break
    BresenhamAlt(x0,y0,x2,y2)
def DrawBody():
    for i in bodyCurveList:





