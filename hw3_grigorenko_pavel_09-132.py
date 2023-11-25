import math
import numpy as np
import re
import tkinter as tk
import matplotlib.pyplot as plt

pointList=[]
bodyCurveList=[]
tailCurveList=[]
root =tk.Tk()
screenWidth = root.winfo_screenwidth()
fig = plt.figure()
image= np.full((int(screenWidth),int(screenWidth),3),[255,255,255],dtype=np.float32)
color=[0,0,0]
class Point:
    x=int
    y=int

    def __str__(self):
        return ("x: "+str(self.x)+" y: "+str(self.y)+" z: "+str(self.z)+" ")
    def __init__(self,x,y):
        self.x=x
        self.y=y

    def  GetX(self):
        return self.x
    def  GetY(self):
        return self.y

    def GetCoords(self):
        return [self.x,self.y]
    def SetX(self, x):
        self.x=x
    def SetY(self, y):
        self.y = y

    def SetCoords(self,coords):
        self.x=coords[0]
        self.y=coords[1]
    def SetCoords(self,x,y):
        self.x = x
        self.y = y

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

def DrawBezierCurveSegment(x0,y0,x1,y1,x2,y2,color):
    sx = x2-x1
    sy = y2-y1
    xx = x0-x1
    yy = y0-y1
    xy = 0
    dx = 0.0
    dy = 0.0
    err = 0.0
    cur = xx*sy-yy*sx
    assert (xx * sx <= 0 and yy * sy <= 0)
    if xx**2+sy**2>xx**2-yy**2:
        x2=x0
        x0=sx+x1
        y2=y0
        y0=sy+y1
        cur=-cur
    if cur!=0:
        xx+=sx
        if (x0 < x2):
            sx = 1
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
        xy = 2 * xx * yy
        xx**2
        yy**2
        if cur*sx*sy<0:
            xx=-xx
            yy=-yy
            xy=-xy
            cur=-cur
        dx=4.0*sy*cur*(x1-x0)+xx-xy
        dy=4.0*sx*cur*(y0-y1)+yy-xy
        xx+=xx
        yy+=yy
        err=dx+dy+xy
        while True:
            image[int(x0),int(y0)]=color
            if(x0==x2 and y0==y2):
                return
            y1=2*err<dx
            if(2*err>dy):
                x0+=sx
                dx-=xy
                dy+=yy
                err+=dy
            if y1:
                y0+=sy
                dy-=xy
                dx+=xx
                err+=dx
            if dy>=dx:
                break
    BresenhamAlt(x0,y0,x2,y2)




def DrawBezierCurve(i,color):
    x0 = i.GetPoints()[0].GetX()
    x1 = i.GetPoints()[1].GetX()
    x2 = i.GetPoints()[2].GetX()
    y0 = i.GetPoints()[0].GetY()
    y1 = i.GetPoints()[1].GetY()
    y2 = i.GetPoints()[2].GetY()
    x=x0-x1
    y=y0-y1
    t=x0-2*x1+x2
    r=0.0
    if(x*(x2-x1)>0):
        if(y*(y2-y1)>0):
            if(math.fabs((y0-2*y1+y2)/t*x) > abs(y)):
                x0=x2
                x2=x+x1
                y0=y2
                y2=y+y1
        t=(x0-x1)/t
        r=(1-t)*((1-t)*y0+2.0*t*y1)+t*t*y2
        t = (x0 * x2 - x1 * x1) * t / (x0 - x1)
        x = math.floor(t + 0.5)
        y = math.floor(r + 0.5)
        r = (y1 - y0) * (t - x0) / (x1 - x0) + y0
        DrawBezierCurveSegment(x0, y0, x, math.floor(r + 0.5), x, y,color)
        r = (y1 - y2) * (t - x2) / (x1 - x2) + y2
        x1=x
        x0=x1
        y0=y
        y1=math.floor(r+0.5)
    if (y0-y1)*(y2-y1) > 0:
        t = y0 - 2 * y1 + y2
        t = (y0 - y1) / t
        r = (1 - t) * ((1 - t) * x0 + 2.0 * t * x1) + t * t * x2
        t = (y0 * y2 - y1 * y1) * t / (y0 - y1)
        x = math.floor(r + 0.5)
        y = math.floor(t + 0.5)
        r = (x1 - x0) * (t - y0) / (y1 - y0) + x0
        DrawBezierCurveSegment(x0, y0, math.floor(r + 0.5), y, x, y,color)
        r = (x1 - x2) * (t - y2) / (y1 - y2) + x2
        x0 = x
        x1 = math.floor(r + 0.5)
        y1=y
        y0=y1
    DrawBezierCurveSegment(x0, y0, x1, y1, x2, y2,color)

# def DrawBody():
#     for i in bodyCurveList:

bc1=BezierCurve(Point(479,0),Point(100,400),Point(400,500))
print(bc1.GetPoints()[0].GetX())
DrawBezierCurve(bc1,color)

im=plt.imshow(image.astype('uint8'))

plt.show()





