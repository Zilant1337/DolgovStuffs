import math
import numpy as np
import re
import tkinter as tk
import matplotlib.pyplot as plt

pointList=[]
triList=[]
scalePointList=[]
root =tk.Tk()
screenWidth = root.winfo_screenwidth()/3
fig = plt.figure()
image= np.zeros((int(screenWidth),int(screenWidth),3),dtype=np.float32)
transformedPointList=[]
class point:
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
class tri:
    point1=point
    point2=point
    point3=point
    def __str__(self):
        return ("Треугольник: "+"\n1: "+str(self.point1)+"\n2: "+str(self.point2)+"\n3: "+str(self.point3))
    def __init__(self,point1,point2,point3):
        self.point1 = point1
        self.point2 = point2
        self.point3 = point3
    def GetPoints(self):
        return [self.point1,self.point2,self.point3]

def ReadObjFile(file):
    text=[]
    text=file.readlines()
    return text
def PrintPoints():
    cnt=0
    for i in pointList:
        cnt += 1
        print(cnt)
        print("x:"+str( i.GetX()))
        print("y:" + str(i.GetY()))
        print("z:" + str(i.GetZ()))
def ParsePoints(text):
    for i in text:
        tempText=i.split(" ")
        if tempText[0]=="v":
            pointList.append(point(float(tempText[1]),float(tempText[2]),float(tempText[3])))
def PrintTriList():
    for i in triList:
        print("1")
        print("x:" + str(i.GetPoints()[0].GetX()))
        print("y:" + str(i.GetPoints()[0].GetY()))
        print("z:" + str(i.GetPoints()[0].GetZ()))
        print("2")
        print("x:" + str(i.GetPoints()[1].GetX()))
        print("y:" + str(i.GetPoints()[1].GetY()))
        print("z:" + str(i.GetPoints()[1].GetZ()))
        print("3")
        print("x:" + str(i.GetPoints()[2].GetX()))
        print("y:" + str(i.GetPoints()[2].GetY()))
        print("z:" + str(i.GetPoints()[2].GetZ()))
def ParseTris(text):

    for i in text:

        tempText=i.split(" ")
        if tempText[0]=="f":

            triList.append(tri(transformedPointList[int(tempText[1])-1],transformedPointList[int(tempText[2])-1],transformedPointList[int(tempText[3])-1]))



def ScalePoints(Scale):
    # Scale matrix
    # A = (a, 0, 0
    #      0, b, 0
    #      0, 0, 1)
    for i in pointList:
        X=np.matrix([[i.GetX()],[i.GetY()],[1]])
        A=np.matrix([[Scale,0,0],[0,Scale,0],[0,0,1]])
        T=np.matrix([[1,0,screenWidth/2-1],[0,1,screenWidth/2-1],[0,0,1]])
        R=np.matrix([[math.cos(math.radians(90)),-1*math.sin(math.radians(90)),0],[math.sin(math.radians(90)),math.cos(math.radians(90)),0],[0,0,1]])
        scaledX=np.matmul(A,X)
        rotatedX=np.matmul(R,scaledX)
        translatedX=np.matmul(T,rotatedX)
        color=[255,int(translatedX[1][0]/screenWidth*255),0]
        image[int(translatedX[0][0]), int(translatedX[1][0])] = color
        transformedPointList.append(point(translatedX[0][0],translatedX[1][0],i.GetZ))


def BresenhamLineLow(x0,y0,x1,y1):
    dx = x1-x0
    dy = y1-y0
    yi=1
    if dy<0:
        yi=-1
        dy=-dy
    D = 2 * dy - dx
    y = y0

    for j in range(int(x0), int(x1)+1):
        color = [255, int(y / screenWidth * 255), 0]
        image[math.ceil(j), math.ceil(y)] = color
        if D > 0:
            y = y + yi
            D = D - 2 * dx
        else:
            D = D + 2 * dy
def BresenhamLineHigh(x0,y0,x1,y1):
    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx
    D = 2 * dx - dy
    x = x0

    for j in range(int(y0), int(y1)+1):
        color = [255, int(j / screenWidth * 255), 0]
        image[int(x), int(j)] = color
        if D > 0:
            x = x + xi
            D = D + (2 * (dx-dy))
        else:
            D = D + 2 * dx
def BresenhamLineUniversal(x0,y0,x1,y1):
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            BresenhamLineLow(x1, y1, x0, y0)
        else:
            BresenhamLineLow(x0, y0, x1, y1)
    else:
        if y0 > y1:
            BresenhamLineHigh(x1, y1, x0, y0)
        else:
            BresenhamLineHigh(x0, y0, x1, y1)

def BresenhamAlt(x0,y0,x1,y1):
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
        color=[255,int(y0/screenWidth*255),0]
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



def BresenhamLineAlgorithm():
    for i in triList:
        # BresenhamLineUniversal(i.point1.GetX(),i.point1.GetY(),i.point2.GetX(),i.point2.GetY())
        # BresenhamLineUniversal(i.point2.GetX(),i.point2.GetY(),i.point3.GetX(),i.point3.GetY())
        # BresenhamLineUniversal(i.point3.GetX(),i.point3.GetY(),i.point1.GetX(),i.point1.GetY())
        BresenhamAlt(i.point1.GetX(), i.point1.GetY(), i.point2.GetX(), i.point2.GetY())
        BresenhamAlt(i.point2.GetX(), i.point2.GetY(), i.point3.GetX(), i.point3.GetY())
        BresenhamAlt(i.point3.GetX(), i.point3.GetY(), i.point1.GetX(), i.point1.GetY())


def GetMaxCoordinates():
    maximum=0
    for i in pointList:
        maximum=max(maximum,max(abs(i.GetX()),abs(i.GetY())))
    return maximum
def GetMinCoords():
    minimum=0
    for i in pointList:
        minimum=min(minimum,min(i.GetX(),i.GetY()))
    return minimum



f= open("teapot.obj","r")
text = ReadObjFile(f)
ParsePoints(text)
Scale=screenWidth/GetMaxCoordinates()/2

ScalePoints(Scale)
ParseTris(text)
BresenhamLineAlgorithm()

im=plt.imshow(image.astype('uint8'))


plt.savefig("teapot.png")

plt.show()
