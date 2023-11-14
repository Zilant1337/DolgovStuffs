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
ScaleX
ScaleY

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
        return ("Треугольник: "+"\n1: "+str(point1)+"\n2: "+str(point2)+"\n3: "+str(point3))
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

            triList.append(tri(pointList[int(tempText[1])-1],pointList[int(tempText[2])-1],pointList[int(tempText[3])-1]))
def ScalePoints():
    # Scale matrix
    # A = (a, 0, 0
    #      0, b, 0
    #      0, 0, 1)
    for i in pointList:
        X=np.matrix([[i.GetX()],[i.Gety()],[1]])
        A=np.matrix([[ScaleX,0,0],[0,ScaleY,0],[0,0,1]])
        scaledX=np.matmul(A,X)
        image[scaledX[0][0], scaledX[1][0]] = yellow




f= open("teapot.obj","r")



text = ReadObjFile(f)
ParsePoints(text)
ParseTris(text)

fig = plt.figure()

yellow=np.array([255,255,0],dtype=np.uint8)


image= np.zeros((int(screenWidth),int(screenWidth),3),dtype=np.float32)

for i in pointList:
    image[i.GetX(),i.GetY()]=yellow

im=plt.imshow(image)

