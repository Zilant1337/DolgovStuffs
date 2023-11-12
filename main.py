import math
import numpy as np
import re
pointList=[]
triList=[]
class tri:
    points=[]
    def __init__(self,points):
        self.points=points.copy()
    def __init__(self,point1,point2,point3):
        self.points.append(point1)
        self.points.append(point2)
        self.points.append(point3)
    def GetPoints(self):
        return self.points
    def CalculateIncirleArea(self):
        a = math.sqrt((self.points[1].GetX() -self.points[0].GetX()) ** 2 + (self.points[1].GetY() - self.points[0].GetY()) ** 2 +(self.points[1].GetZ() - self.points[0].GetZ()) **  2)
        b = math.sqrt((self.points[2].GetX() - self.points[1].GetX()) ** 2 + (self.points[2].GetY() - self.points[1].GetY()) ** 2 + (self.points[2].GetZ() - self.points[1].GetZ()) **  2)
        c = math.sqrt((self.points[0].GetX() - self.points[2].GetX()) ** 2 + (self.points[0].GetY() - self.points[2].GetY()) ** 2 + (self.points[0].GetZ() - self.points[2].GetZ()) **  2)

        # Calculate the semi-perimeter
        s = (a + b + c) / 2

        # Calculate the area of the triangle
        triangle_area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        # Calculate the inradius
        inradius = triangle_area / s

        # Calculate the area of the incircle
        incircle_area = math.pi * inradius ** 2

        return incircle_area

    def CalculateCosine(self,edgeIndex1,edgeIndex2):
        a = math.sqrt((self.points[1].GetX() - self.points[0].GetX()) ** 2 + (
                    self.points[1].GetY() - self.points[0].GetY()) ** 2 + (
                                  self.points[1].GetZ() - self.points[0].GetZ()) ** 2)
        b = math.sqrt((self.points[2].GetX() - self.points[1].GetX()) ** 2 + (
                    self.points[2].GetY() - self.points[1].GetY()) ** 2 + (
                                  self.points[2].GetZ() - self.points[1].GetZ()) ** 2)
        c = math.sqrt((self.points[0].GetX() - self.points[2].GetX()) ** 2 + (
                    self.points[0].GetY() - self.points[2].GetY()) ** 2 + (
                                  self.points[0].GetZ() - self.points[2].GetZ()) ** 2)
        if ((edgeIndex1==1 and edgeIndex2==0) or (edgeIndex1==0 and edgeIndex2==1)):
            scalar=self.points[1].GetX()*self.points[0].GetX()+self.points[1].GetY()*self.points[0].GetY()+self.points[1].GetZ()*self.points[0].GetZ()
            return scalar/(a*b)
        if ((edgeIndex1==2 and edgeIndex2==1) or (edgeIndex1==1 and edgeIndex2==2)):
            scalar=self.points[2].GetX()*self.points[1].GetX()+self.points[2].GetY()*self.points[1].GetY()+self.points[2].GetZ()*self.points[1].GetZ()
            return scalar/(c*b)
        if ((edgeIndex1==2 and edgeIndex2==0) or (edgeIndex1==0 and edgeIndex2==2)):
            scalar=self.points[2].GetX()*self.points[0].GetX()+self.points[2].GetY()*self.points[0].GetY()+self.points[2].GetZ()*self.points[0].GetZ()
            return scalar/(a*c)


    def CalculateMaximumCosine(self):
        return max(self.CalculateCosine(0,1),max(self.CalculateCosine(1,2),self.CalculateCosine(0,2)))



class point:
    x=int
    y=int
    z=int
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

def ReadObjFile(file):
    text=[]
    text=file.readlines()
    return text

def ParsePoints(text):
    for i in text:
        tempText=i.split(" ")
        for j in tempText:
            print(j)
        if tempText[0]=="v":

            pointList.append(point(float(tempText[1]),float(tempText[2]),float(tempText[3])))

def ParseTris(text):
    for i in text:
        tempText=i.split(" ")
        if tempText[0]=="f":
            print(tempText[0])
            print(tempText[1])
            print(tempText[2])
            print(tempText[3])
            triList.append(tri(pointList[int(tempText[1])-1],pointList[int(tempText[2])-1],pointList[int(tempText[3])-1]))

f= open("teapot.obj","r")

text = ReadObjFile(f)
ParsePoints(text)
ParseTris(text)
# print("Points:")
# cnt=0
# for i in pointList:
#     cnt+=1
#     print(str(cnt)+ ": "+ str(i.GetX()) + " "+ str(i.GetY())+ " "+ str(i.GetZ()))
# cnt=0
# for i in triList:
#     cnt+=1
#     print(str(cnt)+": "+ str(i.GetPoints()))
circleAreaSum=0
maximumCos=0
for i in triList:
    if i.CalculateMaximumCosine()>maximumCos:
        maximumCos=i.CalculateMaximumCosine()
    circleAreaSum+=i.CalculateIncirleArea()
print("Сумма площадей окружностей:" + str(circleAreaSum))
print("Косинус:"+str(maximumCos))
