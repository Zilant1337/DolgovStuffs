import math
import numpy as np
import re
pointList=[]
triList=[]
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
    def CalculateIncirleArea(self):
        a = math.sqrt((self.point2.GetX() -self.point1.GetX()) ** 2 + (self.point2.GetY() - self.point1.GetY()) ** 2 +(self.point2.GetZ() - self.point1.GetZ()) **  2)
        b = math.sqrt((self.point3.GetX() - self.point2.GetX()) ** 2 + (self.point3.GetY() - self.point2.GetY()) ** 2 + (self.point3.GetZ() - self.point1.GetZ()) **  2)
        c = math.sqrt((self.point1.GetX() - self.point3.GetX()) ** 2 + (self.point1.GetY() - self.point3.GetY()) ** 2 + (self.point1.GetZ() - self.point3.GetZ()) **  2)

        s = (a + b + c) / 2

        inradius = math.sqrt(((s-a)*(s-b)*(s-c))/s)

        incircleArea = math.pi * inradius ** 2

        return incircleArea

    def CalculateCosine(self,edgeIndex1,edgeIndex2):
        a = math.sqrt(
            (self.point2.GetX() - self.point1.GetX()) ** 2 + (self.point2.GetY() - self.point1.GetY()) ** 2 + (
                        self.point2.GetZ() - self.point1.GetZ()) ** 2)
        b = math.sqrt(
            (self.point3.GetX() - self.point2.GetX()) ** 2 + (self.point3.GetY() - self.point2.GetY()) ** 2 + (
                        self.point3.GetZ() - self.point1.GetZ()) ** 2)
        c = math.sqrt(
            (self.point1.GetX() - self.point3.GetX()) ** 2 + (self.point1.GetY() - self.point3.GetY()) ** 2 + (
                        self.point1.GetZ() - self.point3.GetZ()) ** 2)
        #Неверные вычисления, тут должны использоваться координаты векторов, а не точек
        if ((edgeIndex1==1 and edgeIndex2==0) or (edgeIndex1==0 and edgeIndex2==1)):
            scalar=self.point2.GetX()*self.point1.GetX()+self.point2.GetY()*self.point1.GetY()+self.point2.GetZ()*self.point1.GetZ()
            return scalar/(a*b)
        if ((edgeIndex1==2 and edgeIndex2==1) or (edgeIndex1==1 and edgeIndex2==2)):
            scalar=self.point3.GetX()*self.point2.GetX()+self.point3.GetY()*self.point2.GetY()+self.point3.GetZ()*self.point2.GetZ()
            return scalar/(c*b)
        if ((edgeIndex1==2 and edgeIndex2==0) or (edgeIndex1==0 and edgeIndex2==2)):
            scalar=self.point3.GetX()*self.point1.GetX()+self.point3.GetY()*self.point1.GetY()+self.point3.GetZ()*self.point1.GetZ()
            return scalar/(a*c)


    def CalculateMaximumCosine(self):
        return max(self.CalculateCosine(0,1),max(self.CalculateCosine(1,2),self.CalculateCosine(0,2)))





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
        print("x:"+str (i.GetPoints()[0].GetX()))
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
            print(tempText[0])
            print(pointList[int(tempText[1])-1])
            print(pointList[int(tempText[2])-1])
            print(pointList[int(tempText[3])-1])
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


#maximumCos=0
for i in triList:
#     #if i.CalculateMaximumCosine()>maximumCos:
#     #    maximumCos=i.CalculateMaximumCosine()
    circleAreaSum+=i.CalculateIncirleArea()
    print(i.CalculateIncirleArea())
print("Сумма площадей окружностей:" + str(circleAreaSum))
#print("Косинус:"+str(maximumCos))
