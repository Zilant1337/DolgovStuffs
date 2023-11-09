import math
import numpy as np
import re

class tri:
    points=[]
    def __init__(self,points):
        self.points=points.copy()
    def __init__(self,point1,point2,point3):
        self.points.append(point1)
        self.points.append(point2)
        self.points.append(point3)
    def CalculateSquareArea(self):
        return 1/2*abs((self.points[1].GetX()-self.points[0].GetX())(self.points[2].GetY()-self.points[0].GetY())-(self.points[2].GetX()-self.points[0].GetX())(self.points[1].GetY()-self.points[0].GetY()))
    def CalculateIncirleArea(self):
        a = math.sqrt((self.points[1].GetX() -self.points[0].GetX()) ** 2 + (self.points[1].GetY() - self.points[0].GetY()) ** 2)
        b = math.sqrt((self.points[2].GetX() - self.points[1].GetX()) ** 2 + (self.points[2].GetY() - self.points[1].GetY()) ** 2)
        c = math.sqrt((self.points[0].GetX() - self.points[2].GetX()) ** 2 + (self.points[0].GetY() - self.points[2].GetY()) ** 2)

        # Calculate the semi-perimeter
        s = (a + b + c) / 2

        # Calculate the area of the triangle
        triangle_area = self.CalculateSquareArea()
        # Calculate the inradius
        inradius = triangle_area / s

        # Calculate the area of the incircle
        incircle_area = math.pi * inradius ** 2

        return incircle_area
class point:
    x=0
    y=0
    z=0
    def __init__(self,x=0,y=0,z=0):
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
