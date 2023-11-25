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
image= np.full((int(screenWidth),int(screenWidth),3),[19,24,98],dtype=np.float32)
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


def make_bezier(xys):
    # xys should be a sequence of 2-tuples (Bezier control points)
    n = len(xys)
    combinations = pascal_row(n-1)
    def bezier(ts):
        # This uses the generalized formula for bezier curves
        # http://en.wikipedia.org/wiki/B%C3%A9zier_curve#Generalization
        result = []
        for t in ts:
            tpowers = (t**i for i in range(n))
            upowers = reversed([(1-t)**i for i in range(n)])
            coefs = [c*a*b for c, a, b in zip(combinations, tpowers, upowers)]
            result.append(
                tuple(sum([coef*p for coef, p in zip(coefs, ps)]) for ps in zip(*xys)))
        return result
    return bezier

def pascal_row(n, memo={}):
    # This returns the nth row of Pascal's Triangle
    if n in memo:
        return memo[n]
    result = [1]
    x, numerator = 1, n
    for denominator in range(1, n//2+1):
        # print(numerator,denominator,x)
        x *= numerator
        x /= denominator
        result.append(x)
        numerator -= 1
    if n&1 == 0:
        # n is even
        result.extend(reversed(result[:-1]))
    else:
        result.extend(reversed(result))
    memo[n] = result
    return result

def AddCurve(point0,point1,point2):
    pointList.append(point0)
    pointList.append(point1)
    pointList.append(point2)
    bodyCurveList.append(BezierCurve(point0,point1,point2))

def Draw(points,color):
    for i in points:
        X = np.matrix([[i[0]], [i[1]], [1]])
        R = np.matrix([[math.cos(math.radians(-90)), -1 * math.sin(math.radians(-90)), 0],
                       [math.sin(math.radians(-90)), math.cos(math.radians(-90)), 0], [0, 0, 1]])
        rotatedX = np.matmul(R, X)
        image[int(rotatedX[0][0]),int(rotatedX[1][0])]=color




ts = [t/screenWidth for t in range(int(screenWidth+1))]
#
#
#
# xys = [(100, 50), (100, 0), (50, 0), (50, 35)]
# bezier = make_bezier(xys)
# points.extend(bezier(ts))
#
# xys = [(50, 35), (50, 0), (0, 0), (0, 50)]
# bezier = make_bezier(xys)
# points.extend(bezier(ts))
#
# xys = [(0, 50), (20, 80), (50, 100)]
# bezier = make_bezier(xys)
# points.extend(bezier(ts))

xys = [(254, 690), (438, 518), (704, 687)]
bezier = make_bezier(xys)
points = bezier(ts)

Draw(points,color)

im=plt.imshow(image.astype('uint8'))

plt.show()





