import math
import numpy as np
import re
import tkinter as tk
import matplotlib.pyplot as plt
import locale

pointList=[]
bodyCurveList=[]
eyesCurveList=[]
tailCurveList=[]
irisesCurveList=[]

root =tk.Tk()
screenWidth = root.winfo_screenwidth()/3
fig = plt.figure()
image= np.full((int(screenWidth),int(screenWidth),3),[19,24,98],dtype=np.float32)
bodyColor=[0,0,0]
eyeColor=[255,255,51]

def FormBezier(xys):
    n = len(xys)
    combinations = PascalRow(n - 1)
    def Bezier(ts):
        result = []
        for t in ts:
            tpowers = (t**i for i in range(n))
            upowers = reversed([(1-t)**i for i in range(n)])
            coefs = [c*a*b for c, a, b in zip(combinations, tpowers, upowers)]
            result.append(tuple(sum([coef*p for coef, p in zip(coefs, ps)]) for ps in zip(*xys)))
        return result
    return Bezier

def PascalRow(n, memo={}):
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

def ReadObjFile(file):
    text=[]
    text=file.readlines()
    return text

def Draw(points,color):
    for i in points:
        X = np.matrix([[i[0]], [i[1]], [1]])
        F = np.matrix([[-1, 0, 0],
                       [0, 1, 0], [0, 0, 1]])
        R = np.matrix([[math.cos(math.radians(-90)), -1 * math.sin(math.radians(-90)), 0],
                       [math.sin(math.radians(-90)), math.cos(math.radians(-90)), 0], [0, 0, 1]])
        F= np.matrix([[1, 0, 0],
                       [0, -1, 0], [0, 0, 1]])
        flippedX= np.matmul(F,X)
        rotatedX = np.matmul(R, X)
        flippedX = np.matmul(F, rotatedX)
        image[int(flippedX[0][0]),int(flippedX[1][0])]=color
def ParseBodyBezierFile(text):
    for i in text:
        tempText = i.split(" ")
        points=[]
        for j in tempText:
            individualCoords=[]
            individualCoords= j.split(",")
            if individualCoords[0] == "\n":
                break


            points.append((int(round(float(individualCoords[0]))),int(round(float(individualCoords[1])))))
        bezier = FormBezier(points)
        points1 = bezier(ts)
        bodyCurveList.append(points1)

def ParseEyesBezierFile(text):
    for i in text:
        tempText = i.split(" ")
        points=[]
        for j in tempText:
            individualCoords=[]
            individualCoords= j.split(",")
            if individualCoords[0] == "\n":
                break
            points.append((int(round(float(individualCoords[0]))),int(round(float(individualCoords[1])))))
        bezier = FormBezier(points)
        points1 = bezier(ts)
        eyesCurveList.append(points1)

def DrawBody(color):
    for i in bodyCurveList:
        Draw(i,color)
    FillIn(color)
def DrawEyes(color):
    for i in eyesCurveList:
        Draw(i,color)
    FillIn(color)

def FillIn(color):
    for i in range (0,int(screenWidth)):
        colorFlag = False
        for j in range(0,int(screenWidth)):
            if (image[i][j] == color).all() and not colorFlag and (image[i][j + 1] == color).all():
                while (image[i][j] == color).all():
                    j+=1
            if (image[i][j]==color).all() and colorFlag and (image[i][j+1]!=color).any():
                colorFlag=False
                j+=1
                print ("Removed color flag")
            if (image[i][j] == color).all() and not colorFlag and not(image[i][j+1]==color).all():
                finishedFlag=False
                for j1 in range(j,int(screenWidth)):
                    if (image[i][j] == color).all():
                        finishedFlag=True
                if finishedFlag:
                    colorFlag=True
                print("Added color flag")
            if (image[i][j]!=color).any() and colorFlag:
                image[i][j]=color


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

f1= open("BodyBezierpoints.txt","r")
bodyText=ReadObjFile(f1)
f2= open("EyesBezierpoints.txt","r")
eyesText=ReadObjFile(f2)

ParseBodyBezierFile(bodyText)
ParseEyesBezierFile(eyesText)
DrawBody(bodyColor)
DrawEyes(eyeColor)

# xys = [(254, 690), (438, 518), (704, 687)]
# bezier = make_bezier(xys)
# points = bezier(ts)

# Draw(points,color)

im=plt.imshow(image.astype('uint8'))

plt.show()





