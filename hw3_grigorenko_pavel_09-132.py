import math
import numpy as np
import re
import tkinter as tk
import matplotlib.pyplot as plt
import locale
import sys
import threading

threading.stack_size(67108864) # 64MB stack
sys.setrecursionlimit(2 ** 20) # something real big
                               # you actually hit the 64MB limit first
                               # going by other answers, could just use 2**32-1

# only new threads get the redefined stack size


pointList=[]
bodyCurveList=[]
transformedBodyPointsList=[]
eye1CurveList=[]
transformedEye1PointsList=[]
eye2CurveList=[]
transformedEye2PointsList=[]
tailCurveList=[]
transformedTailPointsList=[]
iris1CurveList=[]
transformedIris1PointsList=[]
iris1CurveList=[]
transformedIris1PointsList=[]

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
    pointList=[]
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
        pointList.append((int(flippedX[0][0]),int(flippedX[1][0])))
    return pointList
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

def ParseEye1BezierFile(text):
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
        eye1CurveList.append(points1)
def ParseEye2BezierFile(text):
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
        eye2CurveList.append(points1)
def ParseTailBezierFile(text):
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
        tailCurveList.append(points1)

def DrawBody(color):
    for i in bodyCurveList:
        transformedBodyPointsList.extend(Draw(i,color))
    FillInBody(color)
def DrawEyes(color):
    for i in eye1CurveList:
        transformedEye1PointsList.extend(Draw(i,color))
    FillInEye1(color)
    for i in eye2CurveList:
        transformedEye2PointsList.extend(Draw(i,color))
    FillInEye2(color)
def DrawTail(color):
    for i in tailCurveList:
        transformedTailPointsList.extend(Draw(i,color))
    # FillInTail(color)

def Fill(x, y, startColor, color):
    if (image[x][y] == color).all():
        print("cringe, same colour")
        return
    else:
        image[x][y] = color
        print("Coloured X:"+ str(x)+" Y:"+str(y)+" color:"+str(image[x][y]))

        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for n in neighbors:
            if 0 <= n[0] <= screenWidth - 1 and 0 <= n[1] <= screenWidth - 1:
                Fill(n[0], n[1], startColor, color)


def FillInEye1(color):
    minX=screenWidth
    minY=screenWidth
    maxX=0
    maxY=0
    for i in transformedEye1PointsList:
        if i[0]<minX:
            minX=int(i[0])
        if i[0]>maxX:
            maxX=int(i[0])
        if i[1]<minY:
            minY=int(i[1])
        if i[1]>maxY:
            maxY=int(i[1])
    for i in range(minX,maxX+1):
        for j in range(minY,maxY+1):
            left=False
            right=False
            up=False
            down=False
            for k in range(j+1,int(screenWidth)):
                if (image[i][k]==color).all():
                    right=True
                    print("Found right")
                    break
            for k in range(j-1,0,-1):
                if (image[i][k]==color).all():
                    left=True
                    print("Found left")
                    break
            for k in range(i+1,int(screenWidth)):
                if (image[k][j]==color).all():
                    down=True
                    print("Found down")
                    break
            for k in range(i-1, 0,-1):
                if (image[k][j] == color).all():
                    up = True
                    print("Found up")
                    break
            if (left and right and up and down and (image[i][j]!=color).any()):
                print("found all, coloring eye1. Point: X:"+ str(i)+" Y:"+str(j)+" color:"+str(image[i][j]))
                # Fill(i,j,image[i][j],color)
                thread = threading.Thread(target=Fill,args=(i,j,image[i][j],color))
                thread.start()
                thread.join()
                return

def FillInEye2(color):
    minX=screenWidth
    minY=screenWidth
    maxX=0
    maxY=0
    for i in transformedEye2PointsList:
        if i[0]<minX:
            minX=int(i[0])
        if i[0]>maxX:
            maxX=int(i[0])
        if i[1]<minY:
            minY=int(i[1])
        if i[1]>maxY:
            maxY=int(i[1])
    for i in range(minX,maxX+1):
        for j in range(minY,maxY+1):
            left=False
            right=False
            up=False
            down=False
            for k in range(j+1,int(screenWidth)):
                if (image[i][k]==color).all():
                    right=True
                    print("Found right")
                    break
            for k in range(j-1,0,-1):
                if (image[i][k]==color).all():
                    left=True
                    print("Found left")
                    break
            for k in range(i+1,int(screenWidth)):
                if (image[k][j]==color).all():
                    down=True
                    print("Found down")
                    break
            for k in range(i-1, 0,-1):
                if (image[k][j] == color).all():
                    up = True
                    print("Found up")
                    break
            if (left and right and up and down and (image[i][j]!=color).any()):
                print("found all, coloring eye2. Point: X:"+ str(i)+" Y:"+str(j)+" color:"+str(image[i][j]))
                thread = threading.Thread(target=Fill, args=(i, j, image[i][j], color))
                thread.start()
                thread.join()
                return

def FillInBody(color):
    minX = screenWidth
    minY = screenWidth
    maxX = 0
    maxY = 0
    for i in transformedBodyPointsList:
        if i[0] < minX:
            minX = int(i[0])
        if i[0] > maxX:
            maxX = int(i[0])
        if i[1] < minY:
            minY = int(i[1])
        if i[1] > maxY:
            maxY = int(i[1])
    for i in range(maxX,minX-1,-1):
        for j in range(maxY,minY-1,-1):
            left=False
            right=False
            up=False
            down=False
            for k in range(j+1,int(screenWidth)):
                if (image[i][k]==color).all():
                    right=True
                    print("Found right")
                    break
            for k in range(j-1,0,-1):
                if (image[i][k]==color).all():
                    left=True
                    print("Found left")
                    break
            for k in range(i+1,int(screenWidth)):
                if (image[k][j]==color).all():
                    down=True
                    print("Found down")
                    break
            for k in range(i-1, 0,-1):
                if (image[k][j] == color).all():
                    up = True
                    print("Found up")
                    break
            if (left and right and up and down and (image[i][j]!=color).any()):
                print("found all, coloring body. Point: X:"+ str(i)+" Y:"+str(j)+" color:"+str(image[i][j]))
                thread = threading.Thread(target=Fill, args=(i, j, image[i][j], color))
                thread.start()
                thread.join()
                return
def FillInTail(color):
    minX = screenWidth
    minY = screenWidth
    maxX = 0
    maxY = 0
    for i in transformedTailPointsList:
        if i[0] < minX:
            minX = int(i[0])
        if i[0] > maxX:
            maxX = int(i[0])
        if i[1] < minY:
            minY = int(i[1])
        if i[1] > maxY:
            maxY = int(i[1])
    for i in range(maxX, minX - 1, -1):
        for j in range(maxY, minY - 1, -1):
            left = False
            right = False
            up = False
            down = False
            for k in range(j + 1, int(screenWidth)):
                if (image[i][k] == color).all():
                    right = True
                    print("Found right")
                    break
            for k in range(j - 1, 0, -1):
                if (image[i][k] == color).all():
                    left = True
                    print("Found left")
                    break
            for k in range(i + 1, int(screenWidth)):
                if (image[k][j] == color).all():
                    down = True
                    print("Found down")
                    break
            for k in range(i - 1, 0, -1):
                if (image[k][j] == color).all():
                    up = True
                    print("Found up")
                    break
            if (left and right and up and down and (image[i][j] != color).any()):
                print("found all, coloring tail. Point: X:" + str(i) + " Y:" + str(j) + " color:" + str(image[i][j]))
                thread = threading.Thread(target=Fill, args=(i, j, image[i][j], color))
                thread.start()
                thread.join()
                return


# def FillIn(color):
#     for i in range (0,int(screenWidth)):
#         colorFlag = False
#         for j in range(0,int(screenWidth)):
#             test=False
#             if (image[i][j] == color).all() and not colorFlag and (image[i][j + 1] == color).all():
#                 while (image[i][j] == color).all():
#                     test=True
#                     j+=1
#                 colorFlag = False
#                 print("Skipped filled row: x:" + str(i) + " y:"+ str(j)+ " color:" +str(image[i][j]))
#                 test=True
#             if (image[i][j]==color).all() and colorFlag and (image[i][j+1]!=color).any():
#                 colorFlag=False
#                 j+=1
#             if (image[i][j] == color).all() and not colorFlag and (image[i][j+1]!=color).any():
#                 finishedFlag=False
#                 for j1 in range(j,int(screenWidth)):
#                     if (image[i][j] == color).all():
#                         finishedFlag=True
#                 if finishedFlag:
#                     colorFlag=True
#
#             if (image[i][j]!=color).any() and colorFlag:
#                 image[i][j]=color
#                 if(test):
#                     print("x:" + str(i) + " y:"+ str(j))


ts = [t/screenWidth for t in range(int(screenWidth+1))]

f1= open("BodyBezierpoints.txt","r")
bodyText=ReadObjFile(f1)
f2= open("Eye1Bezierpoints.txt","r")
eye1Text=ReadObjFile(f2)
f3= open("Eye2Bezierpoints.txt","r")
eye2Text=ReadObjFile(f3)
f4= open("TailBezierpoints.txt",'r')
tailText=ReadObjFile(f4)

ParseBodyBezierFile(bodyText)
ParseEye1BezierFile(eye1Text)
ParseEye2BezierFile(eye2Text)
ParseTailBezierFile(tailText)
DrawBody(bodyColor)
DrawEyes(eyeColor)
DrawTail(bodyColor)

im = plt.imshow(image.astype('uint8'))

plt.show()






