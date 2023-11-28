import math
import numpy as np
import re
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import locale
import sys
import threading

threading.stack_size(67108864)
sys.setrecursionlimit(2 ** 20)

bodyCurveList=[]
transformedBodyPointsList=[]
bodyFillPoint=(800,466)
eye1CurveList=[]
transformedEye1PointsList=[]
eye1FillPoint=(172,387)
eye2CurveList=[]
transformedEye2PointsList=[]
eye2FillPoint=(172,519)
tailCurvePointList=[]
tailCurveList=[]
tailFillPoint=(640,209)
transformedTailPointsList=[]
iris1CurveList=[]
transformedIris1PointsList=[]
iris1FillPoint=(176,405)
iris2CurveList=[]
transformedIris2PointsList=[]
iris2FillPoint=(180,507)
noseCurveList=[]
transformedNosePointsList=[]
noseFillPoint=(209,452)
mouthCurveList=[]
transformedMouthPointsList=[]


root =tk.Tk()
framesCount=200
frames=[]
screenWidth = 2560/3
fig = plt.figure()
bodyColor=[0,0,0]
tailColor=[1,0,0]
eyeColor=[255,255,51]
irisColor=[0,0,1]
mouthColor=[54,69,79]
backgroundColor=[19,24,98]
image= np.full((int(screenWidth),int(screenWidth),3),[19,24,98],dtype=np.float32)
ts = [t/screenWidth for t in range(int(screenWidth+1))]


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

def InitialDraw(points,color):
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
def Draw(points,color):
    for i in points:
        X = np.matrix([[i[0]], [i[1]], [1]])
        image[int(X[0][0]),int(X[1][0])]=color
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
    global tailCurvePointList
    for i in text:
        tempText = i.split(" ")
        points=[]
        for j in tempText:
            individualCoords=[]
            individualCoords= j.split(",")
            if individualCoords[0] == "\n":
                break
            points.append((int(round(float(individualCoords[0]))),int(round(float(individualCoords[1])))))
        tailCurvePointList.append(points)
        bezier = FormBezier(points)
        points2 = bezier(ts)
        tailCurveList.append(points2)
    newTailCurvePoints = []
    for j in tailCurvePointList:
        points1 = []
        for k in j:
            X = np.matrix([[k[0]], [k[1]], [1]])
            F = np.matrix([[-1, 0, 0],
                           [0, 1, 0], [0, 0, 1]])
            R = np.matrix([[math.cos(math.radians(-90)), -1 * math.sin(math.radians(-90)), 0],
                           [math.sin(math.radians(-90)), math.cos(math.radians(-90)), 0], [0, 0, 1]])
            F = np.matrix([[1, 0, 0],
                           [0, -1, 0], [0, 0, 1]])
            flippedX = np.matmul(F, X)
            rotatedX = np.matmul(R, X)
            flippedX = np.matmul(F, rotatedX)
            points1.append((int(flippedX[0][0]), int(flippedX[1][0])))
        newTailCurvePoints.append(points1)
    tailCurvePointList = newTailCurvePoints
def ParseIris1BezierFile(text):
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
        iris1CurveList.append(points1)
def ParseIris2BezierFile(text):
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
        iris2CurveList.append(points1)
def ParseNoseBezierFile(text):
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
        noseCurveList.append(points1)
def ParseMouthBezierFile(text):
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
        mouthCurveList.append(points1)

def DrawBody(color):
    for i in bodyCurveList:
        transformedBodyPointsList.extend(InitialDraw(i,color))

    FillInBody(color)
def DrawEyes(color):
    for i in eye1CurveList:
        transformedEye1PointsList.extend(InitialDraw(i,color))
    FillInEye1(color)
    for i in eye2CurveList:
        transformedEye2PointsList.extend(InitialDraw(i,color))
    FillInEye2(color)
def InitialDrawTail(color):
    for i in tailCurveList:
        transformedTailPointsList.extend(InitialDraw(i,color))
    FillInTail(color)
def DrawTail(color):
    ReformTailCurves()
    transformedTailPointsList=[]
    for i in tailCurveList:
        transformedTailPointsList.append(Draw(i,color))

def DrawIrises(color):
    for i in iris1CurveList:
        transformedIris1PointsList.extend(InitialDraw(i,color))
    FillInIris1(color)
    for i in iris2CurveList:
        transformedIris2PointsList.extend(InitialDraw(i,color))
    FillInIris2(color)
def DrawNoseMouth(color):
    for i in noseCurveList:
        transformedNosePointsList.extend(InitialDraw(i,color))
    FillInNose(color)
    for i in mouthCurveList:
        transformedMouthPointsList.extend(InitialDraw(i,color))


def Fill(x, y, startColor, color):
    if (image[x][y] == color).all():
        return
    else:
        image[x][y] = color

        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for n in neighbors:
            if 0 <= n[0] <= screenWidth - 1 and 0 <= n[1] <= screenWidth - 1:
                Fill(n[0], n[1], startColor, color)


def FillInEye1(color):
    print("Started filling eye1")
    thread = threading.Thread(target=Fill, args=(
        eye1FillPoint[0], eye1FillPoint[1], image[eye1FillPoint[0]][eye1FillPoint[1]], color))
    thread.start()
    thread.join()
    return

def FillInEye2(color):
    print("Started filling eye2")
    thread = threading.Thread(target=Fill, args=(eye2FillPoint[0], eye2FillPoint[1], image[eye2FillPoint[0]][eye2FillPoint[1]], color))
    thread.start()
    thread.join()
    return

def FillInBody(color):
    print("Started filling body")
    thread = threading.Thread(target=Fill, args=(bodyFillPoint[0], bodyFillPoint[1], image[bodyFillPoint[0]][bodyFillPoint[1]], color))
    thread.start()
    thread.join()
    return
def FillInTail(color):
    global transformedTailPointsList
    print("Started filling tail")
    # minX = screenWidth
    # minY = screenWidth
    # maxX = 0
    # maxY = 0
    # for i in transformedTailPointsList:
    #     if i[0] < minX:
    #         minX = int(i[0])
    #     if i[0] > maxX:
    #         maxX = int(i[0])
    #     if i[1] < minY:
    #         minY = int(i[1])
    #     if i[1] > maxY:
    #         maxY = int(i[1])
    # for i in range(minX, maxX +1):
    #     for j in range(minY, maxY + 1):
    #         left = False
    #         right = False
    #         up = False
    #         down = False
    #         for k in range(j + 1, int(screenWidth)):
    #             if (image[i][k] == color).all():
    #                 right = True
    #                 break
    #         for k in range(j - 1, 0, -1):
    #             if (image[i][k] == color).all():
    #                 left = True
    #                 break
    #         for k in range(i + 1, int(screenWidth)):
    #             if (image[k][j] == color).all():
    #                 down = True
    #                 break
    #         for k in range(i - 1, 0, -1):
    #             if (image[k][j] == color).all():
    #                 up = True
    #                 break
    #         if (left and right and up and down and (image[i][j] == backgroundColor).all()):
    #             print("found all, coloring tail. Point: X:" + str(i) + " Y:" + str(j) + " color:" + str(image[i][j]))
    #             thread = threading.Thread(target=Fill, args=(i, j, image[i][j], color))
    #             thread.start()
    #             thread.join()
    #             return
    thread = threading.Thread(target=Fill, args=(
    tailFillPoint[0], tailFillPoint[1], image[tailFillPoint[0]][tailFillPoint[1]], color))
    thread.start()
    thread.join()
    return
def FillInIris1(color):
    print("Started filling iris1")
    # minX=screenWidth
    # minY=screenWidth
    # maxX=0
    # maxY=0
    # for i in transformedIris1PointsList:
    #     if i[0]<minX:
    #         minX=int(i[0])
    #     if i[0]>maxX:
    #         maxX=int(i[0])
    #     if i[1]<minY:
    #         minY=int(i[1])
    #     if i[1]>maxY:
    #         maxY=int(i[1])
    # for i in range(minX,maxX+1):
    #     for j in range(minY,maxY+1):
    #         left=False
    #         right=False
    #         up=False
    #         down=False
    #         for k in range(j+1,int(screenWidth)):
    #             if (image[i][k]==color).all():
    #                 right=True
    #                 break
    #         for k in range(j-1,0,-1):
    #             if (image[i][k]==color).all():
    #                 left=True
    #                 break
    #         for k in range(i+1,int(screenWidth)):
    #             if (image[k][j]==color).all():
    #                 down=True
    #                 break
    #         for k in range(i-1, 0,-1):
    #             if (image[k][j] == color).all():
    #                 up = True
    #                 break
    #         if (left and right and up and down and (image[i][j]!=color).any()):
    #             print("found all, coloring iris1. Point: X:"+ str(i)+" Y:"+str(j)+" color:"+str(image[i][j]))
    #             # Fill(i,j,image[i][j],color)
    #             thread = threading.Thread(target=Fill,args=(i,j,image[i][j],color))
    #             thread.start()
    #             thread.join()
    #             return
    thread = threading.Thread(target=Fill, args=(iris1FillPoint[0], iris1FillPoint[1], image[iris1FillPoint[0]][iris1FillPoint[1]], color))
    thread.start()
    thread.join()
    return
def FillInIris2(color):
    print("Started filling iris2")
    # minX=screenWidth
    # minY=screenWidth
    # maxX=0
    # maxY=0
    # for i in transformedIris2PointsList:
    #     if i[0]<minX:
    #         minX=int(i[0])
    #     if i[0]>maxX:
    #         maxX=int(i[0])
    #     if i[1]<minY:
    #         minY=int(i[1])
    #     if i[1]>maxY:
    #         maxY=int(i[1])
    # for i in range(minX,maxX+1):
    #     for j in range(minY,maxY+1):
    #         left=False
    #         right=False
    #         up=False
    #         down=False
    #         for k in range(j+1,int(screenWidth)):
    #             if (image[i][k]==color).all():
    #                 right=True
    #                 break
    #         for k in range(j-1,0,-1):
    #             if (image[i][k]==color).all():
    #                 left=True
    #                 break
    #         for k in range(i+1,int(screenWidth)):
    #             if (image[k][j]==color).all():
    #                 down=True
    #                 break
    #         for k in range(i-1, 0,-1):
    #             if (image[k][j] == color).all():
    #                 up = True
    #                 break
    #         if (left and right and up and down and ((image[i][j]==eyeColor).all() or (image[i][j]==backgroundColor).all())):
    #             print("found all, coloring Iris2. Point: X:"+ str(i)+" Y:"+str(j)+" color:"+str(image[i][j]))
    #             # Fill(i,j,image[i][j],color)
    #             thread = threading.Thread(target=Fill,args=(i,j,image[i][j],color))
    #             thread.start()
    #             thread.join()
    #             return
    thread = threading.Thread(target=Fill, args=(
        iris2FillPoint[0], iris2FillPoint[1], image[iris2FillPoint[0]][iris2FillPoint[1]], color))
    thread.start()
    thread.join()
    return
def FillInNose(color):
    print("Started filling nose")
    thread = threading.Thread(target=Fill, args=(
        noseFillPoint[0], noseFillPoint[1], image[noseFillPoint[0]][noseFillPoint[1]], color))
    thread.start()
    thread.join()
    return
def ReformTailCurves():
    global ts
    global tailCurveList
    global tailCurvePointList
    newTailCurveList=[]
    for i in tailCurvePointList:
        bezier = FormBezier(i)
        points2 = bezier(ts)
        newTailCurveList.append(points2)
    tailCurveList=newTailCurveList


f1= open("BodyBezierpoints.txt","r")
bodyText=ReadObjFile(f1)
f2= open("Eye1Bezierpoints.txt","r")
eye1Text=ReadObjFile(f2)
f3= open("Eye2Bezierpoints.txt","r")
eye2Text=ReadObjFile(f3)
f4= open("TailBezierpoints.txt",'r')
tailText=ReadObjFile(f4)
f5=open("Iris1Bezierpoints.txt","r")
iris1Text=ReadObjFile(f5)
f6=open("Iris2Bezierpoints.txt","r")
iris2Text=ReadObjFile(f6)
f7=open("NoseBezierpoints.txt","r")
noseText=ReadObjFile(f7)
f8=open("MouthBezierpoints.txt")
mouthText=ReadObjFile(f8)

ParseBodyBezierFile(bodyText)
ParseEye1BezierFile(eye1Text)
ParseEye2BezierFile(eye2Text)
ParseTailBezierFile(tailText)
ParseIris1BezierFile(iris1Text)
ParseIris2BezierFile(iris2Text)
ParseNoseBezierFile(noseText)
ParseMouthBezierFile(mouthText)
InitialDrawTail(tailColor)
DrawBody(bodyColor)
DrawEyes(eyeColor)
DrawIrises(irisColor)
DrawNoseMouth(mouthColor)
im = plt.imshow(image.astype('uint8'))
frames.append([im])

irisMoveCounter=0
irisTranslateMatrix =np.matrix([[1, 0, 0],[0, 1, 1], [0, 0, 1]])
recenterPointX=-transformedTailPointsList[0][0]
recenterPointY=-transformedTailPointsList[0][1]
tailRecenteringMatrix=np.matrix([[1, 0, recenterPointX],[0, 1, recenterPointY],[0, 0, 1]])
tailRotationMatrix=np.matrix([[math.cos(math.radians(1)), -1 * math.sin(math.radians(1)), 0],
                       [math.sin(math.radians(1)), math.cos(math.radians(1)), 0], [0, 0, 1]])
for i in range(framesCount):
    irisMoveCounter+=1
    image = np.full((int(screenWidth), int(screenWidth), 3), [19, 24, 98], dtype=np.float32)
    if irisMoveCounter%3==0:
        movedTailCurvePoints = []
        for j in tailCurvePointList:
            movedTailPoints = []
            for k in j:
                X = np.matrix([[k[0]], [k[1]], [1]])
                centeredX = np.matmul(tailRecenteringMatrix, X)
                rotatedX = np.matmul(tailRotationMatrix, centeredX)
                finalX = np.matmul(np.linalg.inv(tailRecenteringMatrix), rotatedX)
                movedTailPoints.append((int(finalX[0][0]), int(finalX[1][0])))
            movedTailCurvePoints.append(movedTailPoints)
        tailCurvePointList=movedTailCurvePoints
        X = np.matrix([[tailFillPoint[0]], [tailFillPoint[1]], [1]])
        centeredX = np.matmul(tailRecenteringMatrix, X)
        rotatedX = np.matmul(tailRotationMatrix, centeredX)
        finalX = np.matmul(np.linalg.inv(tailRecenteringMatrix), rotatedX)
        tailFillPoint=(int(finalX[0][0]), int(finalX[1][0]))



        movedIris1 = []
        movedIris2 = []
        for j in transformedIris2PointsList:
            X = np.matrix([[j[0]], [j[1]], [1]])
            movedX = np.matmul(irisTranslateMatrix, X)
            movedIris1.append((int(movedX[0][0]), int(movedX[1][0])))

        X = np.matrix([[iris1FillPoint[0]], [iris1FillPoint[1]], [1]])
        movedX = np.matmul(irisTranslateMatrix, X)
        iris1FillPoint=(int(movedX[0][0]), int(movedX[1][0]))
        for j in transformedIris1PointsList:
            X = np.matrix([[j[0]], [j[1]], [1]])
            movedX = np.matmul(irisTranslateMatrix, X)
            movedIris2.append((int(movedX[0][0]), int(movedX[1][0])))
        print("before:"+ str(iris2FillPoint[0])+" "+str(iris2FillPoint[1]))
        X = np.matrix([[iris2FillPoint[0]], [iris2FillPoint[1]], [1]])
        movedX = np.matmul(irisTranslateMatrix, X)
        iris2FillPoint = (int(movedX[0][0]), int(movedX[1][0]))
        print("after:" + str(iris2FillPoint[0]) + " " + str(iris2FillPoint[1]))
        transformedIris1PointsList=movedIris1
        transformedIris2PointsList=movedIris2

    if i==50 or i==150:
        a = irisTranslateMatrix.getA()
        a[1][2] *= -1
        irisTranslateMatrix = np.asmatrix(a)
    if i==50 or i==150:
        a = np.array([[math.cos(math.radians(-1)), -1 * math.sin(math.radians(-1)), 0],
                      [math.sin(math.radians(-1)), math.cos(math.radians(-1)), 0], [0, 0, 1]])

        tailRotationMatrix = np.asmatrix(a)
    if i==100:
        a = np.array([[math.cos(math.radians(1)), -1 * math.sin(math.radians(1)), 0],
                       [math.sin(math.radians(1)), math.cos(math.radians(1)), 0], [0, 0, 1]])

        tailRotationMatrix = np.asmatrix(a)




    DrawTail(tailColor)
    FillInTail(tailColor)
    Draw(transformedBodyPointsList, bodyColor)
    FillInBody(bodyColor)
    Draw(transformedEye1PointsList, eyeColor)
    FillInEye1(eyeColor)
    Draw(transformedEye2PointsList, eyeColor)
    FillInEye2(eyeColor)
    Draw(transformedIris1PointsList, irisColor)
    Draw(transformedIris2PointsList, irisColor)
    FillInIris1(irisColor)
    FillInIris2(irisColor)
    Draw(transformedMouthPointsList, mouthColor)
    Draw(transformedNosePointsList, mouthColor)
    FillInNose(mouthColor)
    print("Generated frame" + str(i))
    img = plt.imshow(image.astype('uint8'))
    frames.append([img])




ani = animation.ArtistAnimation(fig, frames, interval=40, blit=True, repeat_delay=0)
writer = PillowWriter(fps=24)
ani.save("cat.gif", writer=writer)

plt.show()






