import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation

file_path = 'Skull.obj'

f = []
v = []
v_test = []
vn = []
vt = []
width = 800
height = 800
ZBuffer = np.ones((width, height))

def Bresenham(x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    error = dx - dy

    points = []

    while x1 != x2 or y1 != y2:
        points.append((x1, y1))
        e2 = 2 * error
        if e2 > -dy:
            error -= dy
            x1 += sx
        if e2 < dx:
            error += dx
            y1 += sy

    points.append((x2, y2))
    return points



def DrawRectangle(image, coords):
    for i in range(4):
        x1, y1, _ = coords[i]
        x2, y2, _ = coords[(i + 1) % 4]
        points = Bresenham(int(x1), int(y1), int(x2), int(y2))

        for x, y in points:
            if x >= 800:
                x -= 1
            if y >= 800:
                y -= 1
            image[-y, x] = [0, 0, 0]

def DrawTriangle(image, coords):
    for i in range(3):

        x1, y1, _ = coords[i]
        x2, y2, _ = coords[(i + 1) % 3]
        points = Bresenham(int(x1), int(y1), int(x2), int(y2))

        for x, y in points:
            if x >= 800:
                x -= 1
            if y >= 800:
                y -= 1
            image[-y, x] = [0, 0, 0]


# Барецентрические координаты
# (a, b, c) - барицентрические координаты
# XA = aBA + bCA
def GetBarycentricCoords(p, p1, p2, p3):
    matrix_A = np.array([
        [p1[0] - p3[0], p2[0] - p3[0]],
        [p1[1] - p3[1], p2[1] - p3[1]]
    ])
    vectorB = np.array([p[0] - p3[0], p[1] - p3[1]])

    try:
        solution = np.linalg.solve(matrix_A, vectorB)
    except np.linalg.LinAlgError:
        return 0, 0, 0

    c = 1.0 - solution.sum()
    return solution[0], solution[1], c

def GetZCoord(a, b, c, z0, z1, z2):
    z = a * z0 + b * z1 + c * z2
    return z

def GetBorder(v0, v1, v2):
    left = int(min(v0[0], v1[0], v2[0]))
    right = int(max(v0[0], v1[0], v2[0])) + 1
    bottom = int(min(v0[1], v1[1], v2[1]))
    top = int(max(v0[1], v1[1], v2[1])) + 1
    return left, right, bottom, top

def DrawEdge(image, v0, v1, v2, bright):
    left, right, bottom, top = GetBorder(v0, v1, v2)
    for x in range(left, right):
        for y in range(bottom, top):
            point = np.array([x, y, 0])
            a, b, c = GetBarycentricCoords(point, v0, v1, v2)
            if CheckBarycentricCoordsInTriangle(a, b, c):
                z_coordinate = GetZCoord(a, b, c, v0[2], v1[2], v2[2])
                if ZBufferChange(x, y, z_coordinate):
                    image[-y, x, 0] = bright
                    image[-y, x, 1] = bright
                    image[-y, x, 2] = bright

def DrawLightning(image, v0, v1, v2, vt0, vt1, vt2, vn0, vn1, vn2, texture, w, h, bright,
                  alpha, ia, id, i_s, ka, kd, ks, light_source, camera_source):
    left, right, bottom, top = GetBorder(v0, v1, v2)
    for x in range(left, right):
        for y in range(bottom, top):
            point = np.array([x, y, 0])
            a, b, c = GetBarycentricCoords(point, v0, v1, v2)
            if CheckBarycentricCoordsInTriangle(a, b, c):
                z_coordinate = GetZCoord(a, b, c, v0[2], v1[2], v2[2])
                if ZBufferChange(x, y, z_coordinate):
                    uv = GetUV(a, b, c, vt0, vt1, vt2)

                    texture_coord_y = w - 1 - int(uv[1] * w)
                    texture_coord_x = int(uv[0] * h)

                    normal = GetNormal(a, b, c, vn0, vn1, vn2)
                    normal = np.array(normal)

                    I = GetI(normal, alpha, ia, id, i_s, ka, kd, ks, light_source, point, camera_source)
                    I = CheckI(I)
                    global_bright = bright * (I / 255)
                    if global_bright > 255:
                        global_bright = 255
                    image[-y, x, 0] = texture[texture_coord_y, texture_coord_x, 0] * global_bright
                    image[-y, x, 1] = texture[texture_coord_y, texture_coord_x, 1] * global_bright
                    image[-y, x, 2] = texture[texture_coord_y, texture_coord_x, 2] * global_bright

def GetUV(a, b, c, vt0, vt1, vt2):
    return [a * vt0[0] + b * vt1[0] + c * vt2[0], a * vt0[1] + b * vt1[1] + c * vt2[1]]

def GetNormal(a, b, c, vn0, vn1, vn2):
    return [a * vn0[0] + b * vn1[0] + c * vn2[0], a * vn0[1] + b * vn1[1] + c * vn2[1],
            a * vn0[2] + b * vn1[2] + c * vn2[2]]

def ZBufferChange(x, y, z):
    global ZBuffer
    if z <= ZBuffer[x, y]:
        ZBuffer[x, y] = z
        return 1
    return 0

def CheckBarycentricCoordsInTriangle(a, b, c):
    if 0 <= a and 0 <= b and 0 <= c:
        return True
    else:
        return False


def DrawTexture(image, v0, v1, v2, vt0, vt1, vt2, texture, w, h, bright):
    xMin, xMax, yMin, yMax = GetBorder(v0, v1, v2)
    for x in range(xMin, xMax):
        for y in range(yMin, yMax):
            point = np.array([x, y])
            a, b, c = GetBarycentricCoords(point, v0, v1, v2)
            if CheckBarycentricCoordsInTriangle(a, b, c):
                zCoord = GetZCoord(a, b, c, v0[2], v1[2], v2[2])
                if ZBufferChange(x, y, zCoord):
                    uv = GetUV(a, b, c, vt0, vt1, vt2)
                    TextureCoordY = w - int(uv[1] * w)
                    TextureCoordX = int(uv[0] * h)
                    image[-y, x, 0] = texture[TextureCoordY, TextureCoordX, 0] * bright
                    image[-y, x, 1] = texture[TextureCoordY, TextureCoordX, 1] * bright
                    image[-y, x, 2] = texture[TextureCoordY, TextureCoordX, 2] * bright


def CheckI(I):
    if I < 0:
        return np.abs(I)
    else:
        return I
def GetR(L, N):
    R = 2 * np.dot(N, L) * N - L
    return R


def GetV(point, camera_source):
    return camera_source - point


def CalculateIS(R, V, ks, i_s, alpha):
    cos = np.dot(R, V) / (np.linalg.norm(R) * np.linalg.norm(V))
    return np.dot(ks * (np.sign(cos) * (np.abs(cos)) ** (alpha)), i_s)


def GetL(l_source, point):
    return l_source - point


def CalculateID(id, kd, N, L):
    cos = np.dot(N, L) / (np.linalg.norm(L) * np.linalg.norm(N))
    return np.dot(id, kd) * cos


def CalculateIA(ia, ka):
    return np.dot(ia, ka)


def GetI(N, alpha, ia, id, i_s, ka, kd, ks, light_source, point, camera_source):
    IA = CalculateIA(ia, ka)
    L = GetL(light_source, point)
    ID = CalculateID(id, kd, N, L)
    R = GetR(L, N)
    V = GetV(point, camera_source)
    IS = CalculateIS(R, V, ks, i_s, alpha)
    return IA + IS + ID

def ReorderVertices(vertexA, vertexB, vertexC):
    x_min_val = min(vertexA[0], vertexB[0], vertexC[0])

    if vertexA[0] == x_min_val:
        return vertexC, vertexA, vertexB
    elif vertexC[0] == x_min_val:
        return vertexB, vertexC, vertexA
    else:
        return vertexA, vertexB, vertexC

def BackFaceCulling(P, vertexA, vertexB, vertexC, coeff, color_bright):
    # Ставим вершину с минимальным x как смежную
    pointA, pointB, pointC = ReorderVertices(vertexA, vertexB, vertexC)

    normalVec = np.cross((pointB - pointA), (pointB - pointC))

    # Меняем порядок обхода
    if np.any(normalVec <= 0):
        pointA, pointB = pointB, pointA

    translationVec = (vertexA - P)
    normalVec = np.cross((pointB - pointA), (pointC - pointB))

    result = np.dot(translationVec, normalVec)
    result = result * coeff + color_bright

    return checkResult(result)

def checkResult(result):
    if result is not None and result >= 0:
        result *= 255
        return 255 if result > 255 else result
    else:
        return None


with open(file_path, 'r') as file:
    for line in file:
        if not line:
            continue
        # vn
        if line.startswith('vn '):
            data = line.split()
            vn.append([float(data[1]), float(data[2]), float(data[3])])
        # vt
        elif line.startswith('vt '):
            data = line.split()
            vt.append([float(data[1]), float(data[2])])
        # v
        elif line.startswith('v '):
            data = line.split()
            v.append([float(data[1]), float(data[2]), float(data[3])])
            # f
        elif line.startswith('f '):
            data = line.split()
            faceVertices = []
            for vertexData in data[1:]:
                vertexIndices = [int(index) - 1 if index else None for index in vertexData.split('/')]
                faceVertices.append(vertexIndices)
            f.append(faceVertices)

# Mo2w

xRotationAngle = 5
yRotationAngle = 15
zRotationAngle = 10


def DegreesToRadians(alpha):
    return (alpha * np.pi) / 180


cameraCoords = np.array([[1, 0, 0, 0],
                         [0, np.cos(DegreesToRadians(xRotationAngle)), -np.sin(DegreesToRadians(xRotationAngle)), 0],
                         [0, np.sin(DegreesToRadians(xRotationAngle)), np.cos(DegreesToRadians(xRotationAngle)), 0],
                         [0, 0, 0, 1]])

cameraDirection = np.array([[np.cos(DegreesToRadians(yRotationAngle)), 0, np.sin(DegreesToRadians(yRotationAngle)), 0],
                            [0, 1, 0, 0],
                            [-np.sin(DegreesToRadians(yRotationAngle)), 0, np.cos(DegreesToRadians(yRotationAngle)), 0],
                            [0, 0, 0, 1]])

C = np.array([[np.cos(DegreesToRadians(zRotationAngle)), -np.sin(DegreesToRadians(zRotationAngle)), 0, 0],
              [np.sin(DegreesToRadians(zRotationAngle)), np.cos(DegreesToRadians(zRotationAngle)), 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

R = cameraCoords @ cameraDirection @ C

# Components of shift matrix

translationVector = [-2, 3, -2]

T = np.array([[1, 0, 0, translationVector[0]],
              [0, 1, 0, translationVector[1]],
              [0, 0, 1, translationVector[2]],
              [0, 0, 0, 1]])

scale = 0.8

S = np.array([[scale, 0, 0, 0],
              [0, scale, 0, 0],
              [0, 0, scale, 0],
              [0, 0, 0, 1]])

Mo2w = R @ T @ S

for i in range(0, len(v)):
    v[i] = (Mo2w @ np.hstack([v[i], 1]))[:3]

for i in range(0, len(vn)):
    vn[i] = (np.linalg.inv(Mo2w.T) @ np.hstack([vn[i], 1]))[:3]

cameraCoords = np.array([2, 2, 2])

cameraDirection = np.array([-2, -2, 0])

Tc = np.array([[1, 0, 0, -cameraCoords[0]],
               [0, 1, 0, -cameraCoords[1]],
               [0, 0, 1, -cameraCoords[2]],
               [0, 0, 0, 1]])

BA = cameraCoords - cameraDirection

lengthBA = np.sqrt(BA[0] ** 2 + BA[1] ** 2 + BA[2] ** 2)

BANorm = BA / lengthBA


gamma = BANorm

beta = np.array([0, 1, 0]) - gamma[1] * gamma

alpha = np.cross(beta, gamma)

Rc = np.array([[alpha.T[0], beta.T[0], gamma.T[0], 0],
               [alpha.T[1], beta.T[1], gamma.T[1], 0],
               [alpha.T[2], beta.T[2], gamma.T[2], 0],
               [0, 0, 0, 1]])

Mw2c = Rc @ Tc

for i in range(0, len(v)):
    v[i] = (Mw2c @ np.hstack([v[i], 1]))[:3]

for i in range(0, len(vn)):
    vn[i] = (np.linalg.inv(Mw2c.T) @ np.hstack([vn[i], 1]))[:3]

# Mproj
l = min(v[i][0] for i in range(len(v)))
r = max(v[i][0] for i in range(len(v)))
t = max(v[i][1] for i in range(len(v)))
b = min(v[i][1] for i in range(len(v)))
n = min(v[i][2] for i in range(len(v)))
f_p = max(v[i][2] for i in range(len(v)))

Mproj = np.array([[2 / (r - l), 0, 0, -((r + l) / (r - l))],
                  [0, 2 / (t - b), 0, -((t + b) / (t - b))],
                  [0, 0, -2 / (f_p - n), -((f_p + n) / (f_p - n))],
                  [0, 0, 0, 1]])


for i in range(0, len(v)):
    v[i] = (Mproj @ np.hstack([v[i], 1]))[:3]

for i in range(0, len(vn)):
    vn[i] = (np.linalg.inv(Mproj.T) @ np.hstack([vn[i], 1]))[:3]

x = 0
y = 0
ox = x + width // 2
oy = y + height // 2

Tw = np.array([[1, 0, 0, ox],
               [0, 1, 0, oy],
               [0, 0, 1, 1],
               [0, 0, 0, 1]])

Sw = np.array([[width // 2, 0, 0, 0],
               [0, height // 2, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])

Mviewport = Tw @ Sw


for i in range(0, len(v)):
    v[i] = (Mviewport @ np.hstack([v[i], 1]))[:3]

for i in range(0, len(vn)):
    vn[i] = (np.linalg.inv(Mviewport.T) @ np.hstack([vn[i], 1]))[:3]

# 1. Wireframe
# backgroundColor = np.array([255, 255, 255])
# img1 = np.full((width, height, 3), backgroundColor)
#
# for edge in f:
#     if len(edge) == 3:
#         drawnVertices = [edge[0][0], edge[1][0], edge[2][0]]
#         coords = [v[drawnVertices[0]], v[drawnVertices[1]],
#                   v[drawnVertices[2]]]
#         DrawTriangle(img1, coords)
#     elif len(edge) == 4:
#         drawnVertices = [edge[0][0], edge[1][0], edge[2][0], edge[3][0]]
#         coords = [v[drawnVertices[0]], v[drawnVertices[1]],
#                   v[drawnVertices[2]], v[drawnVertices[3]]]
#         DrawRectangle(img1, coords)
#
# plt.figure(figsize=(8, 8), dpi=100)
# plt.imshow(img1)
# plt.axis('off')
# plt.tight_layout(pad=0)
# plt.savefig('skullWireframe.png')
# plt.show()

# 2. Модель с гранями
# backgroundColor = np.array([255, 255, 255])
# img2 = np.full((height, width, 3), backgroundColor, np.uint8)
# P = np.array([2, 2, 2])
# coeff = 1 / 4000
# colorBright = 0.1
# for i in range(0, len(f)):
#     currentFace = f[i]
#
#     if len(currentFace) == 3:
#         v0 = v[currentFace[0][0]]
#         v1 = v[currentFace[1][0]]
#         v2 = v[currentFace[2][0]]
#         intencity = BackFaceCulling(P, v0, v1, v2, coeff, colorBright)
#         if intencity != None:
#             DrawEdge(img2, v0, v1, v2, intencity)
#
#     elif len(currentFace) == 4:
#         v0 = v[currentFace[0][0]]
#         v1 = v[currentFace[1][0]]
#         v2 = v[currentFace[2][0]]
#         v3 = v[currentFace[3][0]]
#         intencity = BackFaceCulling(P, v0, v1, v2, coeff, colorBright)
#         if intencity != None:
#             DrawEdge(img2, v0, v1, v2, intencity)
#             DrawEdge(img2, v2, v3, v0, intencity)
#
# plt.figure(figsize=(8, 8), dpi=100)
# plt.imshow(img2)
# plt.axis('off')
# plt.tight_layout(pad=0)
# plt.savefig('skullGrayscale.png')
# plt.show()

# 3. Модель с текстурами
backgroundColor = np.array([255, 255, 255])
image_3 = np.full((height, width, 3), backgroundColor, np.uint8)
P = np.array([2, 2, 2])
coeff = 1 / 8000
colorBright = 0.5
texture = matplotlib.image.imread('Skull.png')
textureWidth = texture.shape[0]
textureHeight = texture.shape[1]

for i in range(0, len(f)):
    currentFace = f[i]

    if len(currentFace) == 3:
        v0 = v[currentFace[0][0]]
        v1 = v[currentFace[1][0]]
        v2 = v[currentFace[2][0]]

        vt0 = vt[currentFace[0][1]]
        vt1 = vt[currentFace[1][1]]
        vt2 = vt[currentFace[2][1]]

        intencity = BackFaceCulling(P, v0, v1, v2, coeff, colorBright)
        if intencity != None:
            DrawTexture(image_3, v0, v1, v2, vt0, vt1, vt2, texture, textureWidth, textureHeight, intencity)

    elif len(currentFace) == 4:
        v0 = v[currentFace[0][0]]
        v1 = v[currentFace[1][0]]
        v2 = v[currentFace[2][0]]
        v3 = v[currentFace[3][0]]

        vt0 = vt[currentFace[0][1]]
        vt1 = vt[currentFace[1][1]]
        vt2 = vt[currentFace[2][1]]
        vt3 = vt[currentFace[3][1]]

        intencity = BackFaceCulling(P, v0, v1, v2, coeff, colorBright)
        if intencity != None:
            DrawTexture(image_3, v0, v1, v2, vt0, vt1, vt2, texture, textureWidth, textureHeight, intencity)
            DrawTexture(image_3, v2, v3, v0, vt2, vt3, vt0, texture, textureWidth, textureHeight, intencity)

plt.figure(figsize=(8, 8), dpi=100)
plt.imshow(image_3)
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig('skullTextured.png')
plt.show()


# 4. Анимация. Освещение. Модель Фонга
# coeff1 = 1 / 7000
# colorBright1 = 0.4
# def CheckParams():
#     global ia, id, i_s, ka, kd, ks
#     # ia
#     if ia[0] == 10:
#         ia[0] = 200
#     if ia[1] == 10:
#         ia[1] = 200
#     if ia[2] == 10:
#         ia[2] = 200
#     # id
#     if id[0] == 10:
#         id[0] = 200
#     if id[1] == 10:
#         id[1] = 200
#     if id[2] == 10:
#         id[2] = 200
#     # is
#     if i_s[0] == 10:
#         i_s[0] = 200
#     if i_s[1] == 10:
#         i_s[1] = 200
#     if i_s[2] == 10:
#         i_s[2] = 200
#
#     # ka
#     if ka[0] > 1:
#         ka[0] = 0
#     if ka[1] > 1:
#         ka[1] = 0
#     if ka[2] > 1:
#         ka[2] = 0
#     # kd
#     if kd[0] > 1:
#         kd[0] = 0
#     if kd[1] > 1:
#         kd[1] = 0
#     if kd[2] > 1:
#         kd[2] = 0
#     # ks
#     if ks[0] > 1:
#         ks[0] = 0
#     if ks[1] > 1:
#         ks[1] = 0
#     if ks[2] > 1:
#         ks[2] = 0
#
#
# def ChangeParams():
#     global ia, id, i_s, ka, kd, ks, alpha
#     unitVector = np.array([1, 1, 1])
#     vectorK = np.array([0.05, 0.05, 0.05])
#     ia -= unitVector
#     id -= unitVector
#     i_s -= unitVector
#
#     ka += vectorK
#     kd += vectorK
#     ks += vectorK
#
#     alpha -= 0.01
#
#
# backgroundColor = np.array([255, 255, 255])
# img4 = np.full((height, width, 3), backgroundColor)
#
# # Освещение
#
# # Фоновое
# ia = np.array([30, 30, 30])
# ka = np.array([0.7, 0.7, 0.7])
#
# # Диффузорное
# lightSourсe = np.array([-3, 3, 5])
# cameraSource = np.array([2, 2, 2])
# id = np.array([160, 140, 150])
# kd = np.array([0.8, 1, 0.6])
#
# # Зеркальное
# i_s = np.array([200, 120, 170])
# ks = np.array([0.2, 0.3, 0.7])
# alpha = 1
#
# fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
#
#
# def update(frame):
#     global img4, backgroundColor, ia, id, i_s, ka, kd, ks, alpha, lightSourсe, cameraSource
#     img4 = np.full((height, width, 3), backgroundColor)
#     CheckParams()
#
#     for i in range(0, len(f)):
#         current_face = f[i]
#
#         if len(current_face) == 3:
#             v0 = v[current_face[0][0]]
#             v1 = v[current_face[1][0]]
#             v2 = v[current_face[2][0]]
#
#             vt0 = vt[current_face[0][1]]
#             vt1 = vt[current_face[1][1]]
#             vt2 = vt[current_face[2][1]]
#
#             vn0 = vn[current_face[0][2]]
#             vn1 = vn[current_face[1][2]]
#             vn2 = vn[current_face[2][2]]
#
#             intencity = BackFaceCulling(P, v0, v1, v2, coeff1, colorBright1)
#             if intencity != None:
#                 DrawLightning(img4, v0, v1, v2, vt0, vt1, vt2, vn0, vn1, vn2, texture, textureWidth, textureHeight, intencity,
#                               alpha, ia, id, i_s, ka, kd, ks, lightSourсe, cameraSource)
#
#         elif len(current_face) == 4:
#             v0 = v[current_face[0][0]]
#             v1 = v[current_face[1][0]]
#             v2 = v[current_face[2][0]]
#             v3 = v[current_face[3][0]]
#
#             vt0 = vt[current_face[0][1]]
#             vt1 = vt[current_face[1][1]]
#             vt2 = vt[current_face[2][1]]
#             vt3 = vt[current_face[3][1]]
#
#             vn0 = vn[current_face[0][2]]
#             vn1 = vn[current_face[1][2]]
#             vn2 = vn[current_face[2][2]]
#             vn3 = vn[current_face[2][2]]
#
#             intencity = BackFaceCulling(P, v0, v1, v2, coeff1, colorBright1)
#             if intencity != None:
#                 DrawLightning(img4, v0, v1, v2, vt0, vt1, vt2, vn0, vn1, vn2, texture, textureWidth, textureHeight, intencity,
#                               alpha, ia, id, i_s, ka, kd, ks, lightSourсe, cameraSource)
#                 DrawLightning(img4, v2, v3, v0, vt2, vt3, vt0, vn2, vn3, vn0, texture, textureWidth, textureHeight, intencity,
#                               alpha, ia, id, i_s, ka, kd, ks, lightSourсe, cameraSource)
#
#     ChangeParams()
#
#     ax.imshow(img4)
#     ax.axis('off')
#     plt.draw()
#
#     plt.show()
#
#     print("Кадр сделан")
#
#
# animation_frames = 100
# ani = FuncAnimation(fig, update, frames=animation_frames, repeat=False)
#
# # Сохранение анимации в формате GIF
# writer = PillowWriter(fps=10)
# ani.save("SkullPhongAnimation.gif", writer=writer)

# plt.show()
