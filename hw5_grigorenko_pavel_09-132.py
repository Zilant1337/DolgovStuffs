import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mplimg
from matplotlib.animation import FuncAnimation, PillowWriter
import time

width, height = 800, 800
x_l = 0
y_d = 0
cameraCoords = np.array([2, 2, 2])
cameraDirectionCoords = np.array([-2, -2, 0])
xRotationAngle = np.radians(5)
yRotationAngle = np.radians(15)
zRotationAngle = np.radians(10)
translateMatrixCoords = (-2, 3, -2)
scale = 0.8

ia = np.array([30, 30, 30])
ka = np.array([0.7, 0.7, 0.7])

id = np.array([160, 140, 150])
kd = np.array([0.8, 1, 0.6])

iS = np.array([200, 120, 170])
ks = np.array([0.2, 0.3, 0.7])
alfa = 1
lightSourceCoords = np.array([-3, 3, 5])
lightSourceCoords1 = np.array([-3, 3, 5])
lightSourceCoords2 = np.array([100, -15, -3])


def TranslateLocalToWorldCoords():
    global vertices
    global normals

    vertices = np.array(vertices).T

    Rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(xRotationAngle), -np.sin(xRotationAngle), 0],
                   [0, np.sin(xRotationAngle), np.cos(xRotationAngle), 0],
                   [0, 0, 0, 1]])

    Ry = np.array([[np.cos(yRotationAngle), 0, np.sin(yRotationAngle), 0],
                   [0, 1, 0, 0],
                   [-np.sin(yRotationAngle), 0, np.cos(yRotationAngle), 0],
                   [0, 0, 0, 1]])

    Rz = np.array([[np.cos(zRotationAngle), -np.sin(zRotationAngle), 0, 0],
                   [np.sin(zRotationAngle), np.cos(zRotationAngle), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    R = Rx @ Ry @ Rz
    T = np.array([[1, 0, 0, translateMatrixCoords[0]],
                  [0, 1, 0, translateMatrixCoords[1]],
                  [0, 0, 1, translateMatrixCoords[2]],
                  [0, 0, 0, 1]])

    S = np.array([[scale, 0, 0, 0],
                  [0, scale, 0, 0],
                  [0, 0, scale, 0],
                  [0, 0, 0, 1]])

    Mo2w = R @ T @ S

    vertices = Mo2w @ ToProjCoords(vertices)

    Mo2w_T = Mo2w.T
    for i in range(len(normals)):
        vn_i_extended = np.append(normals[i], 1)
        normals[i] = np.dot(np.linalg.inv(Mo2w_T), vn_i_extended)[:3]


def TranslateWorldToCameraCoords():
    global vertices
    global normals

    Tc = np.array([[1, 0, 0, -cameraCoords[0]],
                   [0, 1, 0, -cameraCoords[1]],
                   [0, 0, 1, -cameraCoords[2]],
                   [0, 0, 0, 1]])

    gamma = np.array([cameraCoords[0] - cameraDirectionCoords[0], cameraCoords[1] - cameraDirectionCoords[1], cameraCoords[2] - cameraDirectionCoords[2]])
    gamma = gamma / np.linalg.norm(gamma)
    beta = np.array([0, 1, 0]) - gamma[1] * gamma
    alpha = np.cross(beta, gamma)
    Rc = np.eye(4)
    Rc[:3, :3] = np.column_stack((alpha, beta, gamma))

    Mw2c = Rc @ Tc

    vertices = Mw2c @ vertices

    Mw2c_T = Mw2c.T
    for i in range(len(normals)):
        vn_i_extended = np.append(normals[i], 1)
        normals[i] = np.dot(np.linalg.inv(Mw2c_T), vn_i_extended)[:3]


def TranslateCameraToNormalizedCoords():
    global vertices
    global normals

    l = min(vertices[0])
    r = max(vertices[0])
    b = min(vertices[1])
    t = max(vertices[1])
    n = min(vertices[2])
    f = max(vertices[2])

    Mproj = np.array([[2 / (r - l), 0, 0, - (r + l) / (r - l)],
                      [0, 2 / (t - b), 0, - (t + b) / (t - b)],
                      [0, 0, - 2 / (f - n), - (f + n) / (f - n)],
                      [0, 0, 0, 1]])

    vertices = Mproj @ vertices

    Mproj_T = Mproj.T
    for i in range(len(normals)):
        vn_i_extended = np.append(normals[i], 1)
        normals[i] = np.dot(np.linalg.inv(Mproj_T), vn_i_extended)[:3]


def TranslateNormalizedToWindowCoords():
    global vertices
    global normals

    Tw = np.array([[1, 0, 0, x_l + width / 2],
                  [0, 1, 0, y_d + height / 2],
                  [0, 0, 1, 1],
                  [0, 0, 0, 1]])

    Sw = np.array([[width / 2, 0, 0, 0],
                  [0, height / 2, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    Mviewport = Tw @ Sw

    vertices = Mviewport @ vertices
    vertices = to_cart_coords(vertices)
    vertices = np.array(vertices).T

    Mviewport_T = Mviewport.T
    for i in range(len(normals)):
        vn_i_extended = np.append(normals[i], 1)
        normals[i] = np.dot(np.linalg.inv(Mviewport_T), vn_i_extended)[:3]

def to_cart_coords(x):
    x = x[:-1] / x[-1]
    return x

def ToProjCoords(x):
    r, c = x.shape
    x = np.concatenate([x, np.ones((1, c))], axis=0)
    return x

def BresenhamAlgorithm(image, x0, y0, x1, y1, color=(0, 0, 0)):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    err = dx - dy

    x, y = x0, y0
    while True:
        if 0 <= x < width and 0 <= y < height:
                image[y, x] = color

        if x == x1 and y == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def BackFaceCulling(edge):
    v0 = vertices[edge[0] - 1]
    v1 = vertices[edge[1] - 1]
    v2 = vertices[edge[2] - 1]
    N = np.cross(v1 - v0, v2 - v0)
    res1 = np.dot((v0 - cameraCoords), N)
    N = np.cross(v2 - v0, v1 - v0)
    res2 = np.dot((v0 - cameraCoords), N)
    if res1 < 0:
        return res1 / 1000
    else:
        return res2 / 1000


def EdgeRasterization(image, edge, color):
    v0 = vertices[edge[0] - 1]
    v1 = vertices[edge[1] - 1]
    v2 = vertices[edge[2] - 1]

    x_min, x_max = map(int, (np.floor(min(v0[0], v1[0], v2[0])), np.ceil(max(v0[0], v1[0], v2[0]))))
    y_min, y_max = map(int, (np.floor(min(v0[1], v1[1], v2[1])), np.ceil(max(v0[1], v1[1], v2[1]))))

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            p = np.array([x, y, 0])
            a, b, c = BarycentricCoords(p, v0, v1, v2)
            if a >= 0 and b >= 0 and c >= 0:
                z = a * v0[2] + b * v1[2] + c * v2[2]
                if ZBufferValue(x, y, z):
                    image[x, y] = color


def FaceTexturing(image, face):
    v0 = vertices[face[0] - 1]
    v1 = vertices[face[1] - 1]
    v2 = vertices[face[2] - 1]

    index = edges.index(face)
    vt0 = textures[edgesTex[index][0] - 1]
    vt1 = textures[edgesTex[index][1] - 1]
    vt2 = textures[edgesTex[index][2] - 1]

    vt0 = np.array(vt0)
    vt1 = np.array(vt1)
    vt2 = np.array(vt2)

    x_min, x_max = map(int, (np.floor(min(v0[0], v1[0], v2[0])), np.ceil(max(v0[0], v1[0], v2[0]))))
    y_min, y_max = map(int, (np.floor(min(v0[1], v1[1], v2[1])), np.ceil(max(v0[1], v1[1], v2[1]))))

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            p = np.array([x, y, 0])
            a, b, c = BarycentricCoords(p, v0, v1, v2)
            if a >= 0 and b >= 0 and c >= 0:
                z = a * v0[2] + b * v1[2] + c * v2[2]
                if ZBufferValue(x, y, z):
                    txtr = a * vt0 + b * vt1 + c * vt2
                    i = w - 1 - int(txtr[1] * w)
                    j = int(txtr[0] * h) - 1
                    image[x, y] = texture[i, j]


def PhongLighting(image, face):
    v0 = vertices[face[0] - 1]
    v1 = vertices[face[1] - 1]
    v2 = vertices[face[2] - 1]

    index = edges.index(face)
    vn0 = normals[edgesNorm[index][0] - 1]
    vn1 = normals[edgesNorm[index][1] - 1]
    vn2 = normals[edgesNorm[index][2] - 1]

    xMin, xMax = map(int, (np.floor(min(v0[0], v1[0], v2[0])), np.ceil(max(v0[0], v1[0], v2[0]))))
    yMin, yMax = map(int, (np.floor(min(v0[1], v1[1], v2[1])), np.ceil(max(v0[1], v1[1], v2[1]))))

    for x in range(xMin, xMax + 1):
        for y in range(yMin, yMax + 1):
            p = np.array([x, y, 0])
            a, b, c = BarycentricCoords(p, v0, v1, v2)
            if a >= 0 and b >= 0 and c >= 0:
                z = a * v0[2] + b * v1[2] + c * v2[2]
                if ZBufferValue(x, y, z):
                    la = np.dot(ia, ka)

                    N = a * vn0 + b * vn1 + c * vn2
                    L = lightSourceCoords - p
                    cos_LN = np.dot(L, N) / (np.linalg.norm(L) * np.linalg.norm(N))
                    ld = np.dot(id, kd) * cos_LN

                    R = 2 * np.dot(N, L) * N - L
                    V = cameraCoords - p
                    cos_RV = np.dot(R, V) / (np.linalg.norm(R) * np.linalg.norm(V))
                    ls = np.dot(ks, iS) * (cos_RV ** alfa)

                    l = np.abs(la + ld + ls)

                    color = imageArray[x, y] * np.clip(l / 255, 0, 255)
                    image[x, y] = color


def LambertLighting1(image, face):
    v0 = vertices[face[0] - 1]
    v1 = vertices[face[1] - 1]
    v2 = vertices[face[2] - 1]

    index = edges.index(face)
    vn0 = normals[edgesNorm[index][0] - 1]
    vn1 = normals[edgesNorm[index][1] - 1]
    vn2 = normals[edgesNorm[index][2] - 1]

    x_min, x_max = map(int, (np.floor(min(v0[0], v1[0], v2[0])), np.ceil(max(v0[0], v1[0], v2[0]))))
    y_min, y_max = map(int, (np.floor(min(v0[1], v1[1], v2[1])), np.ceil(max(v0[1], v1[1], v2[1]))))
    N = None

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            p = np.array([x, y, 0])
            a, b, c = BarycentricCoords(p, v0, v1, v2)
            if a >= 0 and b >= 0 and c >= 0:
                z = a * v0[2] + b * v1[2] + c * v2[2]
                if ZBufferValue(x, y, z):
                    if N is None:
                        N = a * vn0 + b * vn1 + c * vn2
                    la = np.dot(ia, ka)

                    L = lightSourceCoords1 - p
                    cos_LN = np.dot(L, N) / (np.linalg.norm(L) * np.linalg.norm(N))
                    ld = np.dot(id, kd) * cos_LN

                    l = np.abs(la + ld)

                    color = np.array([255, 255, 255]) * np.clip(l / 255, 0, 255)
                    image[x, y] = color


def LambertLighting2(image, face):
    v0 = vertices[face[0] - 1]
    v1 = vertices[face[1] - 1]
    v2 = vertices[face[2] - 1]

    index = edges.index(face)
    vn0 = normals[edgesNorm[index][0] - 1]
    vn1 = normals[edgesNorm[index][1] - 1]
    vn2 = normals[edgesNorm[index][2] - 1]

    x_min, x_max = map(int, (np.floor(min(v0[0], v1[0], v2[0])), np.ceil(max(v0[0], v1[0], v2[0]))))
    y_min, y_max = map(int, (np.floor(min(v0[1], v1[1], v2[1])), np.ceil(max(v0[1], v1[1], v2[1]))))
    N = None

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            p = np.array([x, y, 0])
            a, b, c = BarycentricCoords(p, v0, v1, v2)
            if a >= 0 and b >= 0 and c >= 0:
                z = a * v0[2] + b * v1[2] + c * v2[2]
                if ZBufferValue(x, y, z):
                    if N is None:
                        N = a * vn0 + b * vn1 + c * vn2
                    la = np.dot(ia, ka)

                    L1 = lightSourceCoords1 - p
                    cos_LN1 = np.dot(L1, N) / (np.linalg.norm(L1) * np.linalg.norm(N))
                    ld1 = np.dot(id, kd) * cos_LN1

                    L2 = lightSourceCoords2 - p
                    cos_LN2 = np.dot(L2, N) / (np.linalg.norm(L2) * np.linalg.norm(N))
                    ld2 = np.dot(id, kd) * cos_LN2

                    l = np.abs(la + ld1 + ld2)

                    color = np.array([255, 255, 255]) * np.clip(l / 255, 0, 255)
                    image[x, y] = color


def ColorGrayscale(image):
    first_color = np.array([255, 0, 0], dtype=np.uint8)
    second_color = np.array([255, 255, 0], dtype=np.uint8)

    for x in range(width):
        alpha = x / (width - 1)
        current_color = (1 - alpha) * first_color + alpha * second_color
        for y in range(height):
            pixel_color = image[y, x, :]
            if not np.array_equal(pixel_color, [255, 255, 255]):
                image[y, x, 0] = int(pixel_color[0] + (current_color[0] - pixel_color[0]) * 0.5)
                image[y, x, 1] = int(pixel_color[1] + (current_color[1] - pixel_color[1]) * 0.5)
                image[y, x, 2] = int(pixel_color[2] + (current_color[2] - pixel_color[2]) * 0.5)


def BarycentricCoords(p, p1, p2, p3):
    det_t = (p2[1] - p3[1]) * (p1[0] - p3[0]) + (p3[0] - p2[0]) * (p1[1] - p3[1])
    a = ((p2[1] - p3[1]) * (p[0] - p3[0]) + (p3[0] - p2[0]) * (p[1] - p3[1])) / det_t
    b = ((p3[1] - p1[1]) * (p[0] - p3[0]) + (p1[0] - p3[0]) * (p[1] - p3[1])) / det_t
    c = 1.0 - a - b
    return a, b, c


def ZBufferValue(x, y, z):
    if z <= zBuffer[x, y]:
        zBuffer[x, y] = z
        return True
    return False


def SaveImage(img, name):
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    title = name + '.png'
    plt.imsave(title, img, format='png')
    plt.show()


def WireFrameModel():
    img = np.ones((height, width, 3), dtype=np.uint8) * 0

    for edge in edges:
        x1 = int(vertices[edge[0] - 1][0])
        y1 = int(vertices[edge[0] - 1][1])
        x2 = int(vertices[edge[1] - 1][0])
        y2 = int(vertices[edge[1] - 1][1])
        x3 = int(vertices[edge[2] - 1][0])
        y3 = int(vertices[edge[2] - 1][1])

        BresenhamAlgorithm(img, x1, y1, x2, y2, [255, 255, 255])
        BresenhamAlgorithm(img, x2, y2, x3, y3, [255, 255, 255])
        BresenhamAlgorithm(img, x3, y3, x1, y1, [255, 255, 255])

    first_color = np.array([0, 0, 205], dtype=np.uint8)
    second_color = np.array([0, 250, 154], dtype=np.uint8)

    for x in range(width):
        alpha = x / (width - 1)
        current_color = (1 - alpha) * first_color + alpha * second_color
        for y in range(height):
            if not np.array_equal(img[y, x], [0, 0, 0]):
                img[y, x] = current_color

    img = np.rot90(img)
    SaveImage(img, 'skull_res_1')


def GrayscaleModel():
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    for edge in edges:
        c = backFaces[edges.index(edge)]
        if c < 0:
            c = np.clip(np.abs(c) * 255, 0, 255)
            color = np.array([c, c, c], dtype=np.uint8)
            EdgeRasterization(img, edge, color)

    img = img[::-1, :]
    SaveImage(img, 'skull_res_2')


def TexturedModel():
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    for edge in edges:
        c = BackFaceCulling(edge)
        if c < 0:
            FaceTexturing(img, edge)

    img = img[::-1, :]
    SaveImage(img, 'skull_res_3')


def PhongLightingModelAnimation():
    fig = plt.figure()
    ani = FuncAnimation(fig, update, frames=100)
    writer = PillowWriter()
    ani.save('skull_animation.gif', writer=writer)
    plt.show()

def update():
    global ia, id, iS, ka, kd, ks, alfa
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    for edge in edges:
        c = backFaces[edges.index(edge)]
        if c < 0:
            PhongLighting(img, edge)
    img = img[::-1, :]
    plt.imshow(img)
    plt.axis('off')

    ia -= 1
    id -= 1
    iS -= 1

    if 10 in ia:
        ia = np.array([200, 200, 200])
    if 10 in id:
        id = np.array([200, 200, 200])
    if 10 in iS:
        iS = np.array([200, 200, 200])
    # ia[ia == 10] = 200
    # id[id == 10] = 200
    # iS[iS == 10] = 200

    ka += 0.05
    kd += 0.05
    ks += 0.05

    ka[ka > 1] = 0
    kd[kd > 1] = 0
    ks[ks > 1] = 0

    alfa -= 0.01

zBuffer = np.ones((width, height))

with open("skull.obj", "r") as file:
    vertices = []
    edges = []
    edgesNorm = []
    edgesTex = []
    normals = []
    textures = []
    backFaces = []

    for line in file:
        words = line.split()

        if not line.strip():
            continue

        elif words[0] == "v":
            vertex = [float(words[1]), float(words[2]), float(words[3])]
            vertices.append(vertex)

        elif words[0] == "f":
            edge = []
            tex = []
            norm = []
            for word in words[1:4]:
                word1 = word.split("/")[0]
                word2 = word.split("/")[1]
                word3 = word.split("/")[2]
                edge.append(int(word1))
                tex.append(int(word2))
                norm.append(int(word3))
            edges.append(edge)
            edgesTex.append(tex)
            edgesNorm.append(norm)
            if len(words) == 5:
                edge = edge.copy()
                tex = tex.copy()
                norm = norm.copy()
                edge.pop(1)
                tex.pop(1)
                norm.pop(1)
                word1 = words[4].split("/")[0]
                word2 = words[4].split("/")[1]
                word3 = words[4].split("/")[2]
                edge.append(int(word1))
                tex.append(int(word2))
                norm.append(int(word3))
                edges.append(edge)
                edgesTex.append(tex)
                edgesNorm.append(norm)

        elif words[0] == "vn":
            normal = [float(words[1]), float(words[2]), float(words[3])]
            normals.append(normal)

        elif words[0] == "vt":
            texture = [float(words[1]), float(words[2]), float(words[3])]
            textures.append(texture)


TranslateLocalToWorldCoords()
TranslateWorldToCameraCoords()
TranslateCameraToNormalizedCoords()
TranslateNormalizedToWindowCoords()

for edge in edges:
    back = BackFaceCulling(edge)
    backFaces.append(back)

texture = mplimg.imread('Skull.jpg')
w = texture.shape[0]
h = texture.shape[1]

# imagePath = 'skull31.png'
# image = plt.imread(imagePath)
# height, width, _ = image.shape
# pixelMatrix = [image[y, x, :3] for y in range(height) for x in range(width)]
# imageArray = np.array(pixelMatrix, dtype=np.uint8)
# imageArray = imageArray[::-1, :]

startTime = time.time()

# WireFrameModel()  # 3.125
# GrayscaleModel()  # 175.679
# TexturedModel()  # 229.62
# PhongLightingModelAnimation()  # 7064.271

end_time = time.time()
execution_time = end_time - startTime
print(execution_time)