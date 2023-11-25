# 4. Построить анимацию замкнутой кривой, образующей знак бесконечность, с помощью кривых Безье.
# 
# В процессе анимации контрольные точки движутся следующим образом (часть 1):
# - растягивается левая и правая часть каждой из «окружностей» так, что модель становится растянутой по горизонтали. Растяжение проходит на величину равную 1,5 M, где M – разность X координат крайней левой и крайней правой точек «окружностей».
# - стягивается по вертикали верхняя и нижняя половины «окружностей» модели до касания друг друга.
# 
# После завершения первой части анимация во второй части идет следующим образом:
# - стягивается левая и правая часть каждой из «окружностей» так, что модель становится стянутой по горизонтали. Стяжение проходит на величину равную 0,5 M, где M – разность X координат крайней левой и крайней правой точек «окружностей».
# - растягивается по вертикали верхняя и нижняя половины «окружностей» модели до касания друг друга на величину 1.5 Т, где T – разность Y координат крайней верхней и крайней правой точек «окружностей».
# 
# Процесс стягивания проходит за 3 секунды, а растяжения за 2 секунды.
# 
# Значение параметров из задания определить самостоятельно. Для создания анимации воспользоваться функцией ArtistAnimation, либо FuncAnimation.

import numpy as np
from numpy import array as a
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Задаем начальные значения
X_CENTER = 3 # Центр бесконечности по X
Y_CENTER = 1 # Центр бесконечности по Y
M = 6.0  # Разность X координат крайней левой и крайней правой точек «окружностей»
T = 3.0  # Разность Y координат крайней верхней и крайней правой точек «окружностей»
duration_stretch = 2  # Длительность растяжения в секундах
duration_contract = 1  # Длительность стягивания в секундах
fps = 144  # Количество кадров в секунду
frames_stretch = duration_stretch * fps  # Количество кадров для растяжения
frames_contract = duration_contract * fps  # Количество кадров для стягивания
enableDebugBezier = True # вывод "строительной" линиии и точек для самих линий Безье в ролике
img = np.zeros((800, 800, 3))   
T_ST = 0.0 # фиксатор для стягивания

class Bezier():
    def TwoPoints(t, P1, P2):
        """
        Возвращает точку между P1 и P2, параметризованную значением t.
        входные:
            t float/int; параметризация.
            P1 массив numpy; точка.
            P2 массив numpy; точка.
        результат:
            Q1 массив numpy; точка.
        """

        if not isinstance(P1, np.ndarray) or not isinstance(P2, np.ndarray):
            raise TypeError('Точки должны быть экземпляром numpy.ndarray!')
        if not isinstance(t, (int, float)):
            raise TypeError('Параметр t должен быть int или float!')

        Q1 = (1 - t) * P1 + t * P2
        return Q1

    def Points(t, points):
        """
        Возвращает список точек, интерполированных процессом Безье.
        входные:
            t            float/int; параметризация.
            points       список массивов numpy; точки.
        результат:
            newpoints    список массивов numpy; точки.
        """
        newpoints = []
        #print("points =", points, "\n")
        for i1 in range(0, len(points) - 1):
            #print("i1 =", i1)
            #print("points[i1] =", points[i1])

            newpoints += [Bezier.TwoPoints(t, points[i1], points[i1 + 1])]
            #print("newpoints  =", newpoints, "\n")
        return newpoints

    def Point(t, points):
        """
        Возвращает точку, интерполированную процессом Безье
        входные:
            t            float/int; параметризация
            points       список массивов numpy; точки.
        результат:
            newpoint     массив numpy; точка.
        """
        newpoints = points
        #print("newpoints = ", newpoints)
        while len(newpoints) > 1:
            newpoints = Bezier.Points(t, newpoints)
            #print("newpoints in loop = ", newpoints)

        #print("newpoints = ", newpoints)
        #print("newpoints[0] = ", newpoints[0])
        return newpoints[0]

    def Curve(t_values, points):
        """
        Возвращает точки, интерполированные процессом Безье
        входные:
            t_values     list of floats/ints; параметризация
            points       список массивов numpy; точки.
        результат:
            curve        список массивов numpy; точки.
        """

        if not hasattr(t_values, '__iter__'):
            raise TypeError("`t_values` Должно быть повторяемым из целых чисел или чисел с плавающей запятой длиной больше 0.")
        if len(t_values) < 1:
            raise TypeError("`t_values` Должно быть повторяемым из целых чисел или чисел с плавающей запятой длиной больше 0.")
        if not isinstance(t_values[0], (int, float)):
            raise TypeError("`t_values` Должно быть повторяемым из целых чисел или чисел с плавающей запятой длиной больше 0.")

        curve = np.array([[0.0] * len(points[0])])
        for t in t_values:
            #print("curve                  \n", curve)
            #print("Bezier.Point(t, points) \n", Bezier.Point(t, points))

            curve = np.append(curve, [Bezier.Point(t, points)], axis=0)

            #print("curve after            \n", curve, "\n--- --- --- --- --- --- ")
        curve = np.delete(curve, 0, 0)
        #print("curve final            \n", curve, "\n--- --- --- --- --- --- ")
        return curve

# Функция для обновления графика на каждом кадре
def update(frame):
    plt.cla()  # Очищаем предыдущий график
    t_points = np.arange(0, 1, 0.001)

    # [135, 292], [60, 317], [12, 302], [7, 275], [19,242], [11, 164], [46, 112], [69, 104], [83, 96], [96, 81], [97, 63], [107,40], [125, 33], [137, 10], [143, 13], [145, 36], [166, 36], [168, 50], [179, 54], [179, 60], [183, 72], [159, 85], [159, 92], [167, 132], [145, 163], [138, 184], [138, 233], [158, 238], [160, 245], [143, 249], [160, 255], [146, 264], [122,250], [113, 221], [109, 241], [89, 250], [103, 249], [123, 255], [122, 266], [52, 269], [49, 276], [42, 283], [46, 292], [70, 297], [135, 292]

    plt.xlim(X_CENTER - M, X_CENTER + M)
    plt.ylim(Y_CENTER - T, Y_CENTER + T)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Вычисляем координаты точек на кривой Безье
    t = frame / frames_stretch                               # Прогресс анимации от 0 до 1 для растяжения
    t_contract = (frame - frames_stretch) / frames_contract  # Прогресс анимации от 0 до 1 для стягивания
    global T_ST
    if frame < frames_stretch:
        # Растяжение
        T_ST = t # Фиксатор положения для Стягивания.
        points1 = np.array([ [X_CENTER, Y_CENTER], [X_CENTER + M/2 * (1 + t), Y_CENTER + T/2 * (1 - t) ], [X_CENTER + M/2 * (1 + t), Y_CENTER - T/2 * (1 - t)], [X_CENTER, Y_CENTER]])
        points2 = np.array([ [X_CENTER, Y_CENTER], [X_CENTER - M/2 * (1 + t), Y_CENTER + T/2 * (1 - t) ], [X_CENTER - M/2 * (1 + t), Y_CENTER - T/2 * (1 - t)], [X_CENTER, Y_CENTER]])
    else:
        
        # Стягивание
        points1 = np.array([ [X_CENTER, Y_CENTER], [X_CENTER + M/2 * (1 + T_ST - t_contract), Y_CENTER + T/2 * (1 - T_ST + t_contract) ], [X_CENTER + M/2 * (1 + T_ST - t_contract), Y_CENTER - T/2 * (1 - T_ST + t_contract)], [X_CENTER, Y_CENTER]])
        points2 = np.array([ [X_CENTER, Y_CENTER], [X_CENTER - M/2 * (1 + T_ST - t_contract), Y_CENTER + T/2 * (1 - T_ST + t_contract) ], [X_CENTER - M/2 * (1 + T_ST - t_contract), Y_CENTER - T/2 * (1 - T_ST + t_contract)], [X_CENTER, Y_CENTER]])

    curve1 = Bezier.Curve(t_points, points1) # Генерация точек Безье на основе "проектируемных линий"
    curve2 = Bezier.Curve(t_points, points2) # Генерация точек Безье на основе "проектируемных линий"
    
    #global img
    #img = np.zeros((800, 800, 3))   
    #

    #
    #global_size = curve1.size // 2
    #for i in range(global_size - 1):
    #    bresenham(int(curve1[i][0]), int(curve1[i][1]), int(curve1[i+1][0]), int(curve1[i+1][1]))
    #
    #for i in range(global_size - 1):
    #    bresenham(int(curve2[i][0]), int(curve2[i][1]), int(curve2[i+1][0]), int(curve2[i+1][1]))
    #
    #plt.imshow(img)


    plt.plot(
        curve1[:, 0],   # x-coordinates.
        curve1[:, 1],   # y-coordinates.
        'b' 
    )
    if enableDebugBezier == True:
        plt.plot(
            points1[:, 0],  # x-coordinates.
            points1[:, 1],  # y-coordinates.
            'ro:'           # Styling (red, circles, dotted).
        )
    plt.plot(
        curve2[:, 0],    # x-coordinates.
        curve2[:, 1],    # y-coordinates.
        'b'
    )
    if enableDebugBezier == True:
        plt.plot(
            points2[:, 0],  # x-coordinates.
            points2[:, 1],  # y-coordinates.
            'o:'            # Styling (red, circles, dotted).
        )

# Функция окраски
def plot(x, y):
	img[x, y] = 1

# Алгоритм Брезенхема
def bresenham(x0, y0, x1, y1):
	dx = abs(x1 - x0)
	dy = -abs(y1 - y0)
	sx = 1 if x0 < x1 else -1
	sy = 1 if y0 < y1 else -1
	error = dx + dy
	
	while True:
		plot(x0, y0)
		if x0 == x1 and y0 == y1:
			break
		e2 = 2 * error
		if e2 >= dy:
			if x0 == x1:
				break
			error += dy
			x0 += sx
		if e2 <= dx:
			if y0 == y1:
				break
			error += dx
			y0 += sy


# Создаем анимацию
fig, ax = plt.subplots()
T_ST = 0.0
animation = FuncAnimation(fig, update, frames=frames_stretch+frames_contract, interval=1000/fps)
# Сохраняем анимацию в файл (необязательно)
animation.save('infinity_animation.mp4', writer='ffmpeg')
print(T_ST)
# Показываем анимацию
plt.show()
