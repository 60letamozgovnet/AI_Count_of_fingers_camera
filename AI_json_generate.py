import numpy as np  # Импортируем библиотеку NumPy для работы с массивами
import cv2  # Импортируем библиотеку OpenCV для работы с изображениями и видео
import json  # Импортируем библиотеку json для работы с JSON-файлами

# --- Глобальные переменные ---
background = None  # Переменная для хранения изображения фона
hand = None  # Переменная для хранения данных о руке (объект класса HandData)
frames_elapsed = 0  # Счетчик обработанных кадров
FRAME_HEIGHT = 500  # Высота кадра видео
FRAME_WIDTH = 650  # Ширина кадра видео
CALIBRATION_TIME = 150  # Количество кадров для калибровки фона
BG_WEIGHT = 0.5  # Вес для обновления фона (взвешенное среднее)
stable_frames = 0  # Счетчик стабильных кадров (не используется в полной мере в этой версии)
required_stable_frames = 20  # Требуемое количество стабильных кадров (не используется в полной мере в этой версии)

# --- Класс для кодирования NumPy-массивов в JSON ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Преобразует NumPy-массив в список для JSON
        return super().default(obj)  # Для других типов данных используется стандартный кодировщик

# --- Класс для хранения данных о руке ---
class HandData:
    def __init__(self, top, bottom, left, right, centerX): # Конструктор класса
        self.top = top  # Координата верхней границы руки
        self.bottom = bottom  # Координата нижней границы руки
        self.left = left  # Координата левой границы руки
        self.right = right  # Координата правой границы руки
        self.centerX = centerX  # Координата X центра руки
        self.prevCenterX = 0  # Предыдущая координата X центра руки
        self.isInFrame = False  # Флаг, указывающий, находится ли рука в кадре
        self.isWaving = False  # Флаг, указывающий, машет ли рука
        self.fingers = 0  # Количество пальцев (не определяется в этой версии)

    def update(self, top, bottom, left, right): # Обновляет координаты границ руки
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def check_for_waving(self, centerX): # Определяет, машет ли рука
        self.prevCenterX = self.centerX # Сохраняем текущую координату X в предыдущую
        self.centerX = centerX # Обновляем текущую координату X
        if abs(self.centerX - self.prevCenterX) > 5:  # Если разница между текущей и предыдущей координатой X больше 5 пикселей
            self.isWaving = True  # Значит, рука машет
        else:
            self.isWaving = False # Значит, рука не машет

# --- Функция для отображения текста и прямоугольника на кадре ---
def write_on_image(frame):
    text = "Searching..."  # Исходный текст для отображения

    if frames_elapsed < CALIBRATION_TIME: # Если еще не закончилась калибровка
        text = "Calibrating..." # Отображаем текст о калибровке
    elif hand is None or not hand.isInFrame: # Если рука не обнаружена или не в кадре
        text = "No hand detected" # Отображаем текст, что рука не обнаружена
    else:
        if hand.isWaving: # Если рука машет
            text = "Waving"  # Отображаем текст "машет"
        # elif hand.fingers == 0:
        #     text = "Rock"
        # elif hand.fingers == 1:
        #     text = "Pointing"
        # elif hand.fingers == 2:
        #     text = "Scissors"
        # elif hand.fingers == 3:
        #     text = 'Three fingers'
        else: # Если рука в кадре и не машет
            text = f"Fingers: {hand.fingers}"  # Отображаем количество пальцев (не реализовано в полной мере в этой версии)

    # Отображаем текст на кадре
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    # Рисуем прямоугольник вокруг области интереса (ROI)
    cv2.rectangle(frame, (region_left, region_top), (region_right, region_bottom), (255, 255, 255), 2)

# --- Функция для извлечения области интереса (ROI) и предобработки ---
def get_region(frame):
    region = frame[region_top:region_bottom, region_left:region_right]  # Вырезаем ROI
    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)  # Преобразуем в градации серого
    gray_region = cv2.GaussianBlur(gray_region, (7, 7), 0)  # Применяем размытие по Гауссу для уменьшения шума
    thresholded = cv2.adaptiveThreshold( # Применяем адаптивную пороговую обработку для выделения руки
        gray_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return thresholded  # Возвращаем предобработанную область

# --- Функция для обновления фона ---
def get_average(region):
    global background  # Используем глобальную переменную background
    if background is None:  # Если фон еще не инициализирован
        background = region.copy().astype("float")  # Копируем регион и преобразуем в float
    return cv2.accumulateWeighted(region, background, BG_WEIGHT)  # Обновляем фон с использованием взвешенного среднего

# --- Функция для сегментации руки ---
def segment(region):
    global hand  # Используем глобальную переменную hand
    if hand is None:  # Если объект HandData еще не создан
        hand = HandData((0, 0), (0, 0), (0, 0), (0, 0), 0)  # Создаем объект

    diff = cv2.absdiff(background.astype(np.uint8), region)  # Вычисляем разницу между текущим кадром и фоном
    _, thresholded_region = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)  # Применяем пороговую обработку

    kernel = np.ones((3, 3), np.uint8)  # Создаем ядро для морфологических операций
    thresholded_region = cv2.erode(thresholded_region, kernel, iterations=1)  # Применяем эрозию для уменьшения шума
    thresholded_region = cv2.dilate(thresholded_region, kernel, iterations=2)  # Применяем дилатацию для сглаживания контуров

    contours, _ = cv2.findContours(thresholded_region.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Ищем контуры

    if len(contours) == 0:  # Если контуры не найдены
        hand.isInFrame = False # Рука не в кадре
        return  # Выходим из функции
    else:
        segmented_region = max(contours, key=cv2.contourArea)  # Выбираем контур с наибольшей площадью (предполагаем, что это рука)
        if cv2.contourArea(segmented_region) < 500:  # Если площадь контура меньше заданного значения
            hand.isInFrame = False  # Считаем, что рука не в кадре
            return # Выходим из функции
        else:
            hand.isInFrame = True # Рука в кадре
            return (thresholded_region, segmented_region)  # Возвращаем бинарное изображение руки и контур

# --- Функция для получения данных о руке (вычисление разницы между фоном и текущим кадром) ---
def get_hand_data(region, background):
    bg = region.copy().astype("float")  # Копируем регион и преобразуем в float
    bg = cv2.accumulateWeighted(region, bg, BG_WEIGHT)  # Обновляем фон
    return (background / 255) - (bg / 255)  # Возвращаем разницу (в нормированном виде)

# --- Основной блок программы ---
region_top = 0  # Верхняя граница ROI
region_bottom = 300 # Нижняя граница ROI
region_left = 350  # Левая граница ROI
region_right = FRAME_WIDTH  # Правая граница ROI
capture = cv2.VideoCapture(0)  # Инициализируем захват видео с камеры (0 - индекс первой камеры)
count = 0 # Счетчик кадров
cnt_of_rows = 0 # Счетчик строк данных
with open(r'C:\Users\xezze\Desktop\data.json', 'w', newline='') as file:  # Открываем файл для записи JSON-данных
    data = {'data': [],
            'count_of_fingers': []} # Словарь для хранения данных
    while cnt_of_rows < 100: # Цикл выполняется пока не наберется 100 строк данных
        ret, frame = capture.read()  # Считываем кадр с камеры
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))  # Изменяем размер кадра
        frame = cv2.flip(frame, 1)  # Отражаем кадр по горизонтали

        region = get_region(frame)  # Получаем предобработанную область интереса

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Если нажата клавиша 'q'
            break  # Выходим из цикла
        if cv2.waitKey(1) & 0xFF == ord('r'):  # Если нажата клавиша 'r'
            frames_elapsed = 0  # Сбрасываем счетчик кадров
            stable_frames = 0  # Сбрасываем счетчик стабильных кадров (не используется в полной мере)

        if frames_elapsed < CALIBRATION_TIME:  # Если идет калибровка фона
            background = get_average(region)  # Обновляем фон
        else: # Если калибровка закончена
            region_pair = segment(region) # Выделяем руку
            if region_pair is not None:  # Если рука выделена
                (thresholded_region, segmented_region) = region_pair  # Разделяем бинарное изображение и контур
                cv2.drawContours(region, [segmented_region], -1, (255, 255, 255), 2)  # Рисуем контур на ROI
                cv2.imshow("Segmented Image", thresholded_region)  # Отображаем бинарное изображение
                count += 1  # Увеличиваем счетчик кадров
                # print(get_hand_data(region, background)) # Отладочный вывод (закомментирован)
                # print(get_hand_data(thresholded_region, segmented_region)) # Отладочный вывод (закомментирован)
                if count == 70:  # Каждые 70 кадров
                    print('yooooooooooooo')
                    c = int(input())  # Запрашиваем у пользователя количество пальцев
                    data['data'].append(get_hand_data(region, background))  # Сохраняем данные
                    data['count_of_fingers'].append(c)  # Сохраняем количество пальцев
                    data['data'].append(get_hand_data(region, background).T) # Сохраняем транспонированные данные
                    data['count_of_fingers'].append(c)  # Сохраняем количество пальцев
                    data['data'].append(get_hand_data(region, background)[::-1]) # Сохраняем данные в обратном порядке
                    data['count_of_fingers'].append(c)  # Сохраняем количество пальцев
                    data['data'].append(get_hand_data(region, background).T[::-1]) # Сохраняем транспонированные данные в обратном порядке
                    data['count_of_fingers'].append(c)  # Сохраняем количество пальцев
                    cnt_of_rows += 1  # Увеличиваем счетчик строк
                    count = 0 # Сбрасываем счетчик кадров
                    frames_elapsed = 0 # Сбрасываем счетчик кадров

        write_on_image(frame)  # Отображаем текст и прямоугольник на кадре
        cv2.imshow("Camera Input", frame)  # Отображаем кадр с камеры
        frames_elapsed += 1  # Увеличиваем счетчик кадров
    file.write(json.dumps(data, cls=NumpyEncoder))  # Записываем данные в JSON файл

capture.release()  # Освобождаем ресурсы камеры
cv2.destroyAllWindows()  # Закрываем все окна

