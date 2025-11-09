import numpy as np
import cv2


def find_intersection(line1, line2):
    """Находит точку пересечения двух прямых
    line1, line2: dict с ключами 'type', 'k', 'b' (для y=kx+b) или 'x' (для x=c)
    Возвращает (x, y) или None если прямые параллельны или не пересекаются
    """
    if line1['type'] == 'vertical' and line2['type'] == 'vertical':
        return None  # Параллельные вертикальные линии
    
    if line1['type'] == 'vertical':
        x = line1['x']
        y = line2['k'] * x + line2['b']
        return (x, y)
    
    if line2['type'] == 'vertical':
        x = line2['x']
        y = line1['k'] * x + line1['b']
        return (x, y)
    
    # Обе линии наклонные: y = k1*x + b1 и y = k2*x + b2
    k1, b1 = line1['k'], line1['b']
    k2, b2 = line2['k'], line2['b']
    
    if abs(k1 - k2) < 1e-6:
        return None  # Параллельные линии
    
    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1
    
    return (x, y)


def calculate_vanishing_point_deviation(image):
    """
    Вычисляет отклонение усредненной точки схода от центральной оси.
    
    Args:
        image: numpy array размером (480, 640, 3) - входное изображение BGR
        
    Returns:
        float: разница по оси X между центром (320) и усредненной точкой схода.
               Положительное значение - точка схода правее центра,
               Отрицательное - левее центра.
               None если точка схода не найдена.
    """
    h, w = image.shape[:2]
    center_x = w / 2  # 320 для изображения 640x480
    
    # Преобразуем в градации серого
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Применяем Canny edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Применяем Hough Line Transform для поиска линий
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, 
                            threshold=50, minLineLength=50, maxLineGap=100)
    
    if lines is None or len(lines) == 0:
        return None
    
    # Извлекаем параметры всех найденных линий
    lines_params = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Вычисляем уравнение прямой
        if abs(x2 - x1) < 1e-6:  # Вертикальная линия
            line_params = {'type': 'vertical', 'x': float(x1)}
        else:
            k = (y2 - y1) / (x2 - x1)
            b = y1 - k * x1
            line_params = {'type': 'normal', 'k': k, 'b': b}
        
        lines_params.append(line_params)
    
    # Находим все пересечения линий
    intersections = []
    for i in range(len(lines_params)):
        for j in range(i + 1, len(lines_params)):
            intersection = find_intersection(lines_params[i], lines_params[j])
            if intersection is not None:
                x_int, y_int = intersection
                # Проверяем, что точка находится в разумных пределах
                if -w <= x_int <= 2*w and -h <= y_int <= 2*h:
                    intersections.append((x_int, y_int))
    
    # Если нет пересечений, возвращаем None
    if len(intersections) == 0:
        return None
    
    # Вычисляем усредненную точку схода (vanishing point)
    avg_x = sum(x for x, y in intersections) / len(intersections)
    avg_y = sum(y for x, y in intersections) / len(intersections)
    
    # Вычисляем отклонение от центральной оси
    deviation = avg_x - center_x
    
    return deviation


if __name__ == '__main__':
    # Пример использования
    import argparse
    
    parser = argparse.ArgumentParser(description='Вычисление отклонения точки схода от центра')
    parser.add_argument('--image', type=str, required=True, help='Путь к изображению')
    args = parser.parse_args()
    
    # Загружаем изображение
    image = cv2.imread(args.image)
    
    if image is None:
        print(f'Ошибка: не удалось загрузить изображение {args.image}')
        exit(1)
    
    # Изменяем размер до 640x480 если нужно
    if image.shape[:2] != (480, 640):
        image = cv2.resize(image, (640, 480))
        print(f'Изображение изменено до размера 640x480')
    
    # Вычисляем отклонение
    deviation = calculate_vanishing_point_deviation(image)
    
    if deviation is None:
        print('Точка схода не найдена (нет пересечений линий)')
    else:
        print(f'Отклонение от центра: {deviation:.2f} пикселей')
        if deviation > 0:
            print(f'Точка схода находится ПРАВЕЕ центра на {deviation:.2f} пикселей')
        elif deviation < 0:
            print(f'Точка схода находится ЛЕВЕЕ центра на {abs(deviation):.2f} пикселей')
        else:
            print('Точка схода находится ТОЧНО в центре')
