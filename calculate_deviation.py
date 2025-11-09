import numpy as np
import cv2
import time


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


def calculate_vanishing_point_deviation(image, debug_output_path=None):
    """
    Вычисляет отклонение усредненной точки схода от центральной оси.
    
    Args:
        image: numpy array размером (480, 640, 3) - входное изображение BGR
        debug_output_path: путь для сохранения отладочного изображения (опционально)
        
    Returns:
        tuple: (deviation, processing_time)
               deviation: разница по оси X между центром (320) и усредненной точкой схода.
                         Положительное значение - точка схода правее центра,
                         Отрицательное - левее центра.
                         None если точка схода не найдена.
               processing_time: время обработки в секундах
    """
    start_time = time.time()
    
    h, w = image.shape[:2]
    center_x = w / 2  # 320 для изображения 640x480
    
    # Преобразуем в градации серого
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Применяем ROI маску - интересует только нижняя часть изображения (дорога)
    mask = np.zeros_like(gray)
    roi_vertices = np.array([[(0, h), (0, h*0.6), (w, h*0.6), (w, h)]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_gray = cv2.bitwise_and(gray, mask)
    
    # Улучшаем контраст для лучшего выделения разметки
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(masked_gray)
    
    # Применяем морфологические операции для выделения вертикальных линий
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    morphed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
    
    # Применяем более строгий Canny edge detection
    blurred = cv2.GaussianBlur(morphed, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200, apertureSize=3)
    
    # Применяем Hough Line Transform с более строгими параметрами
    # Увеличиваем threshold и minLineLength для фильтрации шума
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, 
                            threshold=80, minLineLength=100, maxLineGap=50)
    
    if lines is None or len(lines) == 0:
        processing_time = time.time() - start_time
        return None, processing_time
    
    # Извлекаем параметры всех найденных линий с фильтрацией
    lines_params = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Вычисляем угол наклона линии
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        
        # Фильтруем горизонтальные линии (шум) - оставляем только наклонные
        # Дорожная разметка обычно имеет наклон от 20 до 70 градусов
        if abs(angle) < 20 or abs(angle) > 80:
            continue
        
        # Вычисляем длину линии - фильтруем короткие
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length < 80:
            continue
        
        # Вычисляем уравнение прямой
        if abs(x2 - x1) < 1e-6:  # Вертикальная линия
            line_params = {'type': 'vertical', 'x': float(x1), 'length': length}
        else:
            k = (y2 - y1) / (x2 - x1)
            b = y1 - k * x1
            line_params = {'type': 'normal', 'k': k, 'b': b, 'length': length}
        
        lines_params.append(line_params)
    
    # Если после фильтрации не осталось линий
    if len(lines_params) == 0:
        processing_time = time.time() - start_time
        return None, processing_time
    
    # Находим все пересечения линий
    intersections = []
    for i in range(len(lines_params)):
        for j in range(i + 1, len(lines_params)):
            intersection = find_intersection(lines_params[i], lines_params[j])
            if intersection is not None:
                x_int, y_int = intersection
                # Точка схода должна быть выше изображения (в перспективе)
                # и в разумных горизонтальных пределах
                if -w*0.5 <= x_int <= w*1.5 and -h*2 <= y_int <= h*0.8:
                    intersections.append((x_int, y_int))
    
    # Если пересечений слишком мало, результат ненадежен
    if len(intersections) < 2:
        processing_time = time.time() - start_time
        return None, processing_time
    
    # Удаляем выбросы используя медианную абсолютную девиацию (MAD)
    x_coords = np.array([x for x, y in intersections])
    y_coords = np.array([y for x, y in intersections])
    
    # Вычисляем медиану
    median_x = np.median(x_coords)
    median_y = np.median(y_coords)
    
    # Вычисляем MAD
    mad_x = np.median(np.abs(x_coords - median_x))
    mad_y = np.median(np.abs(y_coords - median_y))
    
    # Фильтруем выбросы (точки, отклоняющиеся более чем на 2 MAD)
    threshold = 2.5
    filtered_intersections = []
    for x, y in intersections:
        if mad_x > 0 and mad_y > 0:
            if (abs(x - median_x) <= threshold * mad_x and 
                abs(y - median_y) <= threshold * mad_y):
                filtered_intersections.append((x, y))
        else:
            filtered_intersections.append((x, y))
    
    # Используем отфильтрованные пересечения
    if len(filtered_intersections) >= 2:
        intersections = filtered_intersections
    
    # Если нет пересечений, возвращаем None
    if len(intersections) == 0:
        processing_time = time.time() - start_time
        return None, processing_time
    
    # Вычисляем усредненную точку схода (vanishing point)
    avg_x = sum(x for x, y in intersections) / len(intersections)
    avg_y = sum(y for x, y in intersections) / len(intersections)
    
    # Вычисляем отклонение от центральной оси
    deviation = avg_x - center_x
    
    # Сохраняем отладочное изображение если указан путь
    if debug_output_path is not None:
        # Создаем копию изображения для отрисовки
        debug_img = image.copy()
        
        # Цвета для линий
        line_colors = [
            (255, 0, 0),      # Красный
            (0, 255, 0),      # Зеленый
            (0, 0, 255),      # Синий
            (255, 255, 0),    # Желтый
            (255, 0, 255),    # Пурпурный
            (0, 255, 255),    # Голубой
            (255, 128, 0),    # Оранжевый
            (128, 0, 255),    # Фиолетовый
        ]
        
        # Рисуем найденные линии
        for idx, line_params in enumerate(lines_params):
            color = line_colors[idx % len(line_colors)]
            
            if line_params['type'] == 'vertical':
                x = int(line_params['x'])
                cv2.line(debug_img, (x, 0), (x, h-1), color, 2)
            else:
                k = line_params['k']
                b = line_params['b']
                
                x1_draw, y1_draw = 0, int(b)
                x2_draw, y2_draw = w - 1, int(k * (w - 1) + b)
                
                # Ограничиваем координаты
                if y1_draw < 0:
                    x1_draw = int(-b / k) if k != 0 else 0
                    y1_draw = 0
                elif y1_draw >= h:
                    x1_draw = int((h - 1 - b) / k) if k != 0 else 0
                    y1_draw = h - 1
                
                if y2_draw < 0:
                    x2_draw = int(-b / k) if k != 0 else w - 1
                    y2_draw = 0
                elif y2_draw >= h:
                    x2_draw = int((h - 1 - b) / k) if k != 0 else w - 1
                    y2_draw = h - 1
                
                cv2.line(debug_img, (x1_draw, y1_draw), (x2_draw, y2_draw), color, 2)
        
        # Рисуем центральную вертикальную линию (ось Y в центре)
        cv2.line(debug_img, (int(center_x), 0), (int(center_x), h-1), (255, 255, 255), 3)
        cv2.putText(debug_img, 'CENTER', (int(center_x) + 5, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Рисуем все точки пересечения
        for x_int, y_int in intersections:
            if 0 <= x_int < w and 0 <= y_int < h:
                cv2.circle(debug_img, (int(x_int), int(y_int)), 3, (128, 128, 128), -1)
        
        # Рисуем усредненную точку схода
        cv2.circle(debug_img, (int(avg_x), int(avg_y)), 10, (0, 255, 0), -1)
        cv2.circle(debug_img, (int(avg_x), int(avg_y)), 12, (255, 255, 255), 2)
        
        # Добавляем текст с информацией
        cv2.putText(debug_img, f'Vanishing Point: ({avg_x:.1f}, {avg_y:.1f})', 
                    (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(debug_img, f'Center X: {center_x:.1f}', 
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_img, f'Deviation: {deviation:.1f} px', 
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Сохраняем изображение
        cv2.imwrite(debug_output_path, debug_img)
        print(f'==> Отладочное изображение сохранено: {debug_output_path}')
    
    processing_time = time.time() - start_time
    return deviation, processing_time


if __name__ == '__main__':
    # Пример использования
    import argparse
    
    parser = argparse.ArgumentParser(description='Вычисление отклонения точки схода от центра')
    parser.add_argument('--image', type=str, required=True, help='Путь к изображению')
    parser.add_argument('--debug', type=str, default='debug_output_canny.jpg', help='Путь для сохранения отладочного изображения')
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
    deviation, processing_time = calculate_vanishing_point_deviation(image, debug_output_path=args.debug)
    
    print(f'==> Время обработки: {processing_time:.3f}s')
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
