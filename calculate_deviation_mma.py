import torch
import numpy as np
import cv2
from PIL import Image
from threading import Thread
from queue import Queue
import time

from libs.models.STM import STM
from libs.models.Att import Att
from libs.dataset.transform import TestTransform
from options import OPTION as opt


use_gpu = torch.cuda.is_available() and int(opt.gpu_id) >= 0
device = 'cuda:{}'.format(opt.gpu_id) if use_gpu else 'cpu'

# Загружаем модели один раз при импорте модуля
print('==> Загрузка моделей MMA-Net...')
net = STM(opt.keydim, opt.valdim)
att = Att(save_freq=opt.save_freq, keydim=opt.keydim, valdim=opt.valdim)

net.eval()
att.eval()

if use_gpu:
    net = net.to(device)
    att = att.to(device)

checkpoint = torch.load(opt.resume_STM, map_location=device)
net.load_state_dict(checkpoint['state_dict'], strict=False)

checkpoint = torch.load(opt.resume_ATT, map_location=device)
att.load_state_dict(checkpoint['state_dict'], strict=False)

for p in net.parameters():
    p.requires_grad = False
for p in att.parameters():
    p.requires_grad = False

print('==> Модели загружены')


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
    Вычисляет отклонение усредненной точки схода от центральной оси используя MMA-Net.
    
    Args:
        image: numpy array размером (480, 640, 3) - входное изображение RGB
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
    
    # Убеждаемся что изображение RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Подготовка данных для MMA-Net
    test_transformer = TestTransform(size=opt.input_size)
    dummy_mask = np.zeros(image.shape[:2] + (opt.max_object + 1,))
    frame_t, _ = test_transformer([image], [dummy_mask], False)
    
    if use_gpu:
        frame_t = frame_t.to(device)
    
    # Получаем предсказание от MMA-Net
    with torch.no_grad():
        logits = net(frame=frame_t)
        out = torch.softmax(logits, dim=1)
        pred = out.cpu().numpy()
    
    # Обрабатываем предсказание для получения масок линий
    T = pred.shape[0]
    th, tw = pred.shape[2:]
    factor = min(th / h, tw / w)
    sh, sw = int(factor * h), int(factor * w)
    
    pad_l = (tw - sw) // 2
    pad_t = (th - sh) // 2
    
    m = pred[0, :, pad_t:pad_t + sh, pad_l:pad_l + sw]
    m = m.transpose((1, 2, 0))
    rescale_mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    rescale_mask = rescale_mask.argmax(axis=2).astype(np.uint8)
    
    # Находим количество обнаруженных линий
    num_lanes = rescale_mask.max()
    
    if num_lanes == 0:
        processing_time = time.time() - start_time
        return None, processing_time
    
    # Извлекаем параметры каждой линии
    lines_params = []
    
    for lane_id in range(1, num_lanes + 1):
        # Извлекаем маску для текущей линии
        lane_mask = (rescale_mask == lane_id).astype(np.uint8) * 255
        
        # Находим все точки этой линии
        points = cv2.findNonZero(lane_mask)
        
        if points is None or len(points) < 10:
            continue
        
        # Преобразуем точки для fitLine
        points = points.reshape(-1, 2)
        
        # Аппроксимируем линию методом наименьших квадратов
        [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Вычисляем уравнение прямой
        if abs(vx) < 1e-6:  # Вертикальная линия
            line_params = {'type': 'vertical', 'x': x0[0]}
        else:
            k = vy[0] / vx[0]
            b = y0[0] - k * x0[0]
            line_params = {'type': 'normal', 'k': k, 'b': b}
        
        lines_params.append(line_params)
    
    if len(lines_params) < 2:
        processing_time = time.time() - start_time
        return None, processing_time
    
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
        if len(image.shape) == 3 and image.shape[2] == 3:
            debug_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
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


def calculate_vanishing_point_deviation_async(image, callback, debug_output_path=None):
    """
    Асинхронная версия функции вычисления отклонения точки схода.
    Запускает вычисление в отдельном потоке и вызывает callback с результатом.
    
    Args:
        image: numpy array размером (480, 640, 3) - входное изображение RGB
        callback: функция, которая будет вызвана с результатом callback(deviation, processing_time)
        debug_output_path: путь для сохранения отладочного изображения (опционально)
        
    Returns:
        Thread: объект потока, в котором выполняется вычисление
    """
    def worker():
        """Рабочая функция для потока"""
        try:
            deviation, processing_time = calculate_vanishing_point_deviation(image, debug_output_path)
            callback(deviation, processing_time)
        except Exception as e:
            print(f'Ошибка в потоке вычисления: {e}')
            callback(None, 0.0)
    
    # Создаем и запускаем поток
    thread = Thread(target=worker, daemon=True)
    thread.start()
    
    return thread


if __name__ == '__main__':
    # Пример использования
    import argparse
    
    parser = argparse.ArgumentParser(description='Вычисление отклонения точки схода от центра (MMA-Net)')
    parser.add_argument('--image', type=str, required=True, help='Путь к изображению')
    parser.add_argument('--debug', type=str, default='debug_output.jpg', help='Путь для сохранения отладочного изображения')
    parser.add_argument('--async-mode', action='store_true', help='Использовать асинхронный режим')
    args = parser.parse_args()
    
    # Загружаем изображение
    image = np.array(Image.open(args.image))
    
    if image is None:
        print(f'Ошибка: не удалось загрузить изображение {args.image}')
        exit(1)
    
    # Изменяем размер до 640x480 если нужно
    if image.shape[:2] != (480, 640):
        image = cv2.resize(image, (640, 480))
        print(f'Изображение изменено до размера 640x480')
    
    # Вычисляем отклонение
    if args.async_mode:
        print('==> Обработка изображения (асинхронно)...')
        
        # Callback функция для обработки результата
        def on_result(deviation, processing_time):
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
        
        # Запускаем асинхронное вычисление
        thread = calculate_vanishing_point_deviation_async(image, on_result, debug_output_path=args.debug)
        
        # Ждем завершения потока
        print('==> Ожидание завершения...')
        thread.join()
        print('==> Готово')
    else:
        print('==> Обработка изображения...')
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
