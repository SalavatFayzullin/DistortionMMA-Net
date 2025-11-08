import torch
import torch.utils.data as data
import numpy as np
import os
import argparse
import cv2
import time
from PIL import Image

from libs.models.STM import STM
from libs.models.Att import Att
from libs.dataset.transform import TestTransform
from options import OPTION as opt
from random import shuffle

use_gpu = torch.cuda.is_available() and int(opt.gpu_id) >= 0
device = 'cuda:{}'.format(opt.gpu_id) if use_gpu else 'cpu'

def load_video_frames(path):
    """Загрузка кадров из видео или папки"""
    frames = []
    
    if os.path.isdir(path):
        # Загрузка из папки с изображениями
        image_files = sorted([f for f in os.listdir(path) if f.endswith(('.jpg', '.png'))])
        for img_file in image_files:
            img_path = os.path.join(path, img_file)
            frame = np.array(Image.open(img_path))
            if len(frame.shape) == 2:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            frames.append(frame)
    else:
        # Загрузка из видео файла
        cap = cv2.VideoCapture(path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
    
    return frames

def save_masks(pred, output_dir, video_name, original_size, palette=None):
    """Сохранение масок в формате как у исходной модели"""
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    h, w = original_size
    T = pred.shape[0]
    th, tw = pred.shape[2:]
    factor = min(th / h, tw / w)
    sh, sw = int(factor * h), int(factor * w)
    
    pad_l = (tw - sw) // 2
    pad_t = (th - sh) // 2
    
    # Создаем яркую палитру для линий
    bright_palette = [0, 0, 0]  # Фон - черный
    bright_colors = [
        [255, 0, 0],      # Красный
        [0, 255, 0],      # Зеленый
        [0, 0, 255],      # Синий
        [255, 255, 0],    # Желтый
        [255, 0, 255],    # Пурпурный
        [0, 255, 255],    # Голубой
        [255, 128, 0],    # Оранжевый
        [128, 0, 255],    # Фиолетовый
    ]
    for color in bright_colors:
        bright_palette.extend(color)
    # Заполняем остальные 247 цветов
    bright_palette.extend([0] * (256 - len(bright_colors) - 1) * 3)
    
    for t in range(T):
        m = pred[t, :, pad_t:pad_t + sh, pad_l:pad_l + sw]
        m = m.transpose((1, 2, 0))
        rescale_mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        rescale_mask = rescale_mask.argmax(axis=2).astype(np.uint8)
        
        output_path = os.path.join(video_output_dir, f'{t:05d}.png')
        
        if opt.save_indexed_format:
            im = Image.fromarray(rescale_mask).convert('P')
            # Используем яркую палитру вместо оригинальной
            im.putpalette(bright_palette)
            im.save(output_path, format='PNG')
        else:
            cv2.imwrite(output_path, rescale_mask)

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

def apply_hough_lines(pred, output_dir, video_name, original_size, original_frames):
    """Применение Hough Line Transform для получения уравнений линий из масок MMA-NET"""
    hough_output_dir = os.path.join(output_dir, video_name + '_hough')
    equations_dir = os.path.join(output_dir, video_name + '_equations')
    os.makedirs(hough_output_dir, exist_ok=True)
    os.makedirs(equations_dir, exist_ok=True)
    
    h, w = original_size
    T = pred.shape[0]
    th, tw = pred.shape[2:]
    factor = min(th / h, tw / w)
    sh, sw = int(factor * h), int(factor * w)
    
    pad_l = (tw - sw) // 2
    pad_t = (th - sh) // 2
    
    # Цвета для разных линий
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
    
    for t in range(T):
        m = pred[t, :, pad_t:pad_t + sh, pad_l:pad_l + sw]
        m = m.transpose((1, 2, 0))
        rescale_mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        rescale_mask = rescale_mask.argmax(axis=2).astype(np.uint8)
        
        # Создаем чистое изображение для отрисовки только прямых
        result_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Файл для сохранения уравнений
        equations_file = os.path.join(equations_dir, f'{t:05d}_equations.txt')
        
        # Обрабатываем каждую линию отдельно
        num_lanes = rescale_mask.max()
        
        # Список для хранения параметров линий
        lines_params = []
        
        with open(equations_file, 'w', encoding='utf-8') as f:
            f.write(f'Кадр {t:05d} - Уравнения линий из MMA-NET масок\n')
            f.write('=' * 60 + '\n\n')
            
            for lane_id in range(1, num_lanes + 1):
                # Извлекаем маску для текущей линии (найденную MMA-NET)
                lane_mask = (rescale_mask == lane_id).astype(np.uint8) * 255
                
                # Находим все точки этой линии
                points = cv2.findNonZero(lane_mask)
                
                if points is None or len(points) < 10:
                    f.write(f'Линия {lane_id}: недостаточно точек\n\n')
                    continue
                
                # Преобразуем точки для fitLine
                points = points.reshape(-1, 2)
                
                # Аппроксимируем линию методом наименьших квадратов
                [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
                
                # Вычисляем уравнение прямой
                # Прямая задается как: (x - x0)/vx = (y - y0)/vy
                # Преобразуем в y = kx + b
                
                line_params = {}
                
                if abs(vx) < 1e-6:  # Вертикальная линия
                    equation = f'x = {x0[0]:.2f}'
                    x1_draw, y1_draw = int(x0[0]), 0
                    x2_draw, y2_draw = int(x0[0]), h - 1
                    line_params = {'type': 'vertical', 'x': x0[0]}
                else:
                    k = vy[0] / vx[0]
                    b = y0[0] - k * x0[0]
                    equation = f'y = {k:.4f}x + {b:.2f}'
                    line_params = {'type': 'normal', 'k': k, 'b': b}
                    
                    # Находим точки пересечения с границами изображения
                    x1_draw, y1_draw = 0, int(b)
                    x2_draw, y2_draw = w - 1, int(k * (w - 1) + b)
                    
                    # Ограничиваем координаты границами изображения
                    if y1_draw < 0:
                        x1_draw = int(-b / k)
                        y1_draw = 0
                    elif y1_draw >= h:
                        x1_draw = int((h - 1 - b) / k)
                        y1_draw = h - 1
                    
                    if y2_draw < 0:
                        x2_draw = int(-b / k)
                        y2_draw = 0
                    elif y2_draw >= h:
                        x2_draw = int((h - 1 - b) / k)
                        y2_draw = h - 1
                
                lines_params.append(line_params)
                
                # Рисуем прямую на чистом изображении
                color = line_colors[(lane_id - 1) % len(line_colors)]
                cv2.line(result_img, (x1_draw, y1_draw), (x2_draw, y2_draw), color, 3)
                
                # Сохраняем уравнение
                f.write(f'Линия {lane_id}:\n')
                f.write(f'  Уравнение: {equation}\n')
                f.write(f'  Точек в маске: {len(points)}\n')
                f.write(f'  Направляющий вектор: ({vx[0]:.4f}, {vy[0]:.4f})\n')
                f.write(f'  Точка на прямой: ({x0[0]:.2f}, {y0[0]:.2f})\n')
                f.write('\n')
            
            # Находим все пересечения
            f.write('\n' + '=' * 60 + '\n')
            f.write('ТОЧКИ ПЕРЕСЕЧЕНИЯ\n')
            f.write('=' * 60 + '\n\n')
            
            intersections = []
            for i in range(len(lines_params)):
                for j in range(i + 1, len(lines_params)):
                    intersection = find_intersection(lines_params[i], lines_params[j])
                    if intersection is not None:
                        x_int, y_int = intersection
                        # Проверяем, что точка находится в разумных пределах (можно расширить границы)
                        if -w <= x_int <= 2*w and -h <= y_int <= 2*h:
                            intersections.append((x_int, y_int))
                            f.write(f'Пересечение линий {i+1} и {j+1}: ({x_int:.2f}, {y_int:.2f})\n')
            
            # Вычисляем усредненную точку пересечения
            if intersections:
                avg_x = sum(x for x, y in intersections) / len(intersections)
                avg_y = sum(y for x, y in intersections) / len(intersections)
                
                f.write(f'\nВсего пересечений: {len(intersections)}\n')
                f.write(f'Усредненная точка схода: ({avg_x:.2f}, {avg_y:.2f})\n')
                
                # Рисуем все точки пересечения маленькими точками
                for x_int, y_int in intersections:
                    if 0 <= x_int < w and 0 <= y_int < h:
                        cv2.circle(result_img, (int(x_int), int(y_int)), 3, (128, 128, 128), -1)
                
                # Рисуем усредненную точку большой яркой точкой
                if 0 <= avg_x < w and 0 <= avg_y < h:
                    cv2.circle(result_img, (int(avg_x), int(avg_y)), 8, (255, 255, 255), -1)
                    cv2.circle(result_img, (int(avg_x), int(avg_y)), 10, (0, 255, 0), 2)
            else:
                f.write('\nТочек пересечения не найдено\n')
        
        # Сохраняем результат - изображение только с прямыми
        output_path = os.path.join(hough_output_dir, f'{t:05d}_lines.jpg')
        cv2.imwrite(output_path, result_img)

def test_video(video_path, output_dir='output_single'):
    print(f'==> Обработка: {video_path}')
    
    # Загрузка кадров
    frames = load_video_frames(video_path)
    if len(frames) == 0:
        print('Ошибка: нет кадров')
        return
    
    print(f'Загружено кадров: {len(frames)}')
    original_size = frames[0].shape[:2]
    
    # Получение палитры (если есть аннотация)
    palette = None
    if os.path.isdir(video_path):
        anno_path = video_path.replace('JPEGImages', 'Annotations')
        if os.path.exists(anno_path):
            anno_files = sorted([f for f in os.listdir(anno_path) if f.endswith('.png')])
            if anno_files:
                first_anno = Image.open(os.path.join(anno_path, anno_files[0]))
                palette = first_anno.getpalette()
    
    # Подготовка данных
    test_transformer = TestTransform(size=opt.input_size)
    frames_tensor = []
    
    for frame in frames:
        dummy_mask = np.zeros(frame.shape[:2] + (opt.max_object + 1,))
        frame_t, _ = test_transformer([frame], [dummy_mask], False)
        frames_tensor.append(frame_t)
    
    frames_tensor = torch.cat(frames_tensor, dim=0)
    if use_gpu:
        frames_tensor = frames_tensor.to(device)
    
    # Модели
    print('==> Загрузка моделей')
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
    
    # Обработка
    print('==> Обработка кадров')
    start_time = time.time()
    
    with torch.no_grad():
        T, _, H, W = frames_tensor.shape
        num_objects = opt.max_object
        
        pred = []
        keys = []
        vals = []
        keys3 = []
        vals3 = []
        
        # Первые кадры
        for t in range(min(opt.save_freq, T)):
            print(f'Кадр {t+1}/{T}', end='\r')
            logits = net(frame=frames_tensor[t:t+1, :, :, :])
            out = torch.softmax(logits, dim=1)
            pred.append(out)
            
            key, val, _, key3, val3, _ = net(
                frame=frames_tensor[t:t+1, :, :, :],
                mask=out,
                num_objects=num_objects
            )
            
            keys.append(key)
            vals.append(val)
            keys3.append(key3)
            vals3.append(val3)
        
        # Второй проход по первым кадрам с attention для улучшения качества
        if T >= opt.save_freq:
            for t in range(min(opt.save_freq, T)):
                print(f'Улучшение кадра {t+1}/{opt.save_freq}', end='\r')
                
                tmp_key_local = torch.stack(keys[-opt.save_freq:])
                tmp_val_local = torch.stack(vals[-opt.save_freq:])
                tmp_key_local3 = torch.stack(keys3[-opt.save_freq:])
                tmp_val_local3 = torch.stack(vals3[-opt.save_freq:])
                
                shuffle_keys = keys.copy()
                shuffle_vals = vals.copy()
                shuffle(shuffle_keys)
                shuffle(shuffle_vals)
                tmp_key_global = torch.stack(shuffle_keys[-opt.save_freq:])
                tmp_val_global = torch.stack(shuffle_vals[-opt.save_freq:])
                
                shuffle_keys3 = keys3.copy()
                shuffle_vals3 = vals3.copy()
                shuffle(shuffle_keys3)
                shuffle(shuffle_vals3)
                tmp_key_global3 = torch.stack(shuffle_keys3[-opt.save_freq:])
                tmp_val_global3 = torch.stack(shuffle_vals3[-opt.save_freq:])
                
                tmp_key_local = att(f=tmp_key_local, tag='att_in_local')
                tmp_val_local = att(f=tmp_val_local, tag='att_out_local')
                tmp_key_global = att(f=tmp_key_global, tag='att_in_global')
                tmp_val_global = att(f=tmp_val_global, tag='att_out_global')
                
                tmp_key_local3 = att(f=tmp_key_local3, tag='att_in_local3')
                tmp_val_local3 = att(f=tmp_val_local3, tag='att_out_local3')
                tmp_key_global3 = att(f=tmp_key_global3, tag='att_in_global3')
                tmp_val_global3 = att(f=tmp_val_global3, tag='att_out_global3')
                
                tmp_key = tmp_key_local + tmp_key_global
                tmp_val = tmp_val_local + tmp_val_global
                tmp_key3 = tmp_key_local3 + tmp_key_global3
                tmp_val3 = tmp_val_local3 + tmp_val_global3
                
                logits, _, _ = net(
                    frame=frames_tensor[t:t+1, :, :, :],
                    keys=tmp_key,
                    values=tmp_val,
                    keys3=tmp_key3,
                    values3=tmp_val3,
                    num_objects=num_objects
                )
                
                out = torch.softmax(logits, dim=1)
                pred[t] = out
            print()
        
        # Остальные кадры
        for t in range(opt.save_freq, T):
            print(f'Кадр {t+1}/{T}', end='\r')
            
            tmp_key_local = torch.stack(keys[-opt.save_freq:])
            tmp_val_local = torch.stack(vals[-opt.save_freq:])
            tmp_key_local3 = torch.stack(keys3[-opt.save_freq:])
            tmp_val_local3 = torch.stack(vals3[-opt.save_freq:])
            
            shuffle_keys = keys.copy()
            shuffle_vals = vals.copy()
            shuffle(shuffle_keys)
            shuffle(shuffle_vals)
            tmp_key_global = torch.stack(shuffle_keys[-opt.save_freq:])
            tmp_val_global = torch.stack(shuffle_vals[-opt.save_freq:])
            
            shuffle_keys3 = keys3.copy()
            shuffle_vals3 = vals3.copy()
            shuffle(shuffle_keys3)
            shuffle(shuffle_vals3)
            tmp_key_global3 = torch.stack(shuffle_keys3[-opt.save_freq:])
            tmp_val_global3 = torch.stack(shuffle_vals3[-opt.save_freq:])
            
            tmp_key_local = att(f=tmp_key_local, tag='att_in_local')
            tmp_val_local = att(f=tmp_val_local, tag='att_out_local')
            tmp_key_global = att(f=tmp_key_global, tag='att_in_global')
            tmp_val_global = att(f=tmp_val_global, tag='att_out_global')
            
            tmp_key_local3 = att(f=tmp_key_local3, tag='att_in_local3')
            tmp_val_local3 = att(f=tmp_val_local3, tag='att_out_local3')
            tmp_key_global3 = att(f=tmp_key_global3, tag='att_in_global3')
            tmp_val_global3 = att(f=tmp_val_global3, tag='att_out_global3')
            
            tmp_key = tmp_key_local + tmp_key_global
            tmp_val = tmp_val_local + tmp_val_global
            tmp_key3 = tmp_key_local3 + tmp_key_global3
            tmp_val3 = tmp_val_local3 + tmp_val_global3
            
            logits, ps, _ = net(
                frame=frames_tensor[t:t+1, :, :, :],
                keys=tmp_key,
                values=tmp_val,
                keys3=tmp_key3,
                values3=tmp_val3,
                num_objects=num_objects
            )
            
            out = torch.softmax(logits, dim=1)
            pred.append(out)
            
            key, val, _, key3, val3, _ = net(
                frame=frames_tensor[t:t+1, :, :, :],
                mask=out,
                num_objects=num_objects
            )
            
            keys.append(key)
            vals.append(val)
            keys3.append(key3)
            vals3.append(val3)
            
            if len(keys) > opt.save_freq_max:
                keys.pop(0)
                vals.pop(0)
                keys3.pop(0)
                vals3.pop(0)
        
        print()
        pred = torch.cat(pred, dim=0)
        pred = pred.cpu().numpy()
    
    end_time = time.time()
    total_time = end_time - start_time
    fps = T / total_time
    
    # Сохранение
    if os.path.isdir(video_path):
        video_name = os.path.basename(video_path.rstrip('/\\'))
    else:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    save_masks(pred, output_dir, video_name, original_size, palette)
    
    # Вычисляем уравнения линий и строим прямые
    print('==> Вычисление уравнений линий...')
    apply_hough_lines(pred, output_dir, video_name, original_size, frames)
    
    print(f'==> Результаты сохранены: {output_dir}/{video_name}/')
    print(f'==> Прямые линии: {output_dir}/{video_name}_hough/')
    print(f'==> Уравнения линий: {output_dir}/{video_name}_equations/')
    print(f'==> Время обработки: {total_time:.2f}s')
    print(f'==> FPS: {fps:.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='Путь к видео или папке с кадрами')
    parser.add_argument('--output', type=str, default='output_single', help='Папка для результатов')
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f'Ошибка: путь не существует: {args.video}')
    else:
        test_video(args.video, args.output)
