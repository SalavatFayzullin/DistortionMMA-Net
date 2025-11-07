import torch
import torch.utils.data as data
import numpy as np
import os
import argparse
import cv2
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
    
    for t in range(T):
        m = pred[t, :, pad_t:pad_t + sh, pad_l:pad_l + sw]
        m = m.transpose((1, 2, 0))
        rescale_mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        rescale_mask = rescale_mask.argmax(axis=2).astype(np.uint8)
        
        output_path = os.path.join(video_output_dir, f'{t:05d}.png')
        
        if opt.save_indexed_format:
            im = Image.fromarray(rescale_mask).convert('P')
            if palette is not None:
                im.putpalette(palette)
            im.save(output_path, format='PNG')
        else:
            cv2.imwrite(output_path, rescale_mask)

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
        if T > opt.save_freq:
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
    
    # Сохранение
    if os.path.isdir(video_path):
        video_name = os.path.basename(video_path.rstrip('/\\'))
    else:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    save_masks(pred, output_dir, video_name, original_size, palette)
    
    print(f'==> Результаты сохранены: {output_dir}/{video_name}/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='Путь к видео или папке с кадрами')
    parser.add_argument('--output', type=str, default='output_single', help='Папка для результатов')
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f'Ошибка: путь не существует: {args.video}')
    else:
        test_video(args.video, args.output)
