import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from ultralytics import YOLO
import mediapipe as mp
from skimage.segmentation import slic
from skimage.measure import shannon_entropy
from sklearn.cluster import KMeans
import torch.nn.functional as F
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# -------------------------
# Модели
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Grad-CAM для balance
resnet_model = models.resnet50(pretrained=True).to(device)
resnet_model.eval()
gradcam_layer = resnet_model.layer4[-1]
cam = GradCAM(model=resnet_model, target_layers=[gradcam_layer], use_cuda=torch.cuda.is_available())

# YOLOv8 для объектов
yolo_model = YOLO('yolov8n.pt')

# Mediapipe для лиц
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True)

# MiDaS для depth (ленивая загрузка)
_midas_model = None
_midas_transform = None

def get_midas():
    global _midas_model, _midas_transform
    if _midas_model is None:
        try:
            _midas_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(device)
            _midas_model.eval()
            _midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        except Exception as e:
            print(f"Warning: Could not load MiDaS: {e}. Using fallback depth estimation.")
            _midas_model = None
    return _midas_model, _midas_transform

# Style classification model (ResNet50)
style_model = models.resnet50(pretrained=True).to(device)
style_model.eval()
style_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Style classes
STYLE_CLASSES = [
    'minimalist', 'documentary', 'vlog', 'cinematic', 
    'product_centered', 'interview', 'tiktok', 'gaming', 'artistic'
]

# -------------------------
# Вспомогательные функции
# -------------------------

def rule_of_thirds_features(frame, object_centers=None, face_landmarks=None):
    """Правило третей - расположение объектов по сетке 3x3"""
    H, W, _ = frame.shape
    
    # Линии третей
    third_w = W / 3
    third_h = H / 3
    grid_lines_x = [third_w, 2 * third_w]
    grid_lines_y = [third_h, 2 * third_h]
    intersections = [(x, y) for x in grid_lines_x for y in grid_lines_y]
    
    # Находим главный объект (лицо или самый большой объект)
    main_subject_x = W / 2
    main_subject_y = H / 2
    
    if face_landmarks is not None:
        # Используем центр лица
        face_x = np.mean([lm.x * W for lm in face_landmarks])
        face_y = np.mean([lm.y * H for lm in face_landmarks])
        main_subject_x = face_x
        main_subject_y = face_y
    elif object_centers is not None and len(object_centers) > 0:
        # Используем центр первого объекта
        main_subject_x, main_subject_y = object_centers[0]
    
    # Нормализованные координаты относительно сетки
    main_subject_x_pos = main_subject_x / W
    main_subject_y_pos = main_subject_y / H
    
    # Расстояние до ближайшего пересечения
    min_dist = float('inf')
    for ix, iy in intersections:
        dist = np.sqrt((main_subject_x - ix)**2 + (main_subject_y - iy)**2)
        min_dist = min(min_dist, dist)
    
    # Нормализуем расстояние (максимальное расстояние от центра до угла)
    max_dist = np.sqrt((W/2)**2 + (H/2)**2)
    rule_of_thirds_alignment = 1.0 - min(min_dist / max_dist, 1.0)
    
    # Расстояние от центра
    center_x, center_y = W / 2, H / 2
    subject_offcenter_distance = np.sqrt((main_subject_x - center_x)**2 + (main_subject_y - center_y)**2) / max_dist
    subject_offcenter_angle = np.arctan2(main_subject_y - center_y, main_subject_x - center_x)
    
    # Secondary subjects
    secondary_subjects_count = max(0, len(object_centers) - 1) if object_centers else 0
    secondary_alignment_scores = []
    if object_centers and len(object_centers) > 1:
        for obj_x, obj_y in object_centers[1:]:
            min_obj_dist = float('inf')
            for ix, iy in intersections:
                dist = np.sqrt((obj_x - ix)**2 + (obj_y - iy)**2)
                min_obj_dist = min(min_obj_dist, dist)
            alignment = 1.0 - min(min_obj_dist / max_dist, 1.0)
            secondary_alignment_scores.append(alignment)
    
    secondary_subjects_alignment_score = np.mean(secondary_alignment_scores) if secondary_alignment_scores else 0.0
    
    # Balance index
    left_objects = sum(1 for x, y in (object_centers or []) if x < W/2)
    right_objects = sum(1 for x, y in (object_centers or []) if x >= W/2)
    subject_balance_index = abs(left_objects - right_objects) / max(len(object_centers or [1]), 1)
    
    return {
        'main_subject_x_pos': main_subject_x_pos,
        'main_subject_y_pos': main_subject_y_pos,
        'rule_of_thirds_alignment': rule_of_thirds_alignment,
        'subject_offcenter_distance': subject_offcenter_distance,
        'subject_offcenter_angle': subject_offcenter_angle,
        'secondary_subjects_count': secondary_subjects_count,
        'secondary_subjects_alignment_score': secondary_subjects_alignment_score,
        'subject_balance_index': subject_balance_index
    }

def golden_ratio_features(frame, object_centers=None, face_landmarks=None):
    """Golden ratio (phi = 1.618) - золотое сечение"""
    H, W, _ = frame.shape
    phi = 1.618
    
    # Золотые спирали (4 ориентации)
    center_x, center_y = W / 2, H / 2
    
    # Находим главный объект
    main_x = W / 2
    main_y = H / 2
    if face_landmarks is not None:
        main_x = np.mean([lm.x * W for lm in face_landmarks])
        main_y = np.mean([lm.y * H for lm in face_landmarks])
    elif object_centers is not None and len(object_centers) > 0:
        main_x, main_y = object_centers[0]
    
    # Создаем маски золотых спиралей (упрощенная версия)
    golden_scores = []
    for orientation in range(4):
        # Упрощенная проверка: расстояние до золотых точек
        if orientation == 0:  # Top-left to bottom-right
            golden_x = W / (1 + phi)
            golden_y = H / (1 + phi)
        elif orientation == 1:  # Top-right to bottom-left
            golden_x = W * phi / (1 + phi)
            golden_y = H / (1 + phi)
        elif orientation == 2:  # Bottom-left to top-right
            golden_x = W / (1 + phi)
            golden_y = H * phi / (1 + phi)
        else:  # Bottom-right to top-left
            golden_x = W * phi / (1 + phi)
            golden_y = H * phi / (1 + phi)
        
        dist = np.sqrt((main_x - golden_x)**2 + (main_y - golden_y)**2)
        max_dist = np.sqrt(W**2 + H**2)
        score = 1.0 - min(dist / max_dist, 1.0)
        golden_scores.append(score)
    
    golden_ratio_alignment = max(golden_scores)
    golden_ratio_orientation = np.argmax(golden_scores)
    
    return {
        'golden_ratio_alignment': golden_ratio_alignment,
        'golden_ratio_orientation': golden_ratio_orientation
    }

def balance_features(frame, alpha=0.5, beta=0.3, gamma=0.2):
    H,W,_ = frame.shape

    # YOLO объекты
    results = yolo_model.predict(frame, verbose=False)
    object_centers = []
    obj_map = np.zeros((H,W))
    for box in results[0].boxes.xyxy:
        x1,y1,x2,y2 = map(int, box)
        cx,cy = (x1+x2)/2,(y1+y2)/2
        object_centers.append((cx,cy))
        obj_map[y1:y2,x1:x2] +=1
    obj_map /= (obj_map.max()+1e-6)

    # Grad-CAM attention
    pil_img = Image.fromarray(frame)
    preprocess = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    targets = [ClassifierOutputTarget(281)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    attention_map = cv2.resize(grayscale_cam,(W,H))

    # Brightness
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)/255.0
    brightness_map = gray

    weight_map = alpha*attention_map + beta*brightness_map + gamma*obj_map

    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    mass_center_x = np.sum(x_coords*weight_map)/ (np.sum(weight_map)+1e-6)
    mass_center_y = np.sum(y_coords*weight_map)/ (np.sum(weight_map)+1e-6)

    left_weight = np.sum(weight_map[:,:W//2])
    right_weight = np.sum(weight_map[:,W//2:])
    top_weight = np.sum(weight_map[:H//2,:])
    bottom_weight = np.sum(weight_map[H//2:,:])
    balance_left_right_ratio = left_weight / (right_weight+1e-6)
    balance_top_bottom_ratio = top_weight / (bottom_weight+1e-6)
    visual_weight_asymmetry = abs(balance_left_right_ratio-1)+abs(balance_top_bottom_ratio-1)

    return {
        'mass_center_x': mass_center_x/W,
        'mass_center_y': mass_center_y/H,
        'balance_left_right_ratio': balance_left_right_ratio,
        'balance_top_bottom_ratio': balance_top_bottom_ratio,
        'visual_weight_asymmetry': visual_weight_asymmetry,
        'attention_map': attention_map,
        'object_mask': (obj_map>0).astype(np.uint8),
        'object_centers': object_centers
    }

def leading_lines_features(frame, object_centers=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray,50,150)
    lines = cv2.HoughLinesP(edges,1,np.pi/180, threshold=80, minLineLength=50,maxLineGap=10)

    count = 0
    strengths,directions,alignments = [],[],[]

    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            length = np.sqrt((x2-x1)**2+(y2-y1)**2)
            mask = np.zeros_like(gray)
            cv2.line(mask,(x1,y1),(x2,y2),255,1)
            contrast = np.mean(edges[mask==255])
            strengths.append(length*contrast)
            direction = np.arctan2(y2-y1,x2-x1)
            directions.append(direction)

            if object_centers is not None:
                vx,vy = x2-x1,y2-y1
                line_vec = np.array([vx,vy])
                line_vec_norm = line_vec/(np.linalg.norm(line_vec)+1e-6)
                line_center = np.array([(x1+x2)/2,(y1+y2)/2])
                scores=[]
                for obj_c in object_centers:
                    obj_vec = np.array(obj_c) - line_center
                    obj_vec_norm = obj_vec/ (np.linalg.norm(obj_vec)+1e-6)
                    scores.append(abs(np.dot(line_vec_norm,obj_vec_norm)))
                alignments.append(np.mean(scores))
            count +=1

    return {
        'leading_lines_count':count,
        'leading_lines_strength':np.mean(strengths) if strengths else 0,
        'leading_lines_direction_mean':np.mean(directions) if directions else 0,
        'leading_lines_to_subject_alignment':np.mean(alignments) if alignments else 0
    }

def depth_features(frame, num_depth_layers=3, midas_model=None, midas_transform=None):
    H,W,_ = frame.shape
    
    # Используем MiDaS если доступен
    if midas_model is not None and midas_transform is not None:
        try:
            input_image = Image.fromarray(frame)
            input_tensor = midas_transform(input_image).to(device)
            with torch.no_grad():
                prediction = midas_model(input_tensor)
                depth_map = prediction.cpu().numpy()[0, 0]
                depth_map = cv2.resize(depth_map, (W, H))
                # Нормализуем
                depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
        except Exception as e:
            print(f"Warning: MiDaS inference failed: {e}. Using fallback.")
            depth_map = np.random.rand(H,W)
    else:
        # Fallback: используем градиенты яркости как приближение глубины
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        depth_map = 1.0 - (gradient_magnitude / (gradient_magnitude.max() + 1e-6))
    
    depth_mean = depth_map.mean()
    depth_std = depth_map.std()
    depth_dynamic_range = depth_map.max()-depth_map.min()
    hist,_ = np.histogram(depth_map,bins=256)
    hist_norm = hist / (hist.sum() + 1e-6)
    depth_entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-6))

    # KMeans для foreground/background
    flat_depth = depth_map.flatten().reshape(-1,1)
    kmeans = KMeans(n_clusters=num_depth_layers,random_state=42, n_init=10).fit(flat_depth)
    labels = kmeans.labels_.reshape(H,W)
    foreground_label = np.argmin(kmeans.cluster_centers_)
    background_label = np.argmax(kmeans.cluster_centers_)
    foreground_size_ratio = np.sum(labels==foreground_label)/(H*W)
    midground_presence_ratio = 1 - foreground_size_ratio - np.sum(labels==background_label)/(H*W)
    background_depth_balance = kmeans.cluster_centers_[background_label][0]
    
    # Background clutter index
    background_mask = (labels == background_label).astype(np.float32)
    background_edges = cv2.Canny((background_mask * 255).astype(np.uint8), 50, 150)
    background_clutter_index = background_edges.mean() / 255.0
    
    # Foreground depth distance
    foreground_depth_distance = kmeans.cluster_centers_[foreground_label][0]
    num_depth_layers_detected = len(np.unique(labels))

    bokeh_probability = min(depth_std/0.5,1.0)
    shallow_depth_of_field_prob = bokeh_probability
    focus_plane_variation = depth_std

    return {
        'depth_mean':depth_mean,
        'depth_std':depth_std,
        'depth_dynamic_range':depth_dynamic_range,
        'depth_entropy':depth_entropy,
        'foreground_size_ratio':foreground_size_ratio,
        'midground_presence_ratio':midground_presence_ratio,
        'background_depth_balance':background_depth_balance,
        'background_clutter_index':background_clutter_index,
        'foreground_depth_distance':foreground_depth_distance,
        'num_depth_layers':num_depth_layers_detected,
        'bokeh_probability':bokeh_probability,
        'shallow_depth_of_field_prob':shallow_depth_of_field_prob,
        'focus_plane_variation':focus_plane_variation
    }

def symmetry_features(frame, object_centers=None):
    """Улучшенные метрики симметрии"""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)/255.0
    H,W = gray.shape
    
    # Reflection-based symmetry
    h_flip = cv2.flip(gray,1)
    horizontal_score = np.corrcoef(gray.flatten(),h_flip.flatten())[0,1]
    if np.isnan(horizontal_score):
        horizontal_score = 0.0
    
    v_flip = cv2.flip(gray,0)
    vertical_score = np.corrcoef(gray.flatten(),v_flip.flatten())[0,1]
    if np.isnan(vertical_score):
        vertical_score = 0.0

    # Radial symmetry
    polar = cv2.linearPolar(gray,(W//2,H//2),max(H,W)/2,cv2.WARP_FILL_OUTLIERS)
    radial_flip = cv2.flip(polar,1)
    radial_score = np.corrcoef(polar.flatten(),radial_flip.flatten())[0,1]
    if np.isnan(radial_score):
        radial_score = 0.0
    
    # Детальные метрики симметрии
    # Горизонтальная симметрия по квадрантам
    top_half = gray[:H//2, :]
    bottom_half = cv2.flip(gray[H//2:, :], 0)
    top_bottom_symmetry = np.corrcoef(top_half.flatten(), bottom_half.flatten())[0,1] if top_half.size > 0 else 0.0
    if np.isnan(top_bottom_symmetry):
        top_bottom_symmetry = 0.0
    
    # Вертикальная симметрия по квадрантам
    left_half = gray[:, :W//2]
    right_half = cv2.flip(gray[:, W//2:], 1)
    left_right_symmetry = np.corrcoef(left_half.flatten(), right_half.flatten())[0,1] if left_half.size > 0 else 0.0
    if np.isnan(left_right_symmetry):
        left_right_symmetry = 0.0
    
    # Диагональная симметрия
    diag1_flip = cv2.flip(cv2.flip(gray, 0), 1)  # Диагональ top-left to bottom-right
    diag1_score = np.corrcoef(gray.flatten(), diag1_flip.flatten())[0,1] if not np.isnan(np.corrcoef(gray.flatten(), diag1_flip.flatten())[0,1]) else 0.0
    
    # Object symmetry (симметрия объектов)
    object_symmetry_score = 0.0
    if object_centers is not None and len(object_centers) > 0:
        left_objects = [obj for obj in object_centers if obj[0] < W/2]
        right_objects = [(W - obj[0], obj[1]) for obj in object_centers if obj[0] >= W/2]
        if len(left_objects) > 0 and len(right_objects) > 0:
            # Упрощенная метрика: количество объектов слева и справа
            symmetry_ratio = min(len(left_objects), len(right_objects)) / max(len(left_objects), len(right_objects), 1)
            object_symmetry_score = symmetry_ratio

    # Face symmetry
    face_score = 0
    face_landmarks = None
    results = face_mesh.process(frame)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        face_landmarks = landmarks
        left_x = np.mean([lm.x for i,lm in enumerate(landmarks) if i<len(landmarks)//2])
        right_x = np.mean([lm.x for i,lm in enumerate(landmarks) if i>=len(landmarks)//2])
        face_score = 1 - abs(left_x-right_x)
        
        # Детальная симметрия лица (по ключевым точкам)
        left_eye = np.mean([(lm.x, lm.y) for i, lm in enumerate(landmarks) if 33 <= i <= 46], axis=0)
        right_eye = np.mean([(lm.x, lm.y) for i, lm in enumerate(landmarks) if 263 <= i <= 276], axis=0)
        if len(left_eye) > 0 and len(right_eye) > 0:
            eye_symmetry = 1.0 - abs(left_eye[1] - right_eye[1])  # Вертикальная симметрия глаз
        else:
            eye_symmetry = 0.0
    else:
        eye_symmetry = 0.0

    scene_symmetry_type = 'none'
    max_score = max(horizontal_score,vertical_score,radial_score,face_score)
    if max_score == horizontal_score: scene_symmetry_type='horizontal-align'
    elif max_score == vertical_score: scene_symmetry_type='vertical-align'
    elif max_score == radial_score: scene_symmetry_type='central'
    elif max_score == face_score: scene_symmetry_type='face-symmetry'
    elif max_score < 0.3: scene_symmetry_type='none'

    return {
        'horizontal_symmetry_score':horizontal_score,
        'vertical_symmetry_score':vertical_score,
        'radial_symmetry_score':radial_score,
        'face_symmetry_score':face_score,
        'top_bottom_symmetry':top_bottom_symmetry,
        'left_right_symmetry':left_right_symmetry,
        'diagonal_symmetry_score':diag1_score,
        'object_symmetry_score':object_symmetry_score,
        'eye_symmetry_score':eye_symmetry,
        'scene_symmetry_type':scene_symmetry_type
    }

def negative_space_features(object_mask):
    empty_space = 1 - object_mask.astype(np.float32)
    H,W = object_mask.shape
    negative_space_ratio = empty_space.mean()
    negative_space_left = empty_space[:,:W//2].mean()
    negative_space_right = empty_space[:,W//2:].mean()
    negative_space_top = empty_space[:H//2,:].mean()
    negative_space_bottom = empty_space[H//2:,:].mean()
    hist = np.histogram(empty_space,bins=256,range=(0,1))[0]
    hist_norm = hist / (hist.sum() + 1e-6)
    empty_background_entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-6))
    object_to_background_ratio = 1 - negative_space_ratio
    return {
        'negative_space_ratio':negative_space_ratio,
        'negative_space_left':negative_space_left,
        'negative_space_right':negative_space_right,
        'negative_space_top':negative_space_top,
        'negative_space_bottom':negative_space_bottom,
        'empty_background_entropy':empty_background_entropy,
        'object_to_background_ratio':object_to_background_ratio
    }

def framing_features(frame, object_mask, object_centers=None):
    """Расширенные типы framing"""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    
    framing_present = 0
    framing_strength = 0
    framing_type = 'none'
    framing_types_detected = []
    
    # 1. Rectangular framing (существующий)
    edges = cv2.Canny(gray,50,150)
    contours,_ = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        aspect_ratio = w/h
        if 0.8<aspect_ratio<1.2 and w*h>500:
            framing_present = 1
            overlap = np.sum(object_mask[y:y+h,x:x+w]) / (w*h + 1e-6)
            if overlap > framing_strength:
                framing_strength = overlap
                framing_type = 'rectangular'
                framing_types_detected.append('rectangular')
    
    # 2. Doorway framing (дверные проемы)
    # Ищем вертикальные прямоугольники с высоким соотношением сторон
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if h > w * 2 and h > H * 0.3:  # Высокий вертикальный прямоугольник
            # Проверяем, есть ли объекты внутри
            center_x, center_y = x + w//2, y + h//2
            if object_mask[center_y, center_x] > 0:
                framing_present = 1
                framing_types_detected.append('doorway')
                if 'doorway' not in framing_type:
                    framing_type = 'doorway' if framing_type == 'none' else f"{framing_type},doorway"
    
    # 3. Screen within screen (экраны внутри кадра)
    # Ищем прямоугольники с высоким контрастом внутри
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w > W * 0.2 and h > H * 0.2 and w < W * 0.8 and h < H * 0.8:
            roi = gray[y:y+h, x:x+w]
            if roi.size > 0:
                contrast = roi.std()
                if contrast > gray.std() * 1.2:  # Высокий контраст
                    framing_present = 1
                    framing_types_detected.append('screen_within_screen')
                    if 'screen_within_screen' not in framing_type:
                        framing_type = 'screen_within_screen' if framing_type == 'none' else f"{framing_type},screen_within_screen"
    
    # 4. Frame-inside-frame (рамка внутри рамки)
    # Ищем вложенные контуры
    for i, cnt1 in enumerate(contours):
        x1,y1,w1,h1 = cv2.boundingRect(cnt1)
        for j, cnt2 in enumerate(contours):
            if i == j:
                continue
            x2,y2,w2,h2 = cv2.boundingRect(cnt2)
            # Проверяем вложенность
            if (x1 < x2 < x1+w1 and y1 < y2 < y1+h1 and 
                x2+w2 < x1+w1 and y2+h2 < y1+h1):
                framing_present = 1
                framing_types_detected.append('frame_inside_frame')
                if 'frame_inside_frame' not in framing_type:
                    framing_type = 'frame_inside_frame' if framing_type == 'none' else f"{framing_type},frame_inside_frame"
    
    # 5. Natural framing (деревья, окна, коридоры)
    # Ищем вертикальные линии по краям (как деревья или колонны)
    left_edge = gray[:, :W//10]
    right_edge = gray[:, -W//10:]
    left_edges = cv2.Canny(left_edge, 50, 150)
    right_edges = cv2.Canny(right_edge, 50, 150)
    
    left_vertical_lines = np.sum(left_edges > 0) / left_edges.size
    right_vertical_lines = np.sum(right_edges > 0) / right_edges.size
    
    if left_vertical_lines > 0.1 or right_vertical_lines > 0.1:
        # Проверяем, есть ли объекты в центре
        center_region = object_mask[H//4:3*H//4, W//4:3*W//4]
        if np.sum(center_region) > 0:
            framing_present = 1
            framing_types_detected.append('natural')
            if 'natural' not in framing_type:
                framing_type = 'natural' if framing_type == 'none' else f"{framing_type},natural"
    
    # Обновляем силу framing на основе количества типов
    if framing_types_detected:
        framing_strength = max(framing_strength, len(framing_types_detected) / 5.0)
    
    return {
        'framing_present':framing_present,
        'framing_strength':framing_strength,
        'framing_type':framing_type,
        'framing_types_count':len(framing_types_detected)
    }

def complexity_features(frame, object_centers=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray,50,150)
    edge_density = edges.mean() / 255.0
    segments = slic(frame,n_segments=100,compactness=10)
    region_entropy = shannon_entropy(segments)
    
    # Object clutter index
    object_clutter_index = 0
    if object_centers is not None:
        H, W = gray.shape
        # Плотность объектов
        object_density = len(object_centers) / (H * W / 10000)  # Нормализуем
        # Перекрытие объектов (упрощенная метрика)
        if len(object_centers) > 1:
            distances = []
            for i, obj1 in enumerate(object_centers):
                for obj2 in object_centers[i+1:]:
                    dist = np.sqrt((obj1[0]-obj2[0])**2 + (obj1[1]-obj2[1])**2)
                    distances.append(dist)
            avg_distance = np.mean(distances) if distances else W
            # Меньше расстояние = больше clutter
            object_clutter_index = 1.0 - min(avg_distance / W, 1.0)
        object_clutter_index = max(object_clutter_index, object_density / 10.0)
    
    scene_complexity_score = (edge_density + region_entropy / 10.0 + object_clutter_index) / 3.0
    
    return {
        'edge_density':edge_density,
        'region_entropy':region_entropy,
        'object_clutter_index':object_clutter_index,
        'background_texture_complexity':region_entropy,
        'scene_complexity_score':scene_complexity_score
    }

def style_classification_features(frame):
    """Классификация стиля композиции"""
    pil_img = Image.fromarray(frame)
    input_tensor = style_transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = style_model(input_tensor)
        # Используем признаки для классификации стилей
        # Упрощенная версия: используем статистики изображения + признаки модели
    
    # Извлекаем статистики для классификации стилей
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    
    # Minimalist: низкая сложность, много негативного пространства
    edge_density = cv2.Canny(gray, 50, 150).mean() / 255.0
    minimalist_prob = max(0, 1.0 - edge_density * 2)
    
    # Cinematic: высокий контраст, глубина
    contrast = gray.std() / 255.0
    cinematic_prob = min(1.0, contrast * 2)
    
    # Documentary: средняя сложность, естественное освещение
    brightness = gray.mean() / 255.0
    documentary_prob = 0.5 if 0.3 < brightness < 0.7 else 0.3
    
    # Vlog: часто лицо в центре, средняя сложность
    vlog_prob = 0.4
    
    # Product-centered: объекты в центре, высокий контраст
    product_prob = min(1.0, contrast * 1.5)
    
    # Interview: лицо в кадре, низкая динамика
    interview_prob = 0.3
    
    # TikTok: высокая яркость, высокая насыщенность
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    saturation = hsv[:,:,1].mean() / 255.0
    tiktok_prob = min(1.0, (brightness + saturation) / 2)
    
    # Gaming: высокая контрастность, много объектов
    gaming_prob = min(1.0, contrast * 1.2)
    
    # Artistic: высокая энтропия, необычная композиция
    artistic_prob = min(1.0, edge_density * 1.5)
    
    # Нормализуем вероятности
    probs = np.array([minimalist_prob, documentary_prob, vlog_prob, cinematic_prob,
                     product_prob, interview_prob, tiktok_prob, gaming_prob, artistic_prob])
    probs = probs / (probs.sum() + 1e-6)
    
    result = {}
    for i, style in enumerate(STYLE_CLASSES):
        result[f'style_{style}_prob'] = float(probs[i])
    
    return result

def saliency_features(attention_map, object_masks={}):
    H,W = attention_map.shape
    y_coords, x_coords = np.meshgrid(np.arange(H),np.arange(W),indexing='ij')
    cx = np.sum(x_coords*attention_map)/(np.sum(attention_map)+1e-6)/W
    cy = np.sum(y_coords*attention_map)/(np.sum(attention_map)+1e-6)/H
    focus_spread = attention_map.std()
    hist = np.histogram(attention_map,bins=256)[0]
    hist_norm = hist / (hist.sum() + 1e-6)
    saliency_entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-6))
    attention_to_face_ratio = np.sum(attention_map*object_masks.get('face',0))/(np.sum(attention_map)+1e-6)
    attention_to_product_ratio = np.sum(attention_map*object_masks.get('product',0))/(np.sum(attention_map)+1e-6)
    return {
        'saliency_center_bias_x':cx,
        'saliency_center_bias_y':cy,
        'saliency_focus_spread':focus_spread,
        'saliency_entropy':saliency_entropy,
        'attention_to_face_ratio':attention_to_face_ratio,
        'attention_to_product_ratio':attention_to_product_ratio
    }

# -------------------------
# Основной пайплайн по кадрам
# -------------------------
def analyze_video(frames, use_midas=True):
    """Анализ видео с извлечением всех признаков композиции"""
    all_features=[]
    
    # Загружаем MiDaS если нужно
    midas_model, midas_transform = None, None
    if use_midas:
        midas_model, midas_transform = get_midas()
    
    for frame in frames:
        # Баланс и объекты
        bal = balance_features(frame)
        object_centers = bal.get('object_centers', [])
        
        # Лица
        face_landmarks = None
        results = face_mesh.process(frame)
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
        
        # Правило третей
        rot = rule_of_thirds_features(frame, object_centers, face_landmarks)
        
        # Golden ratio
        gr = golden_ratio_features(frame, object_centers, face_landmarks)
        
        # Leading lines
        ll = leading_lines_features(frame, object_centers)
        
        # Depth
        dep = depth_features(frame, midas_model=midas_model, midas_transform=midas_transform)
        
        # Симметрия
        sym = symmetry_features(frame, object_centers)
        
        # Negative space
        neg = negative_space_features(bal['object_mask'])
        
        # Framing
        fram = framing_features(frame, bal['object_mask'], object_centers)
        
        # Complexity
        comp = complexity_features(frame, object_centers)
        
        # Style classification
        style = style_classification_features(frame)
        
        # Saliency
        sal = saliency_features(bal['attention_map'])
        
        # Объединяем все признаки
        feat = {}
        feat.update(bal)
        # Удаляем несериализуемые поля
        if 'attention_map' in feat:
            del feat['attention_map']
        if 'object_mask' in feat:
            del feat['object_mask']
        if 'object_centers' in feat:
            del feat['object_centers']
        feat.update(rot)
        feat.update(gr)
        feat.update(ll)
        feat.update(dep)
        feat.update(sym)
        feat.update(neg)
        feat.update(fram)
        feat.update(comp)
        feat.update(style)
        feat.update(sal)
        all_features.append(feat)

    # Агрегаты по видео
    video_features={}
    keys = all_features[0].keys()
    for k in keys:
        vals = []
        for f in all_features:
            val = f.get(k)
            if val is not None and (isinstance(val, (int, float, np.number)) or 
                                   (isinstance(val, np.ndarray) and val.size == 1)):
                if isinstance(val, np.ndarray):
                    vals.append(float(val.item()))
                else:
                    vals.append(float(val))
        
        if vals:
            vals = np.array(vals)
            video_features[f'{k}_mean'] = float(vals.mean())
            video_features[f'{k}_std'] = float(vals.std())
            video_features[f'{k}_min'] = float(vals.min())
            video_features[f'{k}_max'] = float(vals.max())
            video_features[f'{k}_median'] = float(np.median(vals))
    
    return video_features

# -------------------------
# CLI интерфейс
# -------------------------
def process_video(video_path: str, output_path: Optional[str] = None, 
                 use_midas: bool = True, fps_sample: int = 1):
    """Обработка видео через CLI"""
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frames = []
    frame_count = 0
    
    print(f"Reading video: {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % fps_sample == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        frame_count += 1
    
    cap.release()
    print(f"Processed {len(frames)} frames")
    
    print("Analyzing composition features...")
    features = analyze_video(frames, use_midas=use_midas)
    
    if output_path is None:
        output_path = str(Path(video_path).with_suffix('.json'))
    
    with open(output_path, 'w') as f:
        json.dump(features, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    return features

def main():
    parser = argparse.ArgumentParser(description='Frame Composition Analysis')
    parser.add_argument('video_path', type=str, help='Path to input video file')
    parser.add_argument('-o', '--output', type=str, default=None, 
                       help='Output JSON file path (default: video_name.json)')
    parser.add_argument('--no-midas', action='store_true', 
                       help='Disable MiDaS depth estimation (use fallback)')
    parser.add_argument('--fps-sample', type=int, default=1,
                       help='Sample every N frames (default: 1)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return
    
    try:
        features = process_video(
            args.video_path, 
            args.output, 
            use_midas=not args.no_midas,
            fps_sample=args.fps_sample
        )
        print(f"\nExtracted {len(features)} composition features")
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()

if __name__=="__main__":
    main()
