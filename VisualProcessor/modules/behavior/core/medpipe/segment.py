import cv2
import mediapipe as mp
import numpy as np
import time

mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing = mp.solutions.drawing_utils

# Конфигурация для Selfie Segmentation
cfg = {
    'model_selection': 1,  # 0: общая модель, 1: модель для лица
}

# Эффекты для фона
BACKGROUND_EFFECTS = [
    'none',           # Без эффекта (просто сегментация)
    'blur',           # Размытие фона
    'color',          # Одноцветный фон
    'image',          # Фон из изображения
    'pixelate',       # Пикселизация фона
    'edge_detection', # Детекция краев на фоне
    'invert'          # Инвертирование фона
]

# Цвета для фона
BACKGROUND_COLORS = {
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'red': (0, 0, 255),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'gray': (128, 128, 128),
    'purple': (128, 0, 128),
    'pink': (255, 192, 203)
}

def load_background_image(image_path, target_size):
    """Загружает и подгоняет фоновое изображение"""
    try:
        bg_image = cv2.imread(image_path)
        if bg_image is None:
            print(f"Warning: Could not load background image from {image_path}")
            return None
        
        # Изменяем размер фона под размер кадра
        bg_resized = cv2.resize(bg_image, (target_size[1], target_size[0]))
        return bg_resized
    except Exception as e:
        print(f"Error loading background image: {e}")
        return None

def apply_background_effect(frame, segmentation_mask, effect_type, effect_params=None):
    """
    Применяет различные эффекты к фону на основе маски сегментации.
    
    Args:
        frame: Исходный кадр
        segmentation_mask: Маска сегментации (значения 0.0-1.0)
        effect_type: Тип эффекта из BACKGROUND_EFFECTS
        effect_params: Дополнительные параметры (цвет, изображение и т.д.)
    
    Returns:
        Кадр с примененным эффектом
    """
    if segmentation_mask is None:
        return frame
    
    # Нормализуем маску и создаем условие для переднего плана
    # Маска обычно в диапазоне 0-1, где 1 = человек (передний план)
    mask_condition = segmentation_mask > 0.1
    
    # Создаем 3D маску для применения к RGB кадру
    if len(segmentation_mask.shape) == 2:
        mask_condition_3d = np.stack([mask_condition] * 3, axis=-1)
    else:
        mask_condition_3d = mask_condition
    
    if effect_type == 'none':
        # Просто возвращаем оригинальный кадр с контуром сегментации
        return frame
    
    elif effect_type == 'blur':
        # Сильное размытие фона
        blurred_bg = cv2.GaussianBlur(frame, (99, 99), 30)
        
        # Сохраняем передний план, размываем фон
        output = np.where(mask_condition_3d, frame, blurred_bg)
        return output.astype(np.uint8)
    
    elif effect_type == 'color':
        # Одноцветный фон
        if effect_params and 'color' in effect_params:
            bg_color = effect_params['color']
        else:
            bg_color = BACKGROUND_COLORS['gray']
        
        # Создаем фон заданного цвета
        colored_bg = np.ones_like(frame) * bg_color
        output = np.where(mask_condition_3d, frame, colored_bg)
        return output.astype(np.uint8)
    
    elif effect_type == 'image':
        # Замена фона на изображение
        if effect_params and 'background_image' in effect_params:
            bg_image = effect_params['background_image']
            if bg_image is not None:
                # Убедимся, что размеры совпадают
                if bg_image.shape[:2] != frame.shape[:2]:
                    bg_image = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))
                
                output = np.where(mask_condition_3d, frame, bg_image)
                return output.astype(np.uint8)
        
        # Если изображение не загружено, используем черный фон
        return np.where(mask_condition_3d, frame, np.zeros_like(frame)).astype(np.uint8)
    
    elif effect_type == 'pixelate':
        # Пикселизация фона
        # Уменьшаем размер фона
        small = cv2.resize(frame, (frame.shape[1] // 20, frame.shape[0] // 20), 
                          interpolation=cv2.INTER_LINEAR)
        # Увеличиваем обратно (создаем пиксельный эффект)
        pixelated_bg = cv2.resize(small, (frame.shape[1], frame.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
        
        output = np.where(mask_condition_3d, frame, pixelated_bg)
        return output.astype(np.uint8)
    
    elif effect_type == 'edge_detection':
        # Детекция краев на фоне
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Инвертируем маску для фона
        bg_condition = ~mask_condition_3d
        
        # Применяем детекцию краев только к фону
        output = frame.copy()
        output[bg_condition] = edges_colored[bg_condition]
        return output
    
    elif effect_type == 'invert':
        # Инвертирование фона
        inverted_bg = 255 - frame
        output = np.where(mask_condition_3d, frame, inverted_bg)
        return output.astype(np.uint8)
    
    else:
        # По умолчанию возвращаем оригинальный кадр
        return frame

def draw_segmentation_contour(frame, segmentation_mask, color=(0, 255, 0), thickness=2):
    """
    Рисует контур сегментации на кадре.
    
    Args:
        frame: Кадр для рисования
        segmentation_mask: Маска сегментации
        color: Цвет контура (BGR)
        thickness: Толщина линии
    
    Returns:
        Кадр с контуром
    """
    if segmentation_mask is None:
        return frame
    
    # Преобразуем маску в бинарное изображение
    mask_binary = (segmentation_mask > 0.1).astype(np.uint8) * 255
    
    # Находим контуры
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Рисуем контуры
    contour_frame = frame.copy()
    cv2.drawContours(contour_frame, contours, -1, color, thickness)
    
    return contour_frame

def calculate_segmentation_stats(segmentation_mask):
    """
    Вычисляет статистику по сегментации.
    
    Returns:
        dict: Словарь со статистикой
    """
    if segmentation_mask is None:
        return {
            'foreground_percentage': 0.0,
            'mask_area': 0,
            'mask_mean': 0.0
        }
    
    # Процент пикселей, относящихся к переднему плану
    foreground_pixels = np.sum(segmentation_mask > 0.1)
    total_pixels = segmentation_mask.size
    foreground_percentage = (foreground_pixels / total_pixels) * 100
    
    # Среднее значение маски (уверенность)
    mask_mean = np.mean(segmentation_mask)
    
    return {
        'foreground_percentage': foreground_percentage,
        'mask_area': int(foreground_pixels),
        'mask_mean': mask_mean
    }

def draw_statistics(frame, stats, current_effect, display_scale, fps=0):
    """Отображает статистику на кадре"""
    h, w = frame.shape[:2]
    
    # Полупрозрачная панель для статистики
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    y_offset = 40
    line_height = 25
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y_offset += line_height
    
    # Процент переднего плана
    cv2.putText(frame, f"Foreground: {stats['foreground_percentage']:.1f}%", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    y_offset += line_height
    
    # Площадь маски
    cv2.putText(frame, f"Mask area: {stats['mask_area']:,} px", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 1)
    y_offset += line_height
    
    # Средняя уверенность
    cv2.putText(frame, f"Confidence: {stats['mask_mean']:.3f}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    y_offset += line_height
    
    # Текущий эффект
    cv2.putText(frame, f"Effect: {current_effect}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
    y_offset += line_height
    
    # Масштаб отображения
    cv2.putText(frame, f"Scale: {display_scale}%", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 1)

def resize_frame(frame, scale_percent=50):
    """Изменяет размер кадра для отображения"""
    if scale_percent == 100:
        return frame
        
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    return cv2.resize(frame, (width, height))

def main():
    """Основная функция"""
    
    # Инициализация сегментации
    with mp_selfie_segmentation.SelfieSegmentation(**cfg) as selfie_segmentation:
        
        # Загрузка фонового изображения (опционально)
        background_image_path = None  # Укажите путь к изображению, например: "background.jpg"
        background_image = None
        if background_image_path:
            print(f"Loading background image from {background_image_path}")
            # Предварительная загрузка, размер будет изменен позже
        
        # Открытие видеофайла
        cap = cv2.VideoCapture("-NSumhkOwSg.mp4")
        
        # Проверка успешного открытия видео
        if not cap.isOpened():
            print("Error: Failed to open video file.")
            return
        
        # Получаем параметры видео
        fps_source = cap.get(cv2.CAP_PROP_FPS)
        fps_video = int(fps_source) if fps_source > 0 else 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Для вычисления реального FPS
        frame_count = 0
        start_time = cv2.getTickCount()
        
        # Настройки по умолчанию
        display_scale = 50
        current_effect_idx = 0  # Индекс текущего эффекта
        current_effect = BACKGROUND_EFFECTS[current_effect_idx]
        current_color_idx = 0  # Индекс текущего цвета
        current_color_name = list(BACKGROUND_COLORS.keys())[current_color_idx]
        show_contour = True
        effect_params = {}
        
        print("=" * 60)
        print("MEDIAPIPE SELFIE SEGMENTATION")
        print("=" * 60)
        print(f"Video resolution: {width}x{height}")
        print(f"Video FPS: {fps_video}")
        print(f"Model selection: {cfg['model_selection']}")
        print("=" * 60)
        print("Available effects:")
        for i, effect in enumerate(BACKGROUND_EFFECTS):
            print(f"  {i}: {effect}")
        print("=" * 60)
        print("Controls:")
        print("  q / ESC - Quit")
        print("  +/-     - Change window scale")
        print("  e       - Cycle through effects")
        print("  c       - Change background color (for 'color' effect)")
        print("  s       - Toggle segmentation contour")
        print("  b       - Load background image (for 'image' effect)")
        print("  r       - Reset to default settings")
        print("=" * 60)
        
        # Если загружаем фоновое изображение, ждем первого кадра для определения размера
        if background_image_path and cap.isOpened():
            ret, test_frame = cap.read()
            if ret:
                background_image = load_background_image(
                    background_image_path, 
                    test_frame.shape[:2]
                )
                if background_image is not None:
                    effect_params['background_image'] = background_image
                    print(f"Background image loaded successfully")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Возвращаемся к началу видео
        
        # Создание VideoWriter для сохранения результата (опционально)
        save_output = False
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('output_segmentation.mp4', fourcc, 
                                 fps_video, (width, height))
        
        # Основной цикл обработки
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Video ended.")
                break
            
            frame_count += 1
            original_frame = frame.copy()
            
            # Конвертация BGR в RGB для MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False  # Для оптимизации
            
            # Получение маски сегментации
            results = selfie_segmentation.process(rgb_frame)
            segmentation_mask = results.segmentation_mask
            
            # Применение эффекта к фону
            if current_effect == 'color':
                effect_params['color'] = BACKGROUND_COLORS[current_color_name]
            
            processed_frame = apply_background_effect(
                original_frame, 
                segmentation_mask, 
                current_effect, 
                effect_params
            )
            
            # Добавление контура сегментации (опционально)
            if show_contour and segmentation_mask is not None:
                contour_color = (0, 255, 0)  # Зеленый
                if current_effect == 'color' and 'color' in effect_params:
                    # Используем контрастный цвет
                    bg_color = effect_params['color']
                    contour_color = (
                        255 - bg_color[0],
                        255 - bg_color[1],
                        255 - bg_color[2]
                    )
                processed_frame = draw_segmentation_contour(
                    processed_frame, 
                    segmentation_mask, 
                    contour_color, 
                    thickness=2
                )
            
            # Вычисление статистики
            stats = calculate_segmentation_stats(segmentation_mask)
            
            # Вычисление реального FPS
            current_time = cv2.getTickCount()
            elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Отображение статистики
            draw_statistics(processed_frame, stats, current_effect, 
                           display_scale, current_fps)
            
            # Изменение размера для отображения
            display_frame = resize_frame(processed_frame, display_scale)
            
            # Показ кадра
            cv2.imshow('MediaPipe Selfie Segmentation', display_frame)

            # Сохранение кадра (если включено)
            if save_output:
                out.write(processed_frame)
            
            # Обработка клавиш управления
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # q или ESC
                print("Exit by user request.")
                break
            elif key == ord('+') or key == ord('='):
                display_scale = min(display_scale + 10, 200)
                print(f"Scale increased to {display_scale}%")
            elif key == ord('-') or key == ord('_'):
                display_scale = max(display_scale - 10, 20)
                print(f"Scale decreased to {display_scale}%")
            elif key == ord('e'):
                # Перебор эффектов по кругу
                current_effect_idx = (current_effect_idx + 1) % len(BACKGROUND_EFFECTS)
                current_effect = BACKGROUND_EFFECTS[current_effect_idx]
                print(f"Effect changed to: {current_effect}")
            elif key == ord('c') and current_effect == 'color':
                # Смена цвета фона (только для эффекта 'color')
                current_color_idx = (current_color_idx + 1) % len(BACKGROUND_COLORS)
                current_color_name = list(BACKGROUND_COLORS.keys())[current_color_idx]
                effect_params['color'] = BACKGROUND_COLORS[current_color_name]
                print(f"Background color changed to: {current_color_name}")
            elif key == ord('s'):
                show_contour = not show_contour
                status = "enabled" if show_contour else "disabled"
                print(f"Segmentation contour {status}")
            elif key == ord('b'):
                # Запрос пути к фоновому изображению
                image_path = input("Enter path to background image (or press Enter to skip): ")
                if image_path.strip():
                    bg_image = load_background_image(image_path, original_frame.shape[:2])
                    if bg_image is not None:
                        effect_params['background_image'] = bg_image
                        current_effect = 'image'
                        current_effect_idx = BACKGROUND_EFFECTS.index('image')
                        print(f"Background image loaded and effect set to 'image'")
                    else:
                        print("Failed to load background image")
            elif key == ord('r'):
                # Сброс настроек
                display_scale = 50
                current_effect_idx = 0
                current_effect = BACKGROUND_EFFECTS[current_effect_idx]
                current_color_idx = 0
                current_color_name = list(BACKGROUND_COLORS.keys())[current_color_idx]
                show_contour = True
                effect_params = {}
                if background_image:
                    effect_params['background_image'] = background_image
                print("Settings reset to default")
        
        # Освобождение ресурсов
        cap.release()
        if save_output:
            out.release()
        cv2.destroyAllWindows()
        
        # Вывод итоговой статистики
        print("=" * 60)
        print("PROCESSING COMPLETED")
        print(f"Total frames processed: {frame_count}")
        if elapsed_time > 0:
            print(f"Average FPS: {frame_count/elapsed_time:.2f}")
        print(f"Final effect: {current_effect}")
        if current_effect == 'color':
            print(f"Final background color: {current_color_name}")
        print("Resources released.")
        print("=" * 60)

if __name__ == "__main__":
    main()