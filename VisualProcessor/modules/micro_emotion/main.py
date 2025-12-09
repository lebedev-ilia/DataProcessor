import os
import sys
import subprocess
import json
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
from datetime import datetime

class OpenFaceAnalyzer:
    def __init__(self, docker_image="openface/openface:latest"):
        """
        Инициализация анализатора OpenFace.
        
        Args:
            docker_image: имя Docker образа OpenFace
        """
        self.docker_image = docker_image
        self.base_dir = Path(__file__).parent.parent
        
        # Директории
        self.input_dir = self.base_dir / "input_videos"
        self.output_dir = self.base_dir / "output_data"
        self.temp_dir = self.base_dir / "temp_frames"
        
        # Создаем директории если их нет
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Проверяем доступность Docker
        self.check_docker()
    
    def check_docker(self):
        """Проверяет доступность Docker и образа."""
        try:
            # Проверяем Docker
            result = subprocess.run(["docker", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError("Docker не установлен или не запущен")
            print(f"[INFO] Docker доступен: {result.stdout.strip()}")
            
            # Проверяем образ OpenFace
            result = subprocess.run(["docker", "images", "openface/openface:latest", "--quiet"], capture_output=True, text=True)
            if not result.stdout.strip():
                raise RuntimeError("Образ OpenFace не найден. Скачайте: docker pull openface/openface:latest")
            print("[INFO] Образ OpenFace доступен")
            
        except FileNotFoundError:
            raise RuntimeError("Docker не установлен. Установите: sudo apt install docker.io")
    
    def analyze_video(self, video_path, output_name=None, features="all"):
        """
        Анализирует видео файл с помощью OpenFace.
        
        Args:
            video_path: путь к видео файлу
            output_name: имя для выходных файлов (без расширения)
            features: какие признаки извлекать ("all", "basic", "au", "pose", "gaze")
        
        Returns:
            dict: словарь с результатами анализа
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Видео не найдено: {video_path}")
        
        # Определяем имя выходного файла
        if output_name is None:
            output_name = video_path.stem
        
        # Копируем видео во входную директорию
        target_video = self.input_dir / video_path.name
        if not target_video.exists() or target_video.stat().st_size != video_path.stat().st_size:
            import shutil
            print(f"[INFO] Копирую видео: {video_path} -> {target_video}")
            shutil.copy2(video_path, target_video)
        
        # Параметры для FeatureExtraction
        feature_flags = {
            "all": "-pose -aus -gaze -2Dfp -3Dfp -tracked",
            "basic": "-pose -2Dfp",
            "au": "-aus",
            "pose": "-pose",
            "gaze": "-gaze"
        }
        
        flags = feature_flags.get(features, feature_flags["all"])
        
        # Создаем команду для Docker
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{self.input_dir.absolute()}:/input",
            "-v", f"{self.output_dir.absolute()}:/output",
            self.docker_image,
            "/usr/local/bin/FeatureExtraction",
            "-f", f"/input/{video_path.name}",
            "-out_dir", "/output",
            "-of", f"/output/{output_name}",
        ] + flags.split()
        
        print(f"[INFO] Запускаю OpenFace анализ: {' '.join(cmd)}")
        
        # Запускаем анализ
        start_time = datetime.now()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = (datetime.now() - start_time).total_seconds()
        
        if result.returncode != 0:
            print(f"[ERROR] OpenFace завершился с ошибкой:")
            print(result.stderr[:500])  # Показываем первые 500 символов ошибки
            return None
        
        print(f"[INFO] Анализ завершен за {elapsed:.2f} секунд")
        
        # Читаем результаты
        results = self.load_results(output_name)
        
        return results
    
    def analyze_frames(self, frames, frame_indices=None, output_prefix="frames"):
        """
        Анализирует список кадров (numpy arrays).
        
        Args:
            frames: список numpy arrays (каждое изображение BGR)
            frame_indices: индексы кадров (если None, используются 0,1,2,...)
            output_prefix: префикс для выходных файлов
        
        Returns:
            list: список результатов для каждого кадра
        """
        if frame_indices is None:
            frame_indices = list(range(len(frames)))
        
        all_results = []
        
        for i, (frame, idx) in enumerate(zip(frames, frame_indices)):
            print(f"[INFO] Анализ кадра {i+1}/{len(frames)} (индекс {idx})")
            
            # Сохраняем кадр как временное изображение
            frame_filename = f"{output_prefix}_frame_{idx:06d}.png"
            frame_path = self.temp_dir / frame_filename
            
            # Конвертируем BGR в RGB для сохранения
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(frame_path), frame_rgb)
            
            # Анализируем кадр
            result = self.analyze_single_image(frame_path, output_prefix=f"{output_prefix}_{idx}")
            
            if result is not None:
                result['frame_index'] = idx
                all_results.append(result)
            
            # Удаляем временный файл
            frame_path.unlink(missing_ok=True)
        
        return all_results
    
    def analyze_single_image(self, image_path, output_prefix="image"):
        """
        Анализирует одно изображение.
        
        Args:
            image_path: путь к изображению
            output_prefix: префикс для выходных файлов
        
        Returns:
            dict: результаты анализа
        """
        image_path = Path(image_path)
        
        # Копируем изображение во входную директорию
        target_image = self.input_dir / image_path.name
        import shutil
        shutil.copy2(image_path, target_image)
        
        # Команда для анализа изображения
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{self.input_dir.absolute()}:/input",
            "-v", f"{self.output_dir.absolute()}:/output",
            self.docker_image,
            "./build/bin/FaceLandmarkImg",
            "-f", f"/input/{image_path.name}",
            "-out_dir", f"/output",
            "-of", f"/output/{output_prefix}"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Удаляем копию изображения
        target_image.unlink(missing_ok=True)
        
        if result.returncode != 0:
            print(f"[WARNING] Не удалось проанализировать изображение: {image_path.name}")
            return None
        
        # Загружаем результаты
        csv_path = self.output_dir / f"{output_prefix}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                return self.extract_features_from_df(df)
            except Exception as e:
                print(f"[ERROR] Ошибка чтения CSV: {e}")
                return None
        
        return None
    
    def load_results(self, output_name):
        """
        Загружает результаты анализа из CSV файла.
        
        Args:
            output_name: имя выходного файла (без расширения)
        
        Returns:
            dict: словарь с результатами
        """
        csv_path = self.output_dir / f"{output_name}.csv"
        
        if not csv_path.exists():
            print(f"[WARNING] CSV файл не найден: {csv_path}")
            return None
        
        try:
            df = pd.read_csv(csv_path)
            print(f"[INFO] Загружено {len(df)} кадров из {csv_path}")
            
            # Извлекаем основные признаки
            features = self.extract_features_from_df(df)
            
            # Сохраняем также полный DataFrame
            features['dataframe'] = df
            features['csv_path'] = str(csv_path)
            features['frame_count'] = len(df)
            
            return features
            
        except Exception as e:
            print(f"[ERROR] Ошибка загрузки CSV: {e}")
            return None
    
    def extract_features_from_df(self, df):
        """
        Извлекает ключевые признаки из DataFrame OpenFace.
        
        Args:
            df: DataFrame с результатами OpenFace
        
        Returns:
            dict: извлеченные признаки
        """
        features = {
            'success': False,
            'face_count': 0,
            'action_units': {},
            'pose': {},
            'gaze': {},
            'facial_landmarks_2d': [],
            'facial_landmarks_3d': [],
            'emotions': {},
            'summary': {}
        }
        
        if df.empty:
            return features
        
        # Проверяем наличие лиц
        if 'success' in df.columns:
            success_frames = df['success'].sum()
            features['success'] = success_frames > 0
            features['face_count'] = int(success_frames)
            features['success_rate'] = float(success_frames / len(df))
        
        # Action Units (AU)
        au_columns = [col for col in df.columns if col.startswith('AU') and col.endswith('_r')]
        for au_col in au_columns:
            au_name = au_col.replace('_r', '')
            if au_name in df.columns:  # Есть ли интенсивность
                features['action_units'][au_name] = {
                    'intensity_mean': float(df[au_name].mean()),
                    'intensity_std': float(df[au_name].std()),
                    'presence_mean': float(df[au_col].mean()),
                    'presence_std': float(df[au_col].std())
                }
        
        # Pose (положение головы)
        pose_columns = ['pose_Rx', 'pose_Ry', 'pose_Rz', 'pose_Tx', 'pose_Ty', 'pose_Tz']
        for col in pose_columns:
            if col in df.columns:
                features['pose'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
        
        # Gaze (направление взгляда)
        gaze_columns = ['gaze_angle_x', 'gaze_angle_y']
        for col in gaze_columns:
            if col in df.columns:
                features['gaze'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std())
                }
        
        # Facial Landmarks (2D и 3D)
        # 2D landmarks (x_0, y_0, x_1, y_1, ...)
        landmark_2d_cols = [col for col in df.columns if col.startswith('x_') or col.startswith('y_')]
        if landmark_2d_cols:
            # Группируем по точкам
            points_2d = []
            for i in range(68):  # OpenFace использует 68 точек
                x_col = f'x_{i}'
                y_col = f'y_{i}'
                if x_col in df.columns and y_col in df.columns:
                    point_data = {
                        'x_mean': float(df[x_col].mean()),
                        'x_std': float(df[x_col].std()),
                        'y_mean': float(df[y_col].mean()),
                        'y_std': float(df[y_col].std())
                    }
                    points_2d.append(point_data)
            features['facial_landmarks_2d'] = points_2d
        
        # 3D landmarks (X_0, Y_0, Z_0, ...)
        landmark_3d_cols = [col for col in df.columns if col.startswith('X_') or col.startswith('Y_') or col.startswith('Z_')]
        if landmark_3d_cols:
            points_3d = []
            for i in range(68):
                x_col = f'X_{i}'
                y_col = f'Y_{i}'
                z_col = f'Z_{i}'
                if all(col in df.columns for col in [x_col, y_col, z_col]):
                    point_data = {
                        'X_mean': float(df[x_col].mean()),
                        'Y_mean': float(df[y_col].mean()),
                        'Z_mean': float(df[z_col].mean())
                    }
                    points_3d.append(point_data)
            features['facial_landmarks_3d'] = points_3d
        
        # Сводная статистика
        features['summary'] = {
            'total_frames': len(df),
            'frames_with_face': int(df['success'].sum()) if 'success' in df.columns else 0,
            'au_count': len(features['action_units']),
            'landmarks_2d_count': len(features['facial_landmarks_2d']),
            'landmarks_3d_count': len(features['facial_landmarks_3d']),
            'timestamp': datetime.now().isoformat()
        }
        
        return features
    
    def save_results(self, results, output_path=None):
        """Упрощенная версия сохранения результатов."""
        import json
        from datetime import datetime
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = "/home/ilya/Рабочий стол/DataProcessor/VisualProcessor2.0/other/micro_emotion/results/openface_results_{timestamp}.json"
        
        # Функция для конвертации всех типов в JSON-совместимые
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            elif isinstance(obj, (int, np.integer)):
                return int(obj)
            elif isinstance(obj, (float, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (str, Path)):
                return str(obj)
            elif hasattr(obj, '__str__'):
                return str(obj)
            else:
                return obj
        
        # Удаляем dataframe если есть
        results_copy = results.copy()
        if 'dataframe' in results_copy:
            del results_copy['dataframe']
        
        # Конвертируем
        serializable_results = make_serializable(results_copy)
        
        # Сохраняем
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Результаты сохранены в: {output_path}")
        return str(output_path)

def main():
    """Пример использования."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Анализ видео с OpenFace через Docker')
    parser.add_argument('--video_path', help='Путь к видео файлу', default="-NSumhkOwSg.mp4")
    parser.add_argument('--output', '-o', help='Имя выходного файла (без расширения)', default="test_analysis")
    parser.add_argument('--features', '-f', default='all', 
                       choices=['all', 'basic', 'au', 'pose', 'gaze'],
                       help='Какие признаки извлекать')
    
    args = parser.parse_args()
    
    # Создаем анализатор
    analyzer = OpenFaceAnalyzer()
    
    # Анализируем видео
    print(f"[INFO] Начинаю анализ: {args.video_path}")
    results = analyzer.analyze_video(
        video_path=args.video_path,
        output_name=args.output,
        features=args.features
    )
    
    if results:
        # Сохраняем результаты
        json_path = analyzer.save_results(results)
        
        # Выводим сводку
        summary = results.get('summary', {})
        print(f"\n=== СВОДКА АНАЛИЗА ===")
        print(f"Всего кадров: {summary.get('total_frames', 0)}")
        print(f"Кадров с лицами: {summary.get('frames_with_face', 0)}")
        print(f"Обнаружено AU: {summary.get('au_count', 0)}")
        print(f"Файл результатов: {json_path}")
        
        # Пример извлеченных AU
        if results['action_units']:
            print("\n=== ACTION UNITS (средняя интенсивность) ===")
            for au, data in list(results['action_units'].items())[:10]:  # Показываем первые 10
                print(f"{au}: {data['intensity_mean']:.3f}")
    else:
        print("[ERROR] Анализ завершился неудачно")

if __name__ == "__main__":
    main()