"""
Утилиты для сжатия landmark-векторов (PCA, learned projection).
"""

from typing import Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA


class LandmarkCompressor:
    """
    Класс для сжатия landmark-векторов с использованием PCA.
    """
    
    def __init__(self, n_components: int = 16, use_pca: bool = True):
        """
        :param n_components: количество компонент PCA
        :param use_pca: использовать ли PCA (если False - просто берет первые n_components)
        """
        self.n_components = n_components
        self.use_pca = use_pca
        self.pca: Optional[PCA] = None
        self._fitted = False
    
    def fit(self, landmarks_list: list) -> None:
        """
        Обучает PCA на списке landmark-векторов.
        
        :param landmarks_list: список векторов landmarks (каждый - плоский массив координат)
        """
        if not self.use_pca or len(landmarks_list) < 2:
            self._fitted = True
            return
        
        # Преобразуем в numpy array
        X = np.array(landmarks_list)
        if X.shape[0] < 2:
            self._fitted = True
            return
        
        # Обучаем PCA
        n_components = min(self.n_components, X.shape[1], X.shape[0] - 1)
        if n_components > 0:
            self.pca = PCA(n_components=n_components)
            self.pca.fit(X)
            self._fitted = True
    
    def transform(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Сжимает landmark-вектор.
        
        :param landmarks: вектор landmarks (плоский массив координат)
        :return: сжатый вектор
        """
        if not self._fitted:
            # Если не обучен, просто берем первые n_components
            return landmarks[:self.n_components] if len(landmarks) > self.n_components else landmarks
        
        if self.use_pca and self.pca is not None:
            landmarks_2d = landmarks.reshape(1, -1)
            compressed = self.pca.transform(landmarks_2d)
            return compressed.flatten()
        else:
            # Fallback: просто берем первые n_components
            return landmarks[:self.n_components] if len(landmarks) > self.n_components else landmarks
    
    def compress_face_shape(self, face_shape_vector: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
        """
        Сжимает face_shape_vector (36 точек * 2 координаты = 72 dims -> n_components).
        
        :param face_shape_vector: вектор формы лица (72 dims)
        :param n_components: количество компонент (по умолчанию self.n_components)
        :return: сжатый вектор
        """
        n_comp = n_components or self.n_components
        if len(face_shape_vector) <= n_comp:
            return face_shape_vector
        
        # Если PCA не обучен, используем простую проекцию (первые n_comp)
        if not self._fitted or self.pca is None:
            return face_shape_vector[:n_comp]
        
        # Используем обученный PCA
        face_shape_2d = face_shape_vector.reshape(1, -1)
        compressed = self.pca.transform(face_shape_2d)
        return compressed.flatten()[:n_comp]


def simple_landmark_projection(landmarks: np.ndarray, target_dim: int = 16) -> np.ndarray:
    """
    Простая проекция landmarks без обучения PCA (берет первые target_dim компонент).
    
    :param landmarks: вектор landmarks
    :param target_dim: целевая размерность
    :return: сжатый вектор
    """
    if len(landmarks) <= target_dim:
        return landmarks
    
    # Берем первые target_dim компонент
    return landmarks[:target_dim]

