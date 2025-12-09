from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple, Union
import importlib
import importlib.util
import os
import time

from src.core.base_extractor import BaseExtractor
from src.schemas.models import VideoDocument, video_document_from_dict


class MainProcessor:
    """
    Класс-оркестратор: хранит список экстракторов и последовательно применяет их к документу.

    Инициализация поддерживает конфиг вида:
    {"cpu": "ExtractorName" | ["ExtractorName", ...], "gpu": "..." | ["..."]}
    В зависимости от ключа создаются экземпляры на указанном устройстве.
    """

    def __init__(
        self,
        extractors: List[BaseExtractor] | None = None,
        devices_config: Dict[str, Union[str, List[str]]] | None = None,
        extractor_params: Dict[str, Dict[str, Any]] | None = None,
    ) -> None:
        if extractors is not None:
            # Legacy mode: если переданы готовые экстракторы, используем их напрямую
            self._extractor_configs: List[Tuple[str, str, Dict[str, Any]]] | None = None
            self.extractors: List[BaseExtractor] = extractors
            return

        self._extractor_params = extractor_params or {}
        self.extractors: List[BaseExtractor] = []  # экстракторы не создаются в __init__

        print("Devices config: ", devices_config)   

        if devices_config:
            # Сохраняем конфигурацию для ленивой инициализации
            self._extractor_configs = self._build_extractor_configs(devices_config)
            print(f"(init) Extractor configs prepared: {len(self._extractor_configs)} extractors")
        else:
            print("No devices config, using default")
            # по умолчанию — один TitleEmbedder с авто-выбором устройства
            self._extractor_configs = [("TitleEmbedder", "cuda", {})]

    def _build_extractor_configs(self, config: Dict[str, Union[str, List[str]]]) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Строит список конфигураций экстракторов (name, device, params) без их создания.
        """
        def to_list(x: Union[str, List[str]]) -> List[str]:
            return x if isinstance(x, list) else [x]

        configs: List[Tuple[str, str, Dict[str, Any]]] = []
        for device_key, names in config.items():
            device = "cuda" if device_key.lower() in ("gpu", "cuda") else "cpu"
            for name in to_list(names):
                params = dict(self._extractor_params.get(name, {}))
                configs.append((name, device, params))
        return configs

    def _get_registry_entry(self, name: str) -> Tuple[List[str], str, str] | None:
        """
        Возвращает ([module_paths], class_name, relative_file_path) для зарегистрированного экстрактора.
        Здесь перечисляем доступные экстракторы, но импорт происходит лениво.
        """
        registry: Dict[str, Tuple[List[str], str, str]] = {
            "TagsExtractor": (["src.extractors.simantic_embeddings.tags_extractor"], "TagsExtractor", os.path.join("src", "extractors", "simantic_embeddings", "tags_extractor.py")),
            "TitleEmbedder": (["src.extractors.simantic_embeddings.title_embedder"], "TitleEmbedder", os.path.join("src", "extractors", "simantic_embeddings", "title_embedder.py")),
            "DescriptionEmbedder": (["src.extractors.simantic_embeddings.description_embedder"], "DescriptionEmbedder", os.path.join("src", "extractors", "simantic_embeddings", "description_embedder.py")),
            "TranscriptChunkEmbedder": (["src.extractors.simantic_embeddings.transcript_chunk_embedder"], "TranscriptChunkEmbedder", os.path.join("src", "extractors", "simantic_embeddings", "transcript_chunk_embedder.py")),
            "TranscriptAggregatorExtractor": (["src.extractors.simantic_embeddings.transcript_aggregator"], "TranscriptAggregatorExtractor", os.path.join("src", "extractors", "simantic_embeddings", "transcript_aggregator.py")),
            "CommentsEmbedder": (["src.extractors.simantic_embeddings.comments_embedder"], "CommentsEmbedder", os.path.join("src", "extractors", "simantic_embeddings", "comments_embedder.py")),
            "CommentsAggregationExtractor": (["src.extractors.simantic_embeddings.comments_aggregator"], "CommentsAggregationExtractor", os.path.join("src", "extractors", "simantic_embeddings", "comments_aggregator.py")),
            "HashtagEmbedder": (["src.extractors.simantic_embeddings.hashtag_embedder"], "HashtagEmbedder", os.path.join("src", "extractors", "simantic_embeddings", "hashtag_embedder.py")),
            "CosineMetricsExtractor": (["src.extractors.simantic_embeddings.cosine_metrics_extractor"], "CosineMetricsExtractor", os.path.join("src", "extractors", "simantic_embeddings", "cosine_metrics_extractor.py")),
            "EmbeddingPairTopKExtractor": (["src.extractors.simantic_embeddings.embedding_pair_topk_extractor"], "EmbeddingPairTopKExtractor", os.path.join("src", "extractors", "simantic_embeddings", "embedding_pair_topk_extractor.py")),
            "SemanticClusterExtractor": (["src.extractors.simantic_embeddings.semantic_cluster_extractor"], "SemanticClusterExtractor", os.path.join("src", "extractors", "simantic_embeddings", "semantic_cluster_extractor.py")),
            "EmbeddingStatsExtractor": (["src.extractors.simantic_embeddings.embedding_stats_extractor"], "EmbeddingStatsExtractor", os.path.join("src", "extractors", "simantic_embeddings", "embedding_stats_extractor.py")),
            "TitleToHashtagCosineExtractor": (["src.extractors.simantic_embeddings.title_to_hashtag_cosine_extractor"], "TitleToHashtagCosineExtractor", os.path.join("src", "extractors", "simantic_embeddings", "title_to_hashtag_cosine_extractor.py")),
            "TopKSimilarCorpusTitlesExtractor": (["src.extractors.simantic_embeddings.topk_similar_titles_extractor"], "TopKSimilarCorpusTitlesExtractor", os.path.join("src", "extractors", "simantic_embeddings", "topk_similar_titles_extractor.py")),
            "LongformEmbeddingSummaryExtractor": (["src.extractors.simantic_embeddings.longform_embedding_summary_extractor"], "LongformEmbeddingSummaryExtractor", os.path.join("src", "extractors", "simantic_embeddings", "longform_embedding_summary_extractor.py")),
            "SpeakerTurnEmbeddingsAggregatorExtractor": (["src.extractors.simantic_embeddings.speaker_turn_embeddings_aggregator"], "SpeakerTurnEmbeddingsAggregatorExtractor", os.path.join("src", "extractors", "simantic_embeddings", "speaker_turn_embeddings_aggregator.py")),
            "QAEmbeddingPairsExtractor": (["src.extractors.simantic_embeddings.qa_embedding_pairs_extractor"], "QAEmbeddingPairsExtractor", os.path.join("src", "extractors", "simantic_embeddings", "qa_embedding_pairs_extractor.py")),
            "EmbeddingShiftIndicatorExtractor": (["src.extractors.simantic_embeddings.embedding_shift_indicator_extractor"], "EmbeddingShiftIndicatorExtractor", os.path.join("src", "extractors", "simantic_embeddings", "embedding_shift_indicator_extractor.py")),
            "TitleEmbeddingClusterEntropyExtractor": (["src.extractors.simantic_embeddings.title_embedding_cluster_entropy_extractor"], "TitleEmbeddingClusterEntropyExtractor", os.path.join("src", "extractors", "simantic_embeddings", "title_embedding_cluster_entropy_extractor.py")),
            "EmbeddingSourceIdExtractor": (["src.extractors.embedding.embedding_source_id_extractor"], "EmbeddingSourceIdExtractor", os.path.join("src", "extractors", "embedding", "embedding_source_id_extractor.py")),
            "LexicalStatsExtractor": (["src.extractors.lexico_static_features.lexical_stats_extractor"], "LexicalStatsExtractor", os.path.join("src", "extractors", "lexico_static_features", "lexical_stats_extractor.py")),
            "ASRTextProxyExtractor": (["src.extractors.asr_text_proxy_audio_features.asr_text_proxy_extractor"], "ASRTextProxyExtractor", os.path.join("src", "extractors", "asr_text_proxy_audio_features", "asr_text_proxy_extractor.py")),
            "SemanticTopicExtractor": (["src.extractors.semantics_topics_keyphrases.semantic_topic_extractor"], "SemanticTopicExtractor", os.path.join("src", "extractors", "semantics_topics_keyphrases", "semantic_topic_extractor.py")),
        }
        return registry.get(name)

    def _instantiate_extractor_by_name(self, name: str, device: str | None) -> BaseExtractor | None:
        entry = self._get_registry_entry(name)
        if not entry:
            return None
        module_paths, class_name, rel_file = entry
        cls = None
        last_err: Exception | None = None
        for module_path in module_paths:
            try:
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                break
            except Exception as e:
                last_err = e
                print("(instantiate) Error: ", e)
                continue
        if cls is None:
            # Fallback: import by absolute file path
            try:
                # project root = two levels up from this file (src/core → project)
                this_dir = os.path.dirname(__file__)
                project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
                file_path = os.path.join(project_root, rel_file)
                if os.path.exists(file_path):
                    spec = importlib.util.spec_from_file_location(f"dyn_{class_name}", file_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)  # type: ignore[attr-defined]
                        cls = getattr(module, class_name)
            except Exception:
                cls = None
            if cls is None:
                return None
        # собрать kwargs: из конфигурации + возможно device
        params_local = dict(self._extractor_params.get(name, {}))
        if device is not None:
            params_local.setdefault("device", device)
        try:
            return cls(**params_local)
        except TypeError:
            # конструктор без аргументов device
            try:
                return cls()
            except Exception:
                return None
    
    def _instantiate_extractor_by_name_with_params(self, name: str, device: str | None, params: Dict[str, Any]) -> BaseExtractor | None:
        """
        Вспомогательный метод для создания экстрактора с явными параметрами.
        """
        entry = self._get_registry_entry(name)
        if not entry:
            return None
        module_paths, class_name, rel_file = entry
        cls = None
        last_err: Exception | None = None
        for module_path in module_paths:
            try:
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                break
            except Exception as e:
                last_err = e
                print("(instantiate) Error: ", e)
                continue
        if cls is None:
            # Fallback: import by absolute file path
            try:
                this_dir = os.path.dirname(__file__)
                project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
                file_path = os.path.join(project_root, rel_file)
                if os.path.exists(file_path):
                    spec = importlib.util.spec_from_file_location(f"dyn_{class_name}", file_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)  # type: ignore[attr-defined]
                        cls = getattr(module, class_name)
            except Exception:
                cls = None
            if cls is None:
                return None
        
        # Используем переданные params + device
        params_final = dict(params)
        if device is not None:
            params_final.setdefault("device", device)
        try:
            return cls(**params_final)
        except TypeError:
            try:
                return cls()
            except Exception:
                return None

    def run(self, document: VideoDocument) -> Dict[str, Any]:
        features: Dict[str, Any] = {}
        
        # Определяем какой список использовать: готовые экстракторы или конфигурации
        if self._extractor_configs is not None:
            # Ленивая инициализация: создаём экстрактор → используем → очищаем → удаляем
            extractor_specs = self._extractor_configs
        else:
            # Legacy mode: используем уже созданные экстракторы
            extractor_specs = [(ext.__class__.__name__, getattr(ext, "device", "cpu"), {}) for ext in self.extractors]
        
        # mutable document to allow earlier extractors to influence later ones (e.g., cleaned texts, hashtags)
        current_doc = document
        for spec in extractor_specs:
            ext = None
            try:
                if self._extractor_configs is not None:
                    # Ленивая инициализация
                    name, device, params = spec
                    params = dict(params)
                    print(f"[run] Creating extractor: {name} on device: {device}")
                    t_init_start = time.perf_counter()
                    ext = self._instantiate_extractor_by_name_with_params(name, device=device, params=params)
                    init_s = round(time.perf_counter() - t_init_start, 3)
                    if ext is None:
                        print(f"[run] Failed to create extractor: {name}")
                        continue
                else:
                    # Legacy mode: находим экстрактор по имени
                    ext_name = spec[0] if isinstance(spec, tuple) else spec.__class__.__name__
                    ext = next((e for e in self.extractors if e.__class__.__name__ == ext_name), None)
                    if ext is None:
                        continue
                    init_s = 0.0
                
                # Выполняем извлечение
                part = ext.extract(current_doc) or {}

                # keep per-extractor system snapshots
                if "system" in part and isinstance(part["system"], dict):
                    features.setdefault("systems_by_extractor", {})
                    features["systems_by_extractor"][ext.__class__.__name__] = part["system"]

                # keep per-extractor results (separated by extractor)
                if "result" in part and isinstance(part["result"], dict):
                    features.setdefault("results_by_extractor", {})
                    features["results_by_extractor"][ext.__class__.__name__] = part["result"]
                    # propagate cleaned texts/hashtags to subsequent extractors
                    cleaned = part["result"].get("cleaned_texts") if isinstance(part["result"].get("cleaned_texts"), dict) else None
                    if cleaned:
                        try:
                            if "title" in cleaned:
                                setattr(current_doc, "title", cleaned["title"])
                            if "description" in cleaned:
                                setattr(current_doc, "description", cleaned["description"])
                        except Exception:
                            pass
                    tags = part["result"].get("hashtags") if isinstance(part["result"].get("hashtags"), list) else None
                    if tags is not None:
                        try:
                            setattr(current_doc, "hashtags", tags)
                        except Exception:
                            pass

                # keep per-extractor timings (seconds)
                if "timings_s" in part and isinstance(part["timings_s"], dict):
                    features.setdefault("timings_by_extractor", {})
                    tdict = dict(part["timings_s"])  # copy
                    # добавить время инициализации экстрактора
                    tdict.setdefault("init", init_s)
                    features["timings_by_extractor"][ext.__class__.__name__] = tdict
                else:
                    # нет таймингов от экстрактора — всё равно зафиксируем init
                    features.setdefault("timings_by_extractor", {})
                    features["timings_by_extractor"][ext.__class__.__name__] = {"init": init_s}

                # device/version: keep last non-empty
                if part.get("device"):
                    features["device"] = part.get("device")
                if part.get("version"):
                    features["version"] = part.get("version")

                # error: collect last non-empty (for backward compat); detailed errors can be found per extractor inside result if needed
                if part.get("error"):
                    features["error"] = part.get("error")

            finally:
                # Явно удаляем ссылку и запускаем сборщик мусора (без ручной очистки GPU)
                if ext is not None:
                    try:
                        del ext
                        import gc
                        gc.collect()
                    except Exception:
                        pass

            print()

        return features


def load_document_from_json(path: str) -> VideoDocument:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return video_document_from_dict(data)
