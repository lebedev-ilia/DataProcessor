import argparse
import os
import sys
from datetime import datetime
from typing import List, Tuple

import numpy as np      # type: ignore
import torch            # type: ignore
from PIL import Image   # type: ignore

_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _path not in sys.path:
    sys.path.append(_path)

from utils.frame_manager import FrameManager
from utils.logger import get_logger
from utils.utilites import load_metadata


NAME = "core_clip"
VERSION = "2.0"
LOGGER = get_logger(NAME)


def init_clip(model_name: str) -> Tuple[torch.nn.Module, callable, str]:
    import clip # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)

    model.eval()

    LOGGER.info(
        f"{NAME} | CLIP initialized | model: {model_name} | device: {device}"
    )

    return model, preprocess, device


def compute_clip_embeddings(
    frame_manager: FrameManager,
    frame_indices: List[int],
    model_name: str,
    batch_size: int,
) -> np.ndarray:

    if not frame_indices:
        LOGGER.warning(f"{NAME} | No frame indices provided")
        return np.zeros((0, 0), dtype=np.float32)

    model, preprocess, device = init_clip(model_name)

    n_frames = len(frame_indices)

    embeddings_out = None
    embed_dim = None

    try:
        with torch.no_grad():
            for start in range(0, n_frames, batch_size):
                batch_ids = frame_indices[start : start + batch_size]

                images = []
                for idx in batch_ids:
                    frame = frame_manager.get(idx)
                    img = Image.fromarray(frame)
                    images.append(preprocess(img))

                batch_tensor = torch.stack(images).to(device)
                
                emb = model.encode_image(batch_tensor)

                # L2 normalization (standard CLIP practice)
                emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-9)

                emb_np = emb.cpu().numpy().astype(np.float32)

                if embeddings_out is None:
                    embed_dim = emb_np.shape[1]
                    embeddings_out = np.zeros((n_frames, embed_dim), dtype=np.float32)

                embeddings_out[start : start + len(batch_ids)] = emb_np

                if start % (batch_size * 10) == 0:
                    LOGGER.info(
                        f"{NAME} | processed {start + len(batch_ids)}/{n_frames}"
                    )
    finally:
        del model
        torch.cuda.empty_cache()

    return embeddings_out


def main():
    parser = argparse.ArgumentParser(description="Production CLIP per-frame embedding extractor")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--rs-path", required=True)
    parser.add_argument("--model-name", default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    meta_path = os.path.join(args.frames_dir, "metadata.json")
    meta = load_metadata(meta_path, NAME)

    total_frames = int(meta["total_frames"])

    frame_indices = list(range(0, total_frames, 10))
    LOGGER.warning(
        f"{NAME} | Равномерная выборка | total: {total_frames} | sampled: {len(frame_indices)} | "
        "(Это фиксированая выборка для провайдера core_clip, но использовать ее могут разные модули с разной "
        "логикой извлечения фичей. В будующем нужно грамотно коректировать выборку для получения хорошего качества фичей на всех модулях)"
    )

    frame_manager = FrameManager(
        frames_dir=args.frames_dir,
        chunk_size=meta.get("chunk_size", 32),
        cache_size=meta.get("cache_size", 2),
    )

    embeddings = compute_clip_embeddings(
        frame_manager=frame_manager,
        frame_indices=frame_indices,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )

    LOGGER.info(
        f"{NAME} | embeddings computed | shape: {embeddings.shape}"
    )

    frame_manager.close()

    out_path = os.path.join(args.rs_path, "embeddings.npz")

    np.savez_compressed(
        out_path,
        version=VERSION,
        created_at=datetime.utcnow().isoformat(),
        model_name=args.model_name,
        total_frames=total_frames,
        frame_indices=np.array(frame_indices, dtype=np.int32),
        frame_embeddings=embeddings,
    )

    LOGGER.info(f"{NAME} | Saved result: {out_path}")


if __name__ == "__main__":
    main()