import hashlib
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class FeatureCache:
    """Disk cache for SAM2 backbone multi-scale features."""

    def __init__(self, cache_dir: str = "feature_cache", enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Feature cache enabled at: %s", self.cache_dir)

    def _get_key(self, tensor: torch.Tensor) -> str:
        """Hash tensor stats for a cache key."""
        with torch.no_grad():
            checksum = torch.sum(tensor).item()
            mean = torch.mean(tensor).item()
            std = torch.std(tensor).item()
            hash_str = f"{tensor.shape}_{checksum:.4f}" f"_{mean:.4f}_{std:.4f}"
        return hashlib.md5(hash_str.encode()).hexdigest()

    def get(self, images: torch.Tensor):
        """Return cached features or None."""
        if not self.enabled:
            return None

        key = self._get_key(images)
        path = self.cache_dir / f"{key}.pt"
        if path.exists():
            try:
                return torch.load(
                    path,
                    map_location=images.device,
                    weights_only=True,
                )
            except Exception as e:
                logger.warning("Cache read failed for %s: %s", key, e)
        return None

    def put(self, images: torch.Tensor, features: list):
        """Store features to disk cache."""
        if not self.enabled:
            return

        key = self._get_key(images)
        path = self.cache_dir / f"{key}.pt"
        if path.exists():
            return  # Already cached

        try:
            cpu_features = [f.detach().cpu() for f in features]
            torch.save(cpu_features, path)
        except Exception as e:
            logger.warning("Cache write failed for %s: %s", key, e)

    def clear(self):
        """Remove all cached feature files."""
        if self.cache_dir.exists():
            for f in self.cache_dir.glob("*.pt"):
                try:
                    f.unlink()
                except Exception:
                    pass
            logger.info("Feature cache cleared.")
