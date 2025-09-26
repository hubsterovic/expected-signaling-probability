from expected_signaling_probability.utils.directions import Direction
from pathlib import Path
import json


class Cache:
    def __init__(self, cache_dir: str = ".esp_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _make_filename(
        self, d_A: int, d_B: int, direction: Direction, seed: int
    ) -> str:
        return f"dA{d_A}_dB{d_B}_{direction.to_str()}seed{seed}.json"

    def get(self, d_A: int, d_B: int, direction: Direction, seed: int) -> float | None:
        filename = self._make_filename(d_A, d_B, direction, seed)
        cache_file = self.cache_dir / filename

        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    return data["value"]
            except Exception as e:
                print(f"Warning: Failed to get cache result: {e}")
                cache_file.unlink(missing_ok=True)
        return None

    def set(self, d_A: int, d_B: int, direction: Direction, seed: int, value: float):
        filename = self._make_filename(d_A, d_B, direction, seed)
        cache_file = self.cache_dir / filename

        data = {
            "d_A": d_A,
            "d_B": d_B,
            "direction": direction.value,
            "seed": seed,
            "value": value,
        }

        try:
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            print(f"Warning: Failed to cache result: {e}")

    def get_cache_stats(self) -> dict[str, float]:
        """Get statistics about the cache."""
        cache_files = list(self.cache_dir.glob("*.json"))

        return {
            "cached_computations": len(cache_files),
            "cache_size_mb": sum(f.stat().st_size for f in cache_files) / (1024 * 1024),
        }

    def clear_cache(self):
        """Clear all cached results."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
