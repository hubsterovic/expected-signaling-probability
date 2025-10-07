from expected_signaling_probability.utils.directions import Direction
from expected_signaling_probability.utils.params import ExtraParams
from dataclasses import fields
from pathlib import Path
import pandas as pd


class Cache:
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def _make_filename(self, d_A: int, d_B: int, direction: Direction, extra_params: ExtraParams) -> str:
        filename_parts = [f"dA={d_A}", f"dB={d_B}", f"direction={direction.to_str()}"]

        for field in fields(extra_params):
            value = getattr(extra_params, field.name)
            if value is not None:
                filename_parts.append(f"{field.name}={value}")

        return " ".join(filename_parts) + ".csv"

    def _load_or_create_dataframe(self, cache_file: Path) -> pd.DataFrame:
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file)
                if "seed" not in df.columns or "value" not in df.columns:
                    df = pd.DataFrame({"seed": [], "value": []})
                return df
            except Exception as e:
                print(f"Warning: Failed to load cache file {cache_file}: {e}")
                return pd.DataFrame({"seed": [], "value": []})
        else:
            return pd.DataFrame({"seed": [], "value": []})

    def get(self, d_A: int, d_B: int, direction: Direction, seed: int, extra_params: ExtraParams) -> float | None:
        filename = self._make_filename(d_A, d_B, direction, extra_params)
        cache_file = self.cache_dir / filename

        if not cache_file.exists():
            return None

        try:
            df = self._load_or_create_dataframe(cache_file)
            matching_rows = df[df["seed"] == seed]
            if len(matching_rows) == 1:
                return float(matching_rows.iloc[0]["value"])
            elif len(matching_rows) > 1:
                print(f"WARNING: Found {len(matching_rows)} rows for seed {seed} in {cache_file}")
                return float(matching_rows.iloc[0]["value"])
            return None
        except Exception as e:
            print(f"ERROR: Failed to get cache result from {cache_file}: {e}")
            return None

    def set(self, d_A: int, d_B: int, direction: Direction, seed: int, value: float, extra_params: ExtraParams):
        filename = self._make_filename(d_A, d_B, direction, extra_params)
        cache_file = self.cache_dir / filename

        try:
            df = self._load_or_create_dataframe(cache_file)
            existing_mask = df["seed"] == seed
            if existing_mask.any():
                df.loc[existing_mask, "value"] = value
            else:
                new_row = pd.DataFrame({"seed": [seed], "value": [value]})
                df = pd.concat([df, new_row], ignore_index=True)

            df = df.sort_values("seed").reset_index(drop=True)
            df.to_csv(cache_file, index=False)

        except Exception as e:
            print(f"Warning: Failed to cache result to {cache_file}: {e}")

    def get_all_seeds(self, d_A: int, d_B: int, direction: Direction, extra_params: ExtraParams) -> pd.DataFrame | None:
        filename = self._make_filename(d_A, d_B, direction, extra_params)
        cache_file = self.cache_dir / filename

        if not cache_file.exists():
            return None

        try:
            return self._load_or_create_dataframe(cache_file)
        except Exception as e:
            print(f"Warning: Failed to load all seeds from {cache_file}: {e}")
            return None


CACHE = Cache()
