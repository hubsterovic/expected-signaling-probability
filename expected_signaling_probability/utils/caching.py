from expected_signaling_probability.utils.directions import Direction
from expected_signaling_probability.utils.params import ExtraParams
from dataclasses import fields
from pathlib import Path
import pandas as pd


class Cache:
    def __init__(self, label: str):
        self.label = label
        self.cache_dir = Path(f"data/cache/{self.label}")
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        self._warm_seed_to_value: dict[int, float] | None = None  # seed -> value

    def _make_filename(self, d_A: int, d_B: int, direction: Direction, extra_params: ExtraParams) -> str:
        filename_parts = [f"{self.label}", f"dA={d_A}", f"dB={d_B}", f"direction={direction.to_str()}"]

        for field in fields(extra_params):
            value = getattr(extra_params, field.name)
            if value is not None:
                filename_parts.append(f"{field.name}={value}")

        return " ".join(filename_parts) + ".csv"

    def _load_or_create_dataframe(self, cache_file: Path) -> pd.DataFrame:
        if cache_file.exists():
            df = pd.read_csv(cache_file)
            return df
        else:
            return pd.DataFrame({"seed": [], "value": []})

    def warm(self, d_A: int, d_B: int, direction: Direction, extra_params: ExtraParams) -> None:
        if self._warm_seed_to_value is not None:
            self._warm_seed_to_value.clear()

        filename = self._make_filename(d_A, d_B, direction, extra_params)
        cache_file = self.cache_dir / filename

        df = self._load_or_create_dataframe(cache_file)
        seed_to_value: dict[int, float] = {}
        if len(df) > 0:
            for idx, seed in enumerate(df["seed"].tolist()):
                seed_to_value[int(seed)] = df.iloc[idx]["value"]

        self._warm_seed_to_value = seed_to_value

    def close(self) -> None:
        if self._warm_seed_to_value is not None:
            self._warm_seed_to_value.clear()

        self._warm_seed_to_value = None

    def get(self, d_A: int, d_B: int, direction: Direction, seed: int, extra_params: ExtraParams) -> float | None:
        filename = self._make_filename(d_A, d_B, direction, extra_params)
        cache_file = self.cache_dir / filename

        if self._warm_seed_to_value:
            value = self._warm_seed_to_value.get(int(seed))
            return value

        if not cache_file.exists():
            return None

        df = self._load_or_create_dataframe(cache_file)
        matching_rows = df[df["seed"] == seed]
        if len(matching_rows) == 1:
            return float(matching_rows.iloc[0]["value"])
        elif len(matching_rows) > 1:
            print(f"WARNING: Found {len(matching_rows)} rows for seed {seed} in {cache_file}")
            return float(matching_rows.iloc[0]["value"])
        return None

    def set(self, d_A: int, d_B: int, direction: Direction, seed: int, value: float, extra_params: ExtraParams):
        filename = self._make_filename(d_A, d_B, direction, extra_params)
        cache_file = self.cache_dir / filename

        if self._warm_seed_to_value:
            self._warm_seed_to_value[int(seed)] = value

        df = self._load_or_create_dataframe(cache_file)
        existing_mask = df["seed"] == seed
        if existing_mask.any():
            df.loc[existing_mask, "value"] = value
        else:
            new_row = pd.DataFrame({"seed": [seed], "value": [value]})
            df = pd.concat([df, new_row], ignore_index=True)

        df = df.sort_values("seed").reset_index(drop=True)
        df.to_csv(cache_file, index=False)


SIGNALING_CACHE = Cache(label="S")
REDUCED_DISTINGUISHABILITY_CACHE = Cache(label="R")
DISTINGUISHABILITY_CACHE = Cache(label="D")