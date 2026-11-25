"""Create a fixed trajectory-level TartanAir split."""

import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "src")
)

from close_range.common import ROOT, portable_path
from close_range.training.main import load_split


def main():

    output = ROOT / "configs/tartanair_split.json"

    if output.exists():
        raise FileExistsError(output)

    groups = {}

    paths = sorted(
        (ROOT / "data/manifests/tartanair_v2").glob("*.json")
    )

    for path in paths:

        m = json.loads(path.read_text())

        assert m["source"] == "tartanair"

        key = (
            m["original_scene"],
            m["trajectory"]
        )

        groups.setdefault(key, []).append(
            portable_path(path)
        )

    keys = sorted(groups)
    total = len(paths)

    if len(keys) < 2:
        raise ValueError("Need at least two trajectories")

    print("Scenes per trajectory:")

    for key in keys:
        print(key, len(groups[key]))

    # Find a complete trajectory combination
    # closest to the target validation ratio.
    target = total * 0.20

    candidates = []

    for n in range(1, len(keys)):

        for subset in combinations(keys, n):

            count = sum(
                len(groups[k]) for k in subset
            )

            candidates.append((
                abs(count - target),
                subset
            ))

    _, val_keys = min(candidates)

    val_keys = set(val_keys)
    train_keys = set(keys) - val_keys

    split = {
        "train": [
            p
            for k in sorted(train_keys)
            for p in groups[k]
        ],
        "val": [
            p
            for k in sorted(val_keys)
            for p in groups[k]
        ],
        "test": [],
        "metadata": {
            "dataset": "TartanAir V1",
            "sampling_version": "v2",
            "policy": "trajectory-level split",
            "target_validation_ratio": 0.20,
            "train_trajectories": sorted(train_keys),
            "val_trajectories": sorted(val_keys),
            "limitation": (
                "All trajectories belong to the same "
                "environment and may share physical areas."
            )
        }
    }

    output.parent.mkdir(
        parents=True, exist_ok=True
    )

    temp = output.with_suffix(".json.tmp")

    try:
        temp.write_text(
            json.dumps(split, indent=2) + "\n"
        )

        load_split(temp)

        temp.replace(output)

    finally:
        temp.unlink(missing_ok=True)

    print("\nTrain trajectories:", sorted(train_keys))
    print("Val trajectories:", sorted(val_keys))

    print("\nTrain scenes:", len(split["train"]))
    print("Val scenes:", len(split["val"]))
    print("Total scenes:", total)

    print("\nSaved:", portable_path(output))


if __name__ == "__main__":
    main()
