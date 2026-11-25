from pathlib import Path
import zipfile
import io
import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.home() / "CS541/mapanything-original-data/tartanair-v1"
OUT = Path("data/previews/tartanair")
OUT.mkdir(parents=True, exist_ok=True)

samples = [
    ("P003", "000002"),
    ("P005", "000569"),
    ("P001", "000030"),
    ("P002", "000000"),
]

rng = np.random.default_rng(42)

def backproject(x, y, z):
    return np.array([
        (x - 320) * z / 320,
        (y - 240) * z / 320,
        z
    ])

def load_npy(z, name):
    return np.load(io.BytesIO(z.read(name)), allow_pickle=False)

def sample_pair(depth, seg, valid):
    safe = np.zeros_like(valid)

    safe[1:-1, 1:-1] = (
        valid[1:-1, 1:-1]
        & (np.abs(depth[1:-1, 1:-1] - depth[:-2, 1:-1]) < 0.03)
        & (np.abs(depth[1:-1, 1:-1] - depth[2:, 1:-1]) < 0.03)
        & (np.abs(depth[1:-1, 1:-1] - depth[1:-1, :-2]) < 0.03)
        & (np.abs(depth[1:-1, 1:-1] - depth[1:-1, 2:]) < 0.03)
    )

    ys, xs = np.where(safe)

    if len(xs) < 2:
        return None

    for _ in range(5000):
        i, j = rng.choice(len(xs), 2, replace=False)

        a = (int(xs[i]), int(ys[i]))
        b = (int(xs[j]), int(ys[j]))

        if seg[a[1], a[0]] == seg[b[1], b[0]]:
            continue

        pa = backproject(*a, depth[a[1], a[0]])
        pb = backproject(*b, depth[b[1], b[0]])

        distance = float(np.linalg.norm(pa - pb))

        if 0.05 <= distance <= 1.0:
            return a, b, distance

    return None


with zipfile.ZipFile(ROOT / "japanesealley_Easy_image_left.zip") as rz, \
     zipfile.ZipFile(ROOT / "japanesealley_Easy_depth_left.zip") as dz, \
     zipfile.ZipFile(ROOT / "japanesealley_Easy_seg_left.zip") as sz:

    for trajectory, frame in samples:

        prefix = f"japanesealley/Easy/{trajectory}"

        rgb = np.array(Image.open(io.BytesIO(
            rz.read(f"{prefix}/image_left/{frame}_left.png")
        )).convert("RGB"))

        depth = load_npy(
            dz, f"{prefix}/depth_left/{frame}_left_depth.npy"
        )

        seg = load_npy(
            sz, f"{prefix}/seg_left/{frame}_left_seg.npy"
        )

        assert rgb.shape[:2] == depth.shape == seg.shape
        assert depth.shape == (480, 640)

        valid = (
            np.isfinite(depth)
            & (depth >= 0.3)
            & (depth <= 2.0)
        )

        pair = sample_pair(depth, seg, valid)

        print(f"\n{trajectory}/{frame}")
        print("RGB shape:", rgb.shape)
        print("Depth shape:", depth.shape)
        print("Seg labels:", np.unique(seg))
        print("Near pixels:", f"{valid.mean():.2%}")

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title("RGB")

        im = axes[0, 1].imshow(
            depth, cmap="turbo", vmin=0.3, vmax=5
        )
        axes[0, 1].set_title("Depth (meters)")
        fig.colorbar(im, ax=axes[0, 1])

        axes[0, 2].imshow(seg, cmap="nipy_spectral")
        axes[0, 2].set_title("Segmentation")

        axes[1, 0].imshow(valid, cmap="gray")
        axes[1, 0].set_title("Valid depth: 0.3-2m")

        overlay = rgb.copy()
        overlay[~valid] = (overlay[~valid] * 0.2).astype(np.uint8)

        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title("Close-range RGB")

        ax = axes[1, 2]
        ax.imshow(rgb)

        if pair:
            a, b, distance = pair

            ax.plot(
                [a[0], b[0]], [a[1], b[1]],
                color="yellow", linewidth=2
            )

            for p, c, label in [
                (a, "red", "A"), (b, "cyan", "B")
            ]:
                ax.scatter(*p, c=c, s=60)
                ax.text(p[0]+8, p[1]-8, label,
                        color=c, fontsize=14)

            ax.set_title(f"GT distance: {distance*100:.2f} cm")

            print("Point A:", a)
            print("Point B:", b)
            print("Depth A:", depth[a[1], a[0]])
            print("Depth B:", depth[b[1], b[0]])
            print("Label A:", seg[a[1], a[0]])
            print("Label B:", seg[b[1], b[0]])
            print("Distance (m):", distance)

        else:
            ax.set_title("No valid pair")
            print("No valid pair found")

        for ax in axes.flat:
            ax.axis("off")

        fig.suptitle(f"TartanAir: {trajectory}/{frame}")
        fig.tight_layout()

        output = OUT / f"{trajectory}_{frame}.png"
        fig.savefig(output, dpi=150)
        plt.close(fig)

        print("Saved:", output)

print("\nInspection complete.")
