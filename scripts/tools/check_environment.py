"""检查项目核心依赖和本地模型来源，不下载权重、不启动训练。"""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import close_range.common


def main():
    print(f"Python: {sys.executable}")
    failed = []
    for name in ("torch", "torchvision", "cv2", "jaxtyping", "numpy", "scipy",
                 "matplotlib", "gdown", "huggingface_hub", "timm",
                 "mapanything.models", "close_range.training.dataset",
                 "close_range.training.trainer"):
        try:
            importlib.import_module(name)
            print(f"OK: {name}")
        except Exception as error:
            failed.append(name)
            print(f"FAILED: {name}: {error}")
    for name, folder in (("mapanything", "map-anything"), ("uniception", "uniception")):
        module = sys.modules.get(name)
        if module is None or not Path(module.__file__).resolve().is_relative_to(ROOT / "third_party" / folder):
            failed.append(name)
            print(f"FAILED: {name} must use project-local source")
    if failed:
        raise SystemExit("Environment check failed: " + ", ".join(failed))
    print("Environment check passed; no model weights loaded.")


if __name__ == "__main__":
    main()
