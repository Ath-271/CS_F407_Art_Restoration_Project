"""Download pretrained weights for LaMa (inpainting) and Real-ESRGAN (super-resolution).

Tries GitHub releases first, then Hugging Face mirrors (some networks block
GitHub's release-asset host).
"""
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = {
    "big-lama.pt": [
        "https://huggingface.co/fashn-ai/LaMa/resolve/main/big-lama.pt",
        "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
    ],
    "RealESRGAN_x4plus.pth": [
        "https://huggingface.co/lllyasviel/Annotators/resolve/main/RealESRGAN_x4plus.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    ],
}


def fetch(url: str, dst: Path) -> bool:
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        if shutil.which("curl"):
            subprocess.run(["curl", "-L", "--fail", "--retry", "3", "-o", str(tmp), url], check=True)
        else:
            urllib.request.urlretrieve(url, tmp)
        if tmp.stat().st_size < 1e6:
            raise IOError("file too small")
        tmp.replace(dst)
        return True
    except Exception as e:
        print(f"  failed from {url}: {e}")
        tmp.unlink(missing_ok=True)
        return False


if __name__ == "__main__":
    out = ROOT / "weights"
    out.mkdir(exist_ok=True)
    for name, urls in WEIGHTS.items():
        dst = out / name
        if dst.exists() and dst.stat().st_size > 1e6:
            print(f"{name}: already present")
            continue
        print(f"Downloading {name} ...")
        if not any(fetch(u, dst) for u in urls):
            sys.exit(f"Could not download {name}. Download manually into {out}.")
    print("Done.")
