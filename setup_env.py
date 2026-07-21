"""
Environment setup script for the AI/ML CV/3D Reconstruction learning path.

Usage:
    python setup_env.py

This will:
1. Create a virtual environment (venv)
2. Install core packages needed for Phase 1-5 (OpenCV, Open3D, PyTorch w/ CUDA, etc.)
3. Run a verification check to confirm your GPU/CUDA setup is working

Run this, then activate your venv before doing anything else:
    Windows: venv\\Scripts\\activate
    Linux/Mac: source venv/bin/activate
"""

import subprocess
import sys
import os
import venv

VENV_DIR = "venv"

# Core packages for Phase 0-5 (everything except 3DGS training itself, Phase 6,
# which you'll set up separately once you get there since it needs a custom
# CUDA extension build from the official repo).
PACKAGES = [
    "numpy",
    "opencv-python",
    "opencv-contrib-python",   # includes extra modules (SIFT used to live here on old versions)
    "matplotlib",
    "open3d",
    "jupyter",
    "pillow",
    "tqdm",
    "scipy",
]

# PyTorch needs the CUDA-specific index for GPU support.
# 1050 Ti supports CUDA well; using cu121 wheel (works with most modern NVIDIA drivers).
TORCH_INSTALL_CMD = [
    "torch", "torchvision", "torchaudio",
    "--index-url", "https://download.pytorch.org/whl/cu121",
]


def run(cmd, description):
    print(f"\n{'=' * 60}\n{description}\n{'=' * 60}")
    result = subprocess.run(cmd, shell=isinstance(cmd, str))
    if result.returncode != 0:
        print(f"WARNING: command failed: {cmd}")
    return result.returncode == 0


def main():
    # 1. Create venv if it doesn't exist
    if not os.path.exists(VENV_DIR):
        print("Creating virtual environment...")
        venv.create(VENV_DIR, with_pip=True)
    else:
        print("Virtual environment already exists, skipping creation.")

    # 2. Determine pip path inside venv
    if sys.platform == "win32":
        pip = os.path.join(VENV_DIR, "Scripts", "pip.exe")
        python = os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        pip = os.path.join(VENV_DIR, "bin", "pip")
        python = os.path.join(VENV_DIR, "bin", "python")

    # 3. Upgrade pip
    run([pip, "install", "--upgrade", "pip"], "Upgrading pip")

    # 4. Install PyTorch with CUDA support first (order matters, avoids conflicts)
    run([pip, "install"] + TORCH_INSTALL_CMD, "Installing PyTorch with CUDA 12.1 support")

    # 5. Install remaining packages
    run([pip, "install"] + PACKAGES, "Installing OpenCV, Open3D, and other core packages")

    # 6. Install segment-anything from GitHub (Phase 5, but fine to have ready)
    run(
        [pip, "install", "git+https://github.com/facebookresearch/segment-anything.git"],
        "Installing Segment Anything (SAM)",
    )

    # 7. Run verification
    print(f"\n{'=' * 60}\nVerifying installation\n{'=' * 60}")
    verify_script = os.path.join(os.path.dirname(__file__), "verify_env.py")
    if os.path.exists(verify_script):
        subprocess.run([python, verify_script])
    else:
        print("verify_env.py not found next to this script — skipping auto-verify.")
        print(f"Run manually: {python} verify_env.py")

    print("\nSetup complete. Activate your venv before working:")
    if sys.platform == "win32":
        print(r"  venv\Scripts\activate")
    else:
        print("  source venv/bin/activate")


if __name__ == "__main__":
    main()
