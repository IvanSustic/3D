"""
Verification script — checks that key packages installed correctly and
that PyTorch can see your GPU (1050 Ti / CUDA).

Run after setup_env.py, from inside the activated venv:
    python verify_env.py
"""


def check(name, fn):
    try:
        result = fn()
        print(f"[OK]   {name}: {result}")
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        return False


def main():
    print("=" * 60)
    print("Environment verification")
    print("=" * 60)

    ok = True

    ok &= check("NumPy", lambda: __import__("numpy").__version__)

    ok &= check("OpenCV", lambda: __import__("cv2").__version__)

    def sift_check():
        cv2 = __import__("cv2")
        sift = cv2.SIFT_create()
        return "SIFT_create() available"
    ok &= check("OpenCV SIFT", sift_check)

    ok &= check("Open3D", lambda: __import__("open3d").__version__)

    def torch_check():
        torch = __import__("torch")
        return f"v{torch.__version__}"
    ok &= check("PyTorch", torch_check)

    def cuda_check():
        torch = __import__("torch")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA not available. Check NVIDIA drivers are installed "
                "(nvidia-smi should work in your terminal)."
            )
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return f"{name}, {vram_gb:.1f} GB VRAM"
    ok &= check("CUDA / GPU", cuda_check)

    ok &= check("Segment Anything", lambda: __import__("segment_anything").__name__)

    print("=" * 60)
    if ok:
        print("All checks passed. You're ready for Phase 1.")
    else:
        print("Some checks failed — see [FAIL] lines above.")
        print("Most common fix: update your NVIDIA driver from nvidia.com/drivers")
    print("=" * 60)


if __name__ == "__main__":
    main()
