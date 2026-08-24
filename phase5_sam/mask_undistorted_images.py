"""
Phase 5/2 bridge: Apply SAM masks to COLMAP's undistorted images.

COLMAP's PatchMatch Stereo (the dense matching step) doesn't support mask
files directly. The practical workaround: blacken out the unwanted regions
in the undistorted images themselves, so PatchMatch has nothing meaningful
to match there.

IMPORTANT CAVEAT: COLMAP's undistortion step slightly WARPS images to remove
lens distortion. Your SAM masks were made on the ORIGINAL (distorted)
photos, so they won't align perfectly pixel-for-pixel with the undistorted
versions. For a small, roughly-centered object and typical phone lens
distortion, this misalignment should be minor -- but it's not exact.
If your masked results look off (dinosaur edges clipped, stray background
surviving right at mask boundaries), this is why.

USAGE:
    python phase5_sam/mask_undistorted_images.py <undistorted_images_folder> <masks_folder>

Example:
    python phase5_sam/mask_undistorted_images.py phase2_colmap/object_01/dense_masked/images phase5_sam/masks

This OVERWRITES images in <undistorted_images_folder> in place -- make a
backup copy of that folder first if you want to preserve the unmasked
undistorted originals.

OUTPUT:
    Prints a summary of how many images were masked vs skipped (no
    matching mask found).
"""

import cv2
import numpy as np
import sys
import os
import glob


def main():
    if len(sys.argv) != 3:
        print("Usage: python mask_undistorted_images.py <undistorted_images_folder> <masks_folder>")
        sys.exit(1)

    images_dir = sys.argv[1]
    masks_dir = sys.argv[2]

    if not os.path.isdir(images_dir):
        print(f"Not a folder: {images_dir}")
        sys.exit(1)
    if not os.path.isdir(masks_dir):
        print(f"Not a folder: {masks_dir}")
        sys.exit(1)

    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.jpeg")) +
        glob.glob(os.path.join(images_dir, "*.png"))
    )
    if len(image_paths) == 0:
        print(f"No images found in {images_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} undistorted images.")
    print("This will OVERWRITE them in place with masked (blacked-out) versions.")
    confirm = input("Continue? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        sys.exit(0)

    n_masked = 0
    n_skipped = 0

    for img_path in image_paths:
        original_filename = os.path.basename(img_path)
        # SAM masks were saved as <original_filename>.png (COLMAP convention)
        mask_path = os.path.join(masks_dir, original_filename + ".png")

        if not os.path.exists(mask_path):
            print(f"[SKIP] No mask found for {original_filename} "
                  f"(expected {mask_path})")
            n_skipped += 1
            continue

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"[SKIP] Could not read {original_filename} or its mask.")
            n_skipped += 1
            continue

        # Undistortion can change image dimensions slightly -- resize the
        # mask to match if needed (nearest-neighbor keeps mask edges crisp
        # rather than introducing blurry gray values).
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

        # Blacken everything outside the mask (mask=0 means "not object")
        binary_mask = (mask > 127).astype(np.uint8)
        masked_img = img * binary_mask[:, :, np.newaxis]

        cv2.imwrite(img_path, masked_img)
        n_masked += 1
        print(f"[OK] Masked {original_filename}")

    print(f"\nDone. Masked {n_masked} images, skipped {n_skipped} "
          f"(no matching mask found).")
    if n_skipped > 0:
        print("Skipped images were left as unmasked undistorted originals --")
        print("their background will still contribute to the dense point cloud.")


if __name__ == "__main__":
    main()
