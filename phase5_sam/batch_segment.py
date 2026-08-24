"""
Phase 5, Exercise 2: Batch-segment an entire folder of images automatically,
with interactive fallback for any image where auto-segmentation looks wrong.

For each image, SAM is first prompted with a single center point (works for
most shots in an orbit-style capture where the object sits roughly centered).
If the result looks implausible -- low confidence, OR the mask covers too
much of the frame (a sign SAM grabbed the wall/background instead of your
small object) -- an interactive window pops up immediately so you can click
the correct object by hand. That corrected mask replaces the bad one before
moving to the next image.

USAGE:
    python phase5_sam/batch_segment.py <input_images_folder> <output_masks_folder>

Example:
    python phase5_sam/batch_segment.py data/raw_captures/object_01 phase5_sam/masks

CONTROLS (when the interactive fallback window opens):
    Left-click  : foreground point (part of the object)
    Right-click : background point (NOT part of the object)
    ENTER       : run segmentation with the points you've placed
    r           : reset points and start over
    s           : skip this image (keep the flagged auto-mask as-is)

OUTPUT:
    For every image_NNN.jpg in the input folder, saves a mask named
    image_NNN.jpg.png in the output folder (COLMAP's expected naming
    convention: <original_filename>.png). White = keep, black = ignore.
"""

import numpy as np
import cv2
import torch
import sys
import os
import glob
from segment_anything import sam_model_registry, SamPredictor

CHECKPOINT_PATH = "phase5_sam/checkpoints/sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"
LOW_CONFIDENCE_THRESHOLD = 0.85
MAX_PLAUSIBLE_OBJECT_FRACTION = 0.0  # dinosaur typically covers ~3-7% of frame
MAX_DISPLAY_DIM = 1000

# Global state for the interactive fallback window's mouse callback
click_points = []
click_labels = []
click_display_img = None
click_scale = 1.0


def mouse_callback(event, x, y, flags, param):
    global click_points, click_labels, click_display_img
    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append([x / click_scale, y / click_scale])
        click_labels.append(1)
        cv2.circle(click_display_img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow(WINDOW_NAME, click_display_img)
    elif event == cv2.EVENT_RBUTTONDOWN:
        click_points.append([x / click_scale, y / click_scale])
        click_labels.append(0)
        cv2.circle(click_display_img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow(WINDOW_NAME, click_display_img)


WINDOW_NAME = "FLAGGED - click correct object (ENTER=run, r=reset, s=skip)"


def interactive_reclick(image_bgr, predictor):
    """
    Opens an interactive window for the user to click the correct object.
    Returns (mask, confidence) for the new SAM prediction, or (None, None)
    if the user pressed 's' to skip and keep the original flagged mask.
    """
    global click_points, click_labels, click_display_img, click_scale

    h, w = image_bgr.shape[:2]
    scale = min(1.0, MAX_DISPLAY_DIM / max(h, w))
    display_size = (int(w * scale), int(h * scale))

    click_points = []
    click_labels = []
    click_scale = scale
    click_display_img = cv2.resize(image_bgr, display_size)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, display_size[0], display_size[1])
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    cv2.imshow(WINDOW_NAME, click_display_img)
    cv2.waitKey(1)

    result = "run"
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # ENTER
            if len(click_points) == 0:
                print("  No points clicked yet -- click the object first.")
                continue
            break
        elif key == ord('r'):
            click_points = []
            click_labels = []
            click_display_img = cv2.resize(image_bgr, display_size)
            cv2.imshow(WINDOW_NAME, click_display_img)
        elif key == ord('s'):
            result = "skip"
            break

    cv2.destroyWindow(WINDOW_NAME)

    if result == "skip":
        return None, None

    input_points = np.array(click_points)
    input_labels = np.array(click_labels)
    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,
    )
    best_idx = np.argmax(scores)
    return masks[best_idx], scores[best_idx]


def main():
    if len(sys.argv) != 3:
        print("Usage: python batch_segment.py <input_images_folder> <output_masks_folder>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint not found at {CHECKPOINT_PATH}")
        sys.exit(1)

    image_paths = sorted(
        glob.glob(os.path.join(input_dir, "*.jpg")) +
        glob.glob(os.path.join(input_dir, "*.jpeg")) +
        glob.glob(os.path.join(input_dir, "*.png"))
    )
    if len(image_paths) == 0:
        print(f"No images found in {input_dir}")
        sys.exit(1)
    print(f"Found {len(image_paths)} images to segment.\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading SAM ({MODEL_TYPE})...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print("Model loaded.\n")

    n_success = 0
    n_auto_flagged = 0
    n_manually_fixed = 0

    for i, img_path in enumerate(image_paths):
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"[{i+1}/{len(image_paths)}] Could not read {img_path}, skipping.")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_bgr.shape[:2]

        predictor.set_image(image_rgb)

        # Center-point prompt: assumes the object is roughly centered,
        # which holds for an orbit-style capture like yours.
        center_point = np.array([[w // 2, h // 2]])
        center_label = np.array([1])

        masks, scores, _ = predictor.predict(
            point_coords=center_point,
            point_labels=center_label,
            multimask_output=True,
        )
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        confidence = scores[best_idx]

        # --- Sanity check: does this look like a plausible OBJECT mask? ---
        # High confidence does NOT mean "correct object" -- it just means
        # SAM is sure about SOME boundary. A mask covering way more of the
        # frame than your object realistically does is a strong sign SAM
        # locked onto the wall/background instead.
        mask_area_fraction = best_mask.sum() / (h * w)
        original_filename = os.path.basename(img_path)

        needs_review = (confidence < LOW_CONFIDENCE_THRESHOLD or
                         mask_area_fraction > MAX_PLAUSIBLE_OBJECT_FRACTION)

        if needs_review:
            n_auto_flagged += 1
            reason = ("low confidence" if confidence < LOW_CONFIDENCE_THRESHOLD
                       else f"{mask_area_fraction*100:.0f}% of frame, likely background")
            print(f"[{i+1}/{len(image_paths)}] {original_filename}: "
                  f"FLAGGED ({reason}) -- opening manual click window...")

            new_mask, new_confidence = interactive_reclick(image_bgr, predictor)
            if new_mask is not None:
                best_mask = new_mask
                confidence = new_confidence
                mask_area_fraction = best_mask.sum() / (h * w)
                n_manually_fixed += 1
                print(f"  -> Manually corrected. New confidence={confidence:.4f}, "
                      f"area={mask_area_fraction*100:.0f}%")
            else:
                print("  -> Skipped, keeping original auto-generated mask.")

        # Save as COLMAP-format mask: <original_filename>.png, white=keep
        mask_uint8 = (best_mask * 255).astype(np.uint8)
        mask_out_path = os.path.join(output_dir, original_filename + ".png")
        cv2.imwrite(mask_out_path, mask_uint8)

        if not needs_review:
            print(f"[{i+1}/{len(image_paths)}] {original_filename}: "
                  f"confidence={confidence:.4f}, area={mask_area_fraction*100:.0f}%")
        n_success += 1

    print(f"\nDone. Segmented {n_success}/{len(image_paths)} images.")
    print(f"Masks saved to {output_dir}/")
    print(f"Auto-flagged: {n_auto_flagged}  |  Manually fixed: {n_manually_fixed}  |  "
          f"Skipped (kept auto mask): {n_auto_flagged - n_manually_fixed}")


if __name__ == "__main__":
    main()
