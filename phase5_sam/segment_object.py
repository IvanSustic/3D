"""
Phase 5, Exercise 1: Segment an object using Segment Anything (SAM).

Given an image and a point you click on (or specify), SAM predicts a
segmentation mask isolating that object -- exactly the kind of automatic
object-isolation that would replace the "plain background" workaround
you've been using so far, and that's core to the actual job's pipeline
(SAM used to isolate individual objects before/during reconstruction).

USAGE:
    python phase5_sam/segment_object.py <path_to_image>

This opens the image and lets you CLICK on the object you want to segment.
Left-click = foreground point (this is part of the object)
Right-click = background point (this is NOT part of the object, optional)
Press ENTER when done clicking to run segmentation.
Press 'r' to reset points and start over.

OUTPUT:
    Displays the predicted mask overlaid on the image, and saves it to
    phase5_sam/mask_output.png
"""

import numpy as np
import cv2
import torch
import sys
import os
from segment_anything import sam_model_registry, SamPredictor

CHECKPOINT_PATH = "phase5_sam/checkpoints/sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"

# Global state for mouse click handling
points = []          # full-resolution coordinates (what SAM actually uses)
labels = []           # 1 = foreground, 0 = background
display_img = None
display_scale = 1.0   # set in main(); used to convert click coords -> full-res


def mouse_callback(event, x, y, flags, param):
    global points, labels, display_img
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x / display_scale, y / display_scale])
        labels.append(1)
        cv2.circle(display_img, (x, y), 5, (0, 255, 0), -1)  # green = foreground
        cv2.imshow("Click object to segment (ENTER=run, r=reset, ESC=quit)", display_img)
    elif event == cv2.EVENT_RBUTTONDOWN:
        points.append([x / display_scale, y / display_scale])
        labels.append(0)
        cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)  # red = background
        cv2.imshow("Click object to segment (ENTER=run, r=reset, ESC=quit)", display_img)


def main():
    global points, labels, display_img, display_scale

    if len(sys.argv) != 2:
        print("Usage: python segment_object.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint not found at {CHECKPOINT_PATH}")
        print("Download it from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        sys.exit(1)

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Could not read image: {image_path}")
        sys.exit(1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # SAM runs on the full-resolution image (better mask quality), but we
    # display a downscaled copy for clicking -- large photos (e.g. 4080x3060)
    # can exceed screen size and cause the OpenCV window to render blank on
    # some systems. Clicks on the small display get scaled back up to match
    # full-resolution coordinates before being passed to SAM.
    max_display_dim = 1000
    h, w = image_bgr.shape[:2]
    scale = min(1.0, max_display_dim / max(h, w))
    display_size = (int(w * scale), int(h * scale))
    print(f"Original image: {w}x{h}, display scale: {scale:.3f} -> {display_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading SAM ({MODEL_TYPE}) from {CHECKPOINT_PATH} ...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # This is the expensive step -- SAM computes a full image embedding once,
    # then can answer many different point-prompts against it very cheaply.
    print("Computing image embedding (this may take a few seconds)...")
    predictor.set_image(image_rgb)
    print("Ready. Click on the object in the image window.\n")

    # --- Interactive point collection ---
    display_scale = scale
    display_img = cv2.resize(image_bgr, display_size)
    window_name = "Click object to segment (ENTER=run, r=reset, ESC=quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_size[0], display_size[1])
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, display_img)
    cv2.waitKey(1)  # force an initial paint on some Windows/OpenCV builds

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # ENTER
            if len(points) == 0:
                print("No points clicked yet -- click on the object first.")
                continue
            break
        elif key == ord('r'):
            points = []
            labels = []
            display_img = cv2.resize(image_bgr, display_size)
            cv2.imshow(window_name, display_img)
            print("Points reset.")
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()

    # --- Run SAM prediction ---
    input_points = np.array(points)
    input_labels = np.array(labels)

    print(f"Running SAM with {len(points)} point(s)...")
    # multimask_output=True returns 3 candidate masks (SAM is often uncertain
    # about scale/ambiguity -- e.g. "whole car" vs "car door" -- so it gives
    # options ranked by confidence). We'll take the highest-scoring one.
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,
    )

    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]
    print(f"Best mask confidence score: {scores[best_idx]:.4f}")

    # --- Visualize: overlay mask in a translucent color ---
    overlay = image_bgr.copy()
    color_mask = np.zeros_like(image_bgr)
    color_mask[best_mask] = [0, 255, 0]  # green
    overlay = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)

    # Draw the clicked points on top for reference
    for (px, py), lbl in zip(points, labels):
        px, py = int(round(px)), int(round(py))
        color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
        cv2.circle(overlay, (px, py), 6, color, -1)
        cv2.circle(overlay, (px, py), 6, (255, 255, 255), 2)

    out_dir = "phase5_sam"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mask_output.png")
    cv2.imwrite(out_path, overlay)
    print(f"Saved result to {out_path}")

    result_window = "Segmentation result (close window to exit)"
    cv2.namedWindow(result_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(result_window, display_size[0], display_size[1])
    cv2.imshow(result_window, cv2.resize(overlay, display_size))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
