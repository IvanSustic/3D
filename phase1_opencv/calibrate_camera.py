"""
Phase 1, Exercise 1: Camera Calibration

Finds your camera's intrinsic parameters (focal length, optical center,
lens distortion coefficients) from photos of a checkerboard pattern.

HOW TO CAPTURE PHOTOS:
1. Display checkerboard_10x7.png full-screen on a monitor/tablet
   (10x7 squares = 9x6 internal corners, the values used below)
2. Take 15-20 photos of the screen with your phone, covering:
   - Straight-on shots
   - Angled shots (tilt left/right, up/down)
   - Different distances (fill more/less of the frame)
   - Checkerboard in different parts of the image (corners, edges, center)
   Variety matters more than quantity here — angled/corner shots teach
   the calibration about lens distortion, straight-on shots alone won't.
3. Save all photos into: data/raw_captures/calibration/
   (jpg or png, any resolution)

USAGE:
    python phase1_opencv/calibrate_camera.py

OUTPUT:
    Prints the camera matrix and distortion coefficients, saves them to
    phase1_opencv/camera_calibration.npz for reuse in later phases
    (COLMAP and Open3D steps will want these numbers).
"""

import cv2
import numpy as np
import glob
import os

# --- Configuration ---
CHECKERBOARD = (9, 6)  # internal corners (squares_x - 1, squares_y - 1)
IMAGES_DIR = "data/raw_captures/calibration"
OUTPUT_FILE = "phase1_opencv/camera_calibration.npz"

# Termination criteria for corner sub-pixel refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    images = glob.glob(os.path.join(IMAGES_DIR, "*.jpg")) + \
             glob.glob(os.path.join(IMAGES_DIR, "*.jpeg")) + \
             glob.glob(os.path.join(IMAGES_DIR, "*.png"))

    if len(images) < 10:
        print(f"Found only {len(images)} images in {IMAGES_DIR}")
        print("You need at least 10-15 for a reliable calibration. Add more and re-run.")
        if len(images) == 0:
            return

    # 3D points in real world space (the checkerboard is flat, so z=0 for all)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    img_shape = None
    used = 0

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"  Could not read {fname}, skipping.")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if found:
            used += 1
            objpoints.append(objp)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_refined)
            print(f"  [OK]   {os.path.basename(fname)}")
        else:
            print(f"  [MISS] {os.path.basename(fname)} — checkerboard not detected")

    print(f"\nUsed {used}/{len(images)} images for calibration.")

    if used < 10:
        print("Fewer than 10 successful detections — calibration will be unreliable.")
        print("Common fixes: better lighting, hold camera steadier, make sure the")
        print("full checkerboard (all 9x6 inner corners) is visible in frame.")
        if used < 5:
            return

    print("\nRunning calibration...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    print(f"\nReprojection error (RMS): {ret:.4f}")
    print("  (lower is better — under 0.5 is excellent, under 1.0 is good, over 1.5 needs more/better photos)")

    print("\nCamera matrix (intrinsics):")
    print(camera_matrix)
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    print(f"\n  Focal length: fx={fx:.1f}, fy={fy:.1f} (pixels)")
    print(f"  Optical center: cx={cx:.1f}, cy={cy:.1f} (pixels)")

    print("\nDistortion coefficients (k1, k2, p1, p2, k3):")
    print(dist_coeffs.ravel())

    np.savez(OUTPUT_FILE, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs,
             image_size=img_shape, reprojection_error=ret)
    print(f"\nSaved calibration to {OUTPUT_FILE}")

    # Quick visual sanity check: undistort one of the images
    sample = cv2.imread(images[0])
    undistorted = cv2.undistort(sample, camera_matrix, dist_coeffs)
    sanity_path = "phase1_opencv/undistort_sanity_check.jpg"
    cv2.imwrite(sanity_path, np.hstack([sample, undistorted]))
    print(f"Saved side-by-side original-vs-undistorted comparison to {sanity_path}")
    print("(straight lines that were curved in the original should look straighter on the right)")


if __name__ == "__main__":
    main()
