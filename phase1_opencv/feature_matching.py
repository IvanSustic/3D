"""
Phase 1, Exercise 2: Feature Detection & Matching

Takes two photos of the same object/scene from different angles, finds
distinctive keypoints in each (using SIFT), and matches them between
the two images.

This is literally step one of what COLMAP does internally across every
pair of images in your dataset -- here we're doing it by hand, on two
images, so you can see and understand exactly what's happening.

USAGE:
    python phase1_opencv/feature_matching.py <image1> <image2>

Example:
    python phase1_opencv/feature_matching.py data/raw_captures/object_01/img_001.jpg data/raw_captures/object_01/img_002.jpg

OUTPUT:
    Saves a visualization showing keypoints matched between the two images
    to phase1_opencv/matches_output.jpg
"""

import cv2
import numpy as np
import sys
import os


def main():
    if len(sys.argv) != 3:
        print("Usage: python feature_matching.py <image1> <image2>")
        sys.exit(1)

    path1, path2 = sys.argv[1], sys.argv[2]

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    if img1 is None or img2 is None:
        print("Could not read one or both images. Check the paths.")
        sys.exit(1)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # --- Step 1: Detect keypoints + compute descriptors ---
    # A "keypoint" is a distinctive, repeatable location in the image
    # (corners, blobs, textured spots -- things that look the same even
    # if the camera angle/scale/lighting changes a bit).
    # A "descriptor" is a numerical fingerprint of the local area around
    # that keypoint, used to compare/match it against keypoints in other images.
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    print(f"Image 1: {len(kp1)} keypoints detected")
    print(f"Image 2: {len(kp2)} keypoints detected")

    # --- Step 2: Match descriptors between the two images ---
    # For every keypoint in image 1, find its best-matching keypoint in image 2
    # based on descriptor similarity (nearest neighbor in descriptor space).
    # k=2 means: find the *two* best matches for each, so we can apply Lowe's
    # ratio test next (a standard trick to filter out unreliable matches).
    bf = cv2.BFMatcher()
    raw_matches = bf.knnMatch(des1, des2, k=2)

    # --- Step 3: Lowe's ratio test ---
    # If the best match and second-best match are nearly equally good, the
    # match is ambiguous/unreliable -- discard it. Only keep matches where
    # the best match is clearly, distinctly better than the next-best.
    good_matches = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"Raw matches: {len(raw_matches)}")
    print(f"Good matches (after ratio test): {len(good_matches)}")

    if len(good_matches) < 8:
        print("\nToo few good matches to proceed reliably.")
        print("Try images with more overlap, better lighting, or less motion blur.")
        return

    # --- Step 4: RANSAC filtering using the Fundamental Matrix ---
    # Even after the ratio test, some matches will still be wrong (outliers).
    # RANSAC finds the geometric relationship (the Fundamental Matrix) that's
    # consistent with the *majority* of matches, and throws out the ones that
    # don't fit that consistent geometry -- these are the true outliers.
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)

    inlier_matches = [m for m, keep in zip(good_matches, mask.ravel()) if keep]
    print(f"Inlier matches (after RANSAC geometric filtering): {len(inlier_matches)}")

    print("\nFundamental Matrix:")
    print(F)
    print("(this encodes the geometric relationship between the two camera views --")
    print(" this is exactly what COLMAP computes for every image pair, at scale)")

    # --- Step 5: Visualize ---
    match_img = cv2.drawMatches(
        img1, kp1, img2, kp2, inlier_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    out_dir = "phase1_opencv"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "matches_output.jpg")
    cv2.imwrite(out_path, match_img)
    print(f"\nSaved match visualization to {out_path}")
    print("Open it and look at the lines connecting the two images --")
    print("each line is one point the algorithm believes is the same physical")
    print("point in the real world, seen from two different angles.")


if __name__ == "__main__":
    main()
