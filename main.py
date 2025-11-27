import os
import cv2
import numpy as np


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def get_skin_mask_ycrcb(frame_roi):
    """
    Simple skin detection in YCrCb color space.
    Works decently for many skin tones in controlled lighting.
    """
    ycrcb = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # Typical skin range in YCrCb (tune if needed)
    # Cr: 133–173, Cb: 77–127
    skin_mask = cv2.inRange(
        ycrcb,
        np.array([0, 133, 77], dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8),
    )

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)

    return skin_mask


def count_fingers(contour, drawing_img):
    """
    Use convex hull + convexity defects to estimate number of extended fingers.
    Returns the estimated finger count.
    """
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return 0

    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return 0

    finger_gaps = 0

    for i in range(defects.shape[0]):
        s, e, f, depth = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        # Draw the points and connecting lines for visualization
        cv2.circle(drawing_img, far, 4, (0, 0, 255), -1)
        cv2.line(drawing_img, start, end, (0, 255, 0), 2)

        # Compute angle at the defect (far point)
        a = np.linalg.norm(np.array(end) - np.array(start))
        b = np.linalg.norm(np.array(far) - np.array(start))
        c = np.linalg.norm(np.array(end) - np.array(far))

        if b == 0 or c == 0:
            continue

        # Cosine rule to get angle at far point
        cos_angle = (b**2 + c**2 - a**2) / (2 * b * c)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))

        # depth is in fixed-point 8.24, so scale it down a bit for thresholding
        depth_real = depth / 256.0

        # Heuristic:
        # - angle < 80 degrees (sharp valley between fingers)
        # - depth_real > some threshold (separation significant enough)
        if angle < 80 and depth_real > 20:
            finger_gaps += 1

    # Typically, number of fingers ≈ number of gaps + 1 (if hand open)
    if finger_gaps == 0:
        return 0
    return min(finger_gaps + 1, 5)  # clamp to 5 for sanity


def main():
    print("=== OpenCV Hand Tracking + Finger Count (Webcam) ===")
    print("Place your hand inside the ROI box.")
    print("Controls: 'q' → quit, 's' → save snapshot")

    ensure_dir("outputs")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    snapshot_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame from webcam.")
            break

        # Flip horizontally for mirror-like view
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Define ROI in the frame (e.g., right side box)
        # You can adjust these values if needed
        roi_x_start = int(w * 0.55)
        roi_y_start = int(h * 0.15)
        roi_x_end = int(w * 0.95)
        roi_y_end = int(h * 0.75)

        roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

        # Get skin mask in ROI
        skin_mask = get_skin_mask_ycrcb(roi)

        # Find contours on skin mask
        contours, _ = cv2.findContours(
            skin_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        annotated = frame.copy()

        # Draw ROI rectangle
        cv2.rectangle(
            annotated,
            (roi_x_start, roi_y_start),
            (roi_x_end, roi_y_end),
            (0, 255, 255),
            2
        )

        finger_count = 0

        if contours:
            # Largest contour is likely the hand
            max_cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_cnt)

            # Ignore tiny blobs
            if area > 2000:
                # Offset contour points to full-frame coordinates
                max_cnt_offset = max_cnt + np.array([[roi_x_start, roi_y_start]])

                # Draw contour
                cv2.drawContours(
                    annotated,
                    [max_cnt_offset],
                    -1,
                    (255, 255, 0),
                    2
                )

                # Convex hull (for visualization)
                hull_points = cv2.convexHull(max_cnt_offset)
                cv2.drawContours(
                    annotated,
                    [hull_points],
                    -1,
                    (0, 255, 0),
                    2
                )

                # Count fingers (use original contour but draw on annotated)
                finger_count = count_fingers(max_cnt_offset, annotated)

        # Put finger count text
        text = f"Fingers: {finger_count}"
        cv2.putText(
            annotated,
            text,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Hand Tracking + Finger Count", annotated)
        cv2.imshow("Skin Mask (ROI)", skin_mask)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            out_path = os.path.join(
                "outputs", f"hand_finger_count_snapshot_{snapshot_idx}.jpg"
            )
            cv2.imwrite(out_path, annotated)
            print(f"📸 Saved snapshot to {out_path}")
            snapshot_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Webcam released, windows closed.")


if __name__ == "__main__":
    main()
