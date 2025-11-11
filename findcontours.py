import cv2
import numpy as np

IMG_PATH = "r1.png"  
MIN_AREA = 20        # minimum contour area to keep (tune for small text)
MAX_AREA = 10000     # max contour area to keep (exclude big shapes)
DEBUG = True         # set False to disable visualization


# helper
def preprocess(img):
    """Convert to grayscale, enhance contrast, and binarize adaptively."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive histogram equalization (handles shadows / low light)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)

    # Adaptive threshold for uneven illumination
    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        31, 15
    )

    # Morphological open/close to smooth text regions
    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    return clean


def find_text_contours(binary_img):
    """Find and filter text contours by area and shape."""
    contours, _ = cv2.findContours(
        binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # aspect ratio filter (optional)
        aspect_ratio = w / float(h)
        if 0.2 < aspect_ratio < 10:  # text-like region
            boxes.append((x, y, w, h))

    # Sort contours top-to-bottom, then left-to-right
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes


def visualize_boxes(img, boxes):
    """Draw bounding boxes on the image."""
    vis = img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
    return vis


# main
if __name__ == "__main__":
    # Load image
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"Image not found: {IMG_PATH}")

    # Preprocess
    clean = preprocess(img)

    # Find contours
    boxes = find_text_contours(clean)
    print(f"Detected {len(boxes)} potential text regions")

    # Visualize
    if DEBUG:
        vis = visualize_boxes(img, boxes)
        cv2.imshow("Original", img)
        cv2.imshow("Binary (Thresholded)", clean)
        cv2.imshow("Detected Text Contours", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Optionally save output
    cv2.imwrite("receipt_contours.jpg", visualize_boxes(img, boxes))


mask = cv2.inRange(hsv, (5,150,150), (20,255,255))
(x,y),r = cv2.minEnclosingCircle(contour)
cv2.circle(frame, (int(x),int(y)), int(r), (0,255,0), 2)

