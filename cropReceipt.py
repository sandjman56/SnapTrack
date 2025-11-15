import cv2
import numpy as np

# Helper function: gamma correction
def gammaCorrection(image, gamma):
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# Receipt Detection using LineSegmentDetector
def detect_receipt_lines(img_path):
    img0 = cv2.imread(img_path)
    h0, w0 = img0.shape[:2]
    scale = 1000 / max(h0, w0)
    img = cv2.resize(img0, (int(w0*scale), int(h0*scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(gammaCorrection(img.copy(), 6), cv2.COLOR_BGR2GRAY)

    # Edge detection
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    # cv2.imshow("Edges", edges)

    line_img = img.copy()
    
    # refine with Line Segment Detector
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    dlines = lsd.detect(edges)[0]

    # Create mask from detected lines 
    mask = np.zeros_like(gray)
    if dlines is not None:
        for dline in dlines:
            x0, y0, x1, y1 = map(int, dline[0])
            cv2.line(mask, (x0,y0), (x1,y1), 255, 3)
    mask = cv2.dilate(mask, np.ones((7,7), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11,11), np.uint8))
    # cv2.imshow("Mask", mask)

    # Find bounding box around detected region 
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    crop = img[y:y+h, x:x+w]

    return crop

if __name__ == "__main__":
    files = ["receipt.png", "receipt2.png", "receipt3.png"]
    for f in files:
        print(f"Processing: {f}\n")
        crop = detect_receipt_lines(f)
        cv2.imshow("Final result",crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()