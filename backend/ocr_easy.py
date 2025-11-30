import easyocr
import cv2

reader = easyocr.Reader(['en'], gpu=False)

def run_easy_ocr(image_np):
    """Run EasyOCR on an OpenCV BGR image."""
    # EasyOCR expects RGB
    rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    results = reader.readtext(rgb_image)
    return results
