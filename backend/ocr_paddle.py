from paddleocr import PaddleOCR

# High accuracy English OCR with angle correction
ocr = PaddleOCR(use_angle_cls=True, lang="en")

def run_paddle_ocr(np_image):
    """
    np_image: OpenCV/Numpy image (BGR)
    Returns a single string with newline-delimited text blocks.
    """
    results = ocr.ocr(np_image, cls=True)
    lines = []
    for block in results:
        for box, text, score in block:
            lines.append(text)
    return "\n".join(lines)
