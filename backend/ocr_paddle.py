from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en'
)

def run_paddle_ocr(np_image):
    results = ocr.ocr(np_image)
    lines = []

    for block in results:
        for entry in block:
            # PaddleOCR returns a list: [box_points, (text, confidence)]
            if len(entry) == 2:
                _, text_info = entry
                text, score = text_info
                lines.append(text)

            # Some versions return: [box_points, text, confidence]
            elif len(entry) == 3:
                _, text, score = entry
                lines.append(text)

            else:
                # fallback for unexpected shapes
                try:
                    text = entry[1][0]
                    lines.append(text)
                except:
                    pass

    return "\n".join(lines)
