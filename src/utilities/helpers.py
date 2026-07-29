import cv2
import numpy as np


def preprocess_img(img):
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)

    _, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(img)

    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)

    img = img[y:y+h, x:x+w]

    if h > w:
        new_h = 20
        new_w = int(w * 20 / h)
    else:
        new_w = 20
        new_h = int(h * 20 / w)

    img = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = img

    processed = canvas.astype("float32") / 255.0
    processed = np.expand_dims(processed, -1)

    return processed
