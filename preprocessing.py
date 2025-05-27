import cv2
import numpy as np
from imutils import contours

def preprocess_image(image_path, output_size=(128, 128)):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not cnts:
            return cv2.resize(image, output_size)
        
        cnts = contours.sort_contours(cnts)[0]
        largest_contour = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        card = image[y:y+h, x:x+w]
        return cv2.resize(card, output_size)
    except:
        return None