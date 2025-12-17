import cv2
import numpy as np

def apply_morphology(binary_image, operation_type='thick'):
    kernel = np.ones((2, 2), np.uint8)
    if operation_type == 'thick':
        result = cv2.erode(binary_image, kernel, iterations=1)
        
    elif operation_type == 'clean':
        result = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    elif operation_type == 'thin':
        result = cv2.dilate(binary_image, kernel, iterations=2)
    else:
        result = binary_image
    return result