import cv2 
import numpy as np


def remove_noise(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    denoised = cv2.medianBlur(gray,5)
    
    return denoised   


def sharpen_image(image):
    """
    Làm sắc nét ảnh sử dụng kernel 3x3.
    """
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1], 
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened