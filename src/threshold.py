import cv2

def apply_threshold(gray_image):
    
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    binary_image = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        10
    )
    return binary_image