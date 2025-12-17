import cv2
import numpy as np

def remove_shadows(image):
    """
    Loại bỏ bóng đổ và cân bằng ánh sáng nền.
    """
    # Chuyển sang ảnh xám nếu là ảnh màu
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 1. Ước lượng nền (Background estimation)
    # Dùng Dilate + MedianBlur để làm mờ hết chữ, chỉ giữ lại độ sáng nền
    dilated_img = cv2.dilate(gray, np.ones((7,7), np.uint8)) 
    bg_img = cv2.medianBlur(dilated_img, 21)

    # 2. Tính hiệu số: 255 - |Ảnh gốc - Nền|
    # Logic: Vùng có chữ sẽ tối hơn nền -> hiệu số lớn. Vùng bóng sẽ gần bằng nền -> hiệu số nhỏ.
    diff_img = 255 - cv2.absdiff(gray, bg_img)

    # 3. Chuẩn hóa (Normalize) để kéo dãn độ tương phản
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    return norm_img