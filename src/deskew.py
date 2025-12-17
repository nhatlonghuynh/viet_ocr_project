import cv2
import numpy as np
import math
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
logger = logging.getLogger('Deskew')

def get_skew_angle(image):
    """
    Tính góc nghiêng dựa trên Hough Transform (Tìm dòng kẻ của văn bản).
    """
    # 1. Đảm bảo ảnh đầu vào là ảnh nền đen, nét trắng (để tìm biên)
    # Ảnh từ main.py truyền vào là thick_img (chữ đen nền trắng) hoặc ngược lại.
    # Ta cần Canny để bắt viền chữ trước.
    
    # Nếu ảnh màu thì chuyển xám
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 2. Phát hiện cạnh (Edge Detection)
    # Canny giúp biến các dòng chữ thành các đường viền mảnh
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 3. Dùng HoughLinesP để tìm các đoạn thẳng
    # - rho = 1: Độ chính xác khoảng cách (1 pixel)
    # - theta = np.pi/180: Độ chính xác góc (1 độ)
    # - threshold = 200: Số điểm tối thiểu để coi là 1 đường thẳng (tăng nếu nhiễu nhiều)
    # - minLineLength: Độ dài tối thiểu của dòng (rất quan trọng để bỏ qua rác)
    # - maxLineGap: Khoảng cách tối đa giữa các đoạn nét đứt để nối lại thành 1 dòng
    
    lines = cv2.HoughLinesP(
        edges, 
        1, 
        np.pi / 180, 
        threshold=50,  # Giảm từ 100 → 50 để dễ phát hiện hơn
        minLineLength=max(image.shape[1] // 5, 30),  # Tối thiểu 1/5 chiều rộng hoặc 30px
        maxLineGap=10  # Giảm từ 20 → 10 để chính xác hơn
    )

    if lines is None:
        logger.info("Không tìm thấy đường thẳng để căn chỉnh.")
        return 0.0

    # 4. Tính góc cho từng đường thẳng tìm được
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Tính góc theo độ (degrees)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        
        # LỌC GÓC (Quan trọng):
        # Chỉ lấy những dòng "gần như nằm ngang" (từ -45 đến +45 độ).
        # Điều này giúp loại bỏ các đường kẻ dọc (lề giấy) gây lỗi xoay 90 độ.
        if -45 < angle < 45:
            angles.append(angle)

    if len(angles) == 0:
        return 0.0

    # 5. Lấy trung vị (Median) để loại bỏ nhiễu (outliers)
    # Median tốt hơn Mean (trung bình cộng) vì nó không bị ảnh hưởng bởi 1-2 dòng bị sai lệch hẳn.
    median_angle = np.median(angles)
    logger.debug(f"Phát hiện {len(angles)} dòng nằm ngang, median angle: {median_angle:.2f}°")
    
    return median_angle

def rotate_image(image, angle):
    """
    Xoay ảnh giữ nguyên kích thước (có điền viền trắng).
    """
    if angle == 0: return image
    
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # borderValue=(255, 255, 255): Điền màu trắng vào góc khuyết
    rotated = cv2.warpAffine(
        image, M, (w, h), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(255, 255, 255)
    )
    return rotated

def deskew(image):
    """
    Hàm chính gọi từ bên ngoài.
    """
    try:
        angle = get_skew_angle(image)
        logger.info(f"Góc nghiêng phát hiện: {angle:.2f}°")
        
        # --- PHANH AN TOÀN ---
        # 1. Nếu góc quá nhỏ (< 0.5 độ): Không xoay để giữ nét.
        if abs(angle) < 0.5:
            logger.debug("Góc nhỏ, bỏ qua.")
            return image
            
        # 2. Nếu góc quá lớn (> 20 độ): Log cảnh báo nhưng vẫn xoay
        # Bạn có thể nới lỏng số này nếu muốn hỗ trợ ảnh nghiêng nặng.
        if abs(angle) > 20:
            logger.warning(f"Góc nghiêng lớn ({angle:.2f}°). Vẫn thực hiện xoay.")
            
        return rotate_image(image, angle)
        
    except Exception as e:
        logger.error(f"Lỗi khi deskew: {str(e)}")
        return image

