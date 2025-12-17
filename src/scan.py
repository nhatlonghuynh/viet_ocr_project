import cv2
import numpy as np
from src.deskew import deskew

def order_points(pts):
    """Sắp xếp 4 điểm tọa độ: Trên-Trái, Trên-Phải, Dưới-Phải, Dưới-Trái"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """Kéo phẳng tờ giấy dựa trên 4 điểm góc"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def scan_document(image):
    """
    Scanner 3 cấp độ + Cắt gọt rác (Header/Footer).
    """
    # --- PHẦN 1: CHUẨN BỊ (GIỮ NGUYÊN) ---
    orig = image.copy()
    ratio = image.shape[0] / 500.0
    h = 500
    w = int(image.shape[1] / ratio)
    small_img = cv2.resize(image, (w, h))

    gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    # Biến lưu ảnh kết quả tạm thời
    temp_result = None

    # --- PHẦN 2: CHỌN CHIẾN THUẬT (LOGIC MỚI) ---
    
    # === CẤP ĐỘ 1: Tìm thấy 4 góc chuẩn ===
    if screenCnt is not None:
        print("[Scanner] Cấp độ 1: Tìm thấy 4 góc chuẩn. Đang kéo phẳng...")
        temp_result = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    # === CẤP ĐỘ 2: Nếu Cấp 1 thất bại -> Tìm khối lớn nhất ===
    if temp_result is None:
        print("[Scanner] Cấp độ 1 thất bại. Thử Cấp độ 2 (Cắt khối lớn nhất)...")
        if len(cnts) > 0:
            c = cnts[0]
            x, y, w, h = cv2.boundingRect(c)
            
            # Kiểm tra xem khối này có đủ lớn không (tránh nhiễu)
            image_area = small_img.shape[0] * small_img.shape[1]
            if w * h > image_area * 0.1: # Lớn hơn 10% diện tích ảnh
                real_x = int(x * ratio)
                real_y = int(y * ratio)
                real_w = int(w * ratio)
                real_h = int(h * ratio)
                temp_result = orig[real_y:real_y+real_h, real_x:real_x+real_w]

    # === CẤP ĐỘ 3: Nếu cả 2 cấp đều tạch -> Dùng ảnh gốc ===
    if temp_result is None:
        print("[Scanner] Cấp độ 3: Không tìm thấy giấy. Dùng ảnh gốc.")
        temp_result = image

    # --- PHẦN 2.5: CHỈNH NGHIÊNG (DESKEW) TRƯỚC CẮT ---
    # Quan trọng! Phải deskew TRƯỚC khi cắt, nếu không sẽ mất nội dung ở góc
    print("[Scanner] Chỉnh nghiêng ảnh tài liệu...")
    temp_result = deskew(temp_result)

    # --- PHẦN 3: HẬU XỬ LÝ (CẮT GỌT RÁC) ---
    # Lúc này temp_result chắc chắn đã có ảnh (từ cấp 1, 2 hoặc 3) VÀ ĐÃ ĐƯỢC DESKEW
    
    h_res, w_res = temp_result.shape[:2]
    
    # Cắt bỏ ~5% đỉnh (header rác/bóng đổ) và 5% đáy (footer/số trang)
    # Giữ lại ~90% ảnh chứa nội dung chính
    crop_top = int(h_res * 0.05)
    crop_bottom = int(h_res * 0.95)
    
    # Đảm bảo không cắt lố (crop_top < crop_bottom)
    if crop_top < crop_bottom:
        final_result = temp_result[crop_top:crop_bottom, 0:w_res]
    else:
        final_result = temp_result # Nếu ảnh quá bé thì khỏi cắt
    
    print(f"[Scanner] Ảnh cuối cùng: {final_result.shape}")
    return final_result
