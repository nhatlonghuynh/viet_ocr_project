import cv2
import numpy as np

def smart_resize(image, target_height=30):
    """
    Phóng to ảnh thông minh dựa trên chiều cao chữ trung bình.
    
    Args:
        image: Ảnh đầu vào (BGR hoặc grayscale)
        target_height: Chiều cao chữ mục tiêu (pixel)
    
    Returns:
        Ảnh sau khi phóng to (nếu cần) hoặc ảnh gốc
    """
    # Chuyển sang xám
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Nhị phân hóa để tìm vị trí chữ
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image
    
    # Tìm chiều cao của các ký tự (loại bỏ nhiễu quá nhỏ/lớn)
    char_heights = []
    for cnt in contours:
        _, _, _, h = cv2.boundingRect(cnt)
        if 5 < h < image.shape[0] / 2:
            char_heights.append(h)
            
    if not char_heights:
        return image
    
    # Lấy trung vị chiều cao chữ (tránh outliers)
    median_height = np.median(char_heights)
    
    # Chỉ phóng to nếu cần thiết
    if median_height >= target_height:
        return image  # Đã đủ lớn
    
    scale_factor = target_height / median_height
    scale_factor = min(scale_factor, 3.0)  # Giới hạn phóng to tối đa 3x
    
    # Nếu scale factor nhỏ (< 1.2) thì không cần phóng to
    if scale_factor <= 1.2:
        return image
    
    # Phóng to ảnh
    new_height = int(image.shape[0] * scale_factor)
    new_width = int(image.shape[1] * scale_factor)
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return resized       


def merge_comparison(img_original, img_resized,max_display_width=1200):
    h1,w1 = img_original.shape[:2]
    h2,w2 = img_resized.shape[:2]
    
    max_h = max(h1,h2)
    total_w = w1 + w2 
    
    canvas =  np.zeros((max_h, total_w,3), dtype= np.uint8)
    
    if (len(img_original.shape) ==2):
        img_original = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
    if (len(img_resized.shape) ==2):
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        
    canvas[0:h1,0:w1] = img_original
    canvas[0:h2,w1:w1+w2] = img_resized
    
    cv2.putText(canvas,f"Gốc ({w1},{h1})",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)    
    cv2.putText(canvas,f"Resize ({w2},{h2})",(w1+10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    
    h_final, w_final = canvas.shape[:2]
    if w_final > max_display_width:
        scale = max_display_width / w_final
        new_h = int(h_final * scale)
        
        canvas = cv2.resize (canvas, (max_display_width, new_h), interpolation=cv2.INTER_AREA)  
    return canvas


# img_original = cv2.imread("data/raw/image1.jpg")
# img_resized = smart_resize(img_original, target_height=30)
# result_image = merge_comparison(img_original, img_resized)

# cv2.imshow("So sánh Trước và Sau khi Resize",result_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

