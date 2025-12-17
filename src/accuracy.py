import Levenshtein

def calculate_cer(ground_truth, ocr_text):
    """
    Tính Character Error Rate (CER) và Độ chính xác (Accuracy).
    - Ground Truth: Văn bản chuẩn gõ bằng tay.
    - OCR Text: Văn bản máy đọc.
    """
    # Chuẩn hóa văn bản (xóa khoảng trắng thừa, xuống dòng) để so sánh công bằng
    s1 = " ".join(ground_truth.strip().split())
    s2 = " ".join(ocr_text.strip().split())
    
    if not s1: return 0.0
    
    # Tính khoảng cách chỉnh sửa (số ký tự cần sửa để s2 thành s1)
    distance = Levenshtein.distance(s1, s2)
    max_len = max(len(s1), len(s2))
    
    # Tính CER (Càng thấp càng tốt)
    cer = distance / max_len
    
    # Tính Accuracy (Càng cao càng tốt)
    accuracy = (1 - cer) * 100
    
    return accuracy, distance
