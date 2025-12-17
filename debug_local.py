import gradio as gr
import cv2
import pytesseract
import numpy as np
import re

from src.deskew import deskew
from src.morphology import apply_morphology
from src.resize import smart_resize
from src.noise_removal import remove_noise
from src.scan import scan_document
from src.threshold import apply_threshold

# from src.accuracy import calculate_accuracy
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def clean_text(text):

    if not text:
        return ""

    # Xóa các ký tự đặc biệt đứng một mình thường do nhiễu sinh ra
    # Ví dụ: "|", "_3", ".."
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        # Xóa ký tự lạ ở đầu và cuối dòng
        line = line.strip(" |._-")
        # Nếu dòng quá ngắn (dưới 2 ký tự) và không phải số -> Khả năng cao là rác -> Bỏ
        if len(line) < 2 and not line.isdigit():
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def tesseract_ocr_pipeline(image):
    if image is None:
        return None, None, None, "Vui lòng tải ảnh lên."
    try:
        # --- BƯỚC 1: SCAN & CẮT GIẤY ---
        scanned_img = scan_document(image)

        # --- BƯỚC 2: TIỀN XỬ LÝ ---
        # Resize chiều cao chữ về 30px
        resized_img = smart_resize(scanned_img, target_height=30)
        # Khử nhiễu
        processed = remove_noise(resized_img)
        # Nhị phân hóa
        binary = apply_threshold(processed)
        # Làm đậm chữ
        thick_img = apply_morphology(binary, operation_type="thick")

        # --- BƯỚC 3: CHỈNH NGHIÊNG (DESKEW) ---
        deskewed_img = deskew(thick_img)

        # --- BƯỚC 4: THÊM VIỀN TRẮNG (PADDING) - MỚI ---
        # Tesseract đọc tốt hơn khi chữ không dính mép
        final_img = cv2.copyMakeBorder(
            deskewed_img,
            top=30,
            bottom=30,
            left=30,
            right=30,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

        # --- BƯỚC 5: OCR ---
        # --psm 6: Giả định một khối văn bản thống nhất (tốt cho trang sách)
        # --psm 3: Tự động phân vùng (tốt nếu trang có nhiều cột/ảnh)
        custom_config = r"--oem 3 --psm 6"
        raw_text = pytesseract.image_to_string(
            final_img, lang="vie", config=custom_config
        )

        # --- BƯỚC 6: HẬU XỬ LÝ (CLEANING) - MỚI ---
        final_text = clean_text(raw_text)

        if not final_text.strip():
            final_text = "Không nhận diện được văn bản."

        # accuracy_info = ""
        # if ground_truth_text.strip():
        #     acc, dist = calculate_cer(ground_truth_text, final_text)
        #     accuracy_info = f"\n\n--- ĐÁNH GIÁ ĐỘ CHÍNH XÁC ---\n" \
        #         f" Độ chính xác: {acc:.2f}%\n" \
        #             f" Số ký tự sai lệch: {dist}"
        #     final_text += accuracy_info
        return scanned_img, thick_img, final_img, final_text

    except Exception as e:
        return None, None, None, str(e)


# --- GIAO DIỆN GRADIO ---
with gr.Blocks(title="VietOCR Advanced Pipeline") as demo:
    gr.Markdown("# 🇻🇳 VietOCR: Hệ thống OCR Tiếng Việt Nâng cao")

    with gr.Row():
        with gr.Column(scale=1):
            inp_image = gr.Image(type="numpy", label="Ảnh đầu vào")
            btn_submit = gr.Button("🚀 Chạy Xử Lý & OCR", variant="primary")

        with gr.Column(scale=1):
            out_text = gr.Textbox(
                label="Kết quả văn bản (Đã làm sạch)", lines=15, buttons=True
            )

    with gr.Accordion("🔍 Debug: Xem các bước trung gian", open=True):
        with gr.Row():
            out_step1 = gr.Image(label="1. Cắt giấy (Scanner)", type="numpy")
            out_step2 = gr.Image(label="2. Nhị phân & Làm đậm", type="numpy")
            out_step3 = gr.Image(label="3. Ảnh cuối (Deskew + Padding)", type="numpy")

    btn_submit.click(
        fn=tesseract_ocr_pipeline,
        inputs=inp_image,
        outputs=[out_step1, out_step2, out_step3, out_text],
    )

if __name__ == "__main__":
    demo.launch()
