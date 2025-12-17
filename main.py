import gradio as gr
import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ===== IMPORT MODULE =====
from src.deskew import deskew
from src.morphology import apply_morphology
from src.resize import smart_resize
from src.noise_removal import remove_noise, sharpen_image
from src.scan import scan_document
from src.threshold import apply_threshold
from src.accuracy import calculate_cer
from src.illumination import remove_shadows


def ocr_pipeline(image, ground_truth=""):
    if image is None:
        return [None] * 11

    # ======================
    # OCR TRÊN ẢNH GỐC
    # ======================
    custom_config = r"--oem 3 --psm 6"
    raw_text = pytesseract.image_to_string(image, lang="vie", config=custom_config)
    raw_text = "\n".join([l for l in raw_text.split("\n") if l.strip()])
    if not raw_text.strip():
        raw_text = "(Không nhận diện được nội dung từ ảnh gốc)"

    # ======================
    # PIPELINE TIỀN XỬ LÝ
    # ======================
    scan_img = scan_document(image)
    resize_img = smart_resize(scan_img, target_height=1500)
    shadow_img = remove_shadows(resize_img)
    denoise_img = remove_noise(shadow_img)
    sharp_img = sharpen_image(denoise_img)
    bin_img = apply_threshold(sharp_img)
    morph_img = apply_morphology(bin_img, operation_type="thick")
    deskew_img = deskew(morph_img)

    deskew_img = cv2.copyMakeBorder(
        deskew_img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )

    # ======================
    # OCR SAU XỬ LÝ
    # ======================
    final_text = pytesseract.image_to_string(
        deskew_img, lang="vie", config=custom_config
    )
    final_text = "\n".join([l for l in final_text.split("\n") if l.strip()])
    if not final_text.strip():
        final_text = "(Không nhận diện được nội dung sau xử lý)"

    # ======================
    # ĐÁNH GIÁ ĐỘ CHÍNH XÁC
    # ======================
    accuracy_result = "Chưa có Ground Truth để đánh giá."
    if ground_truth.strip():
        acc_raw, err_raw = calculate_cer(ground_truth, raw_text)
        acc_final, err_final = calculate_cer(ground_truth, final_text)

        improvement = acc_final - acc_raw
        icon = "🚀" if improvement > 0 else "⚠️"

        accuracy_result = (
            f"📊 BÁO CÁO ĐỘ CHÍNH XÁC\n"
            f"{'-'*40}\n"
            f"🔴 Ảnh gốc:  {acc_raw:.2f}% (Sai {err_raw} ký tự)\n"
            f"🟢 Sau xử lý: {acc_final:.2f}% (Sai {err_final} ký tự)\n"
            f"{'-'*40}\n"
            f"{icon} CẢI THIỆN: {improvement:+.2f}%"
        )

    return (
        scan_img,
        resize_img,
        shadow_img,
        denoise_img,
        sharp_img,
        bin_img,
        morph_img,
        deskew_img,
        raw_text,
        final_text,
        accuracy_result,
    )


# ======================
# GIAO DIỆN GRADIO
# ======================
with gr.Blocks(title="Vietnamese OCR Pipeline Demo") as demo:
    gr.Markdown("# 🇻🇳 HỆ THỐNG OCR TIẾNG VIỆT")
    gr.Markdown("### So sánh OCR trước và sau tiền xử lý ảnh")

    with gr.Row():
        with gr.Column(scale=1):
            inp_image = gr.Image(type="numpy", label="1. Ảnh tài liệu đầu vào")
            inp_gt = gr.Textbox(
                label="2. Ground Truth (văn bản chuẩn)",
                placeholder="Nhập nội dung đúng để chấm điểm...",
                lines=8,
            )
            btn_run = gr.Button("🚀 CHẠY OCR", variant="primary")

        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("🔴 OCR ảnh gốc"):
                    out_raw_text = gr.Textbox(
                        label="Kết quả OCR ảnh gốc",
                        lines=10,
                        max_lines=10,
                        buttons=True,
                    )
                with gr.Tab("🟢 OCR sau xử lý"):
                    out_final_text = gr.Textbox(
                        label="Kết quả OCR sau pipeline",
                        lines=10,
                        max_lines=10,
                        buttons=True,
                    )

            out_accuracy = gr.Textbox(label="📈 Đánh giá độ chính xác", lines=6)

    # ======================
    # HIỂN THỊ TỪNG BƯỚC
    # ======================
    with gr.Accordion("🔍 Quy trình xử lý ảnh từng bước", open=True):
        with gr.Row():
            out_scan = gr.Image(label="1. Scan tài liệu")
            out_resize = gr.Image(label="2. Resize")
            out_shadow = gr.Image(label="3. Remove Shadow")

        with gr.Row():
            out_denoise = gr.Image(label="4. Denoise (Median)")
            out_sharp = gr.Image(label="5. Sharpen")
            out_bin = gr.Image(label="6. Adaptive Threshold")

        with gr.Row():
            out_morph = gr.Image(label="7. Morphology (Làm đậm chữ)")
            out_deskew = gr.Image(label="8. Deskew (Final)")

    btn_run.click(
        fn=ocr_pipeline,
        inputs=[inp_image, inp_gt],
        outputs=[
            out_scan,
            out_resize,
            out_shadow,
            out_denoise,
            out_sharp,
            out_bin,
            out_morph,
            out_deskew,
            out_raw_text,
            out_final_text,
            out_accuracy,
        ],
    )


if __name__ == "__main__":
    demo.launch()
