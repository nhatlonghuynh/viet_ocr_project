import cv2
import numpy as np
import pytesseract
import gradio as gr
import jiwer

# Cấu hình đường dẫn Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def get_skew_angle(image):
    # Tính toán góc nghiêng của văn bản sử dụng Hough Transform
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    lines = cv2.HoughLinesP(
        dilate, 1, np.pi / 180, 200, minLineLength=100, maxLineGap=20
    )
    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -45 < angle < 45:
            angles.append(angle)

    return np.median(angles) if angles else 0.0


def rotate_image(image, angle):
    # Xoay ảnh theo góc lệch để cân chỉnh thẳng hàng
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def smart_noise_filter(binary_img, min_area=5):
    # Lọc bỏ các vùng nhiễu có diện tích nhỏ hơn ngưỡng min_area (Smart Filter)
    img_inv = cv2.bitwise_not(binary_img)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img_inv, connectivity=8
    )
    output_mask = np.zeros_like(img_inv)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            output_mask[labels == i] = 255

    return cv2.bitwise_not(output_mask)


def calculate_accuracy(ground_truth, ocr_text):
    # Tính toán độ chính xác (Accuracy) và tỷ lệ lỗi (CER/WER)
    if not ground_truth.strip():
        return "Chưa nhập văn bản gốc."

    transforms = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ]
    )

    try:
        wer = jiwer.wer(
            ground_truth,
            ocr_text,
            truth_transform=transforms,
            hypothesis_transform=transforms,
        )
        cer = jiwer.cer(
            ground_truth,
            ocr_text,
            truth_transform=transforms,
            hypothesis_transform=transforms,
        )
        acc = (1 - cer) * 100
        return f"Accuracy: {acc:.2f}%\nCER: {cer:.4f}\nWER: {wer:.4f}"
    except:
        return "Lỗi tính toán."


def process_pipeline(
    image, ground_truth, do_deskew, denoise_mode, min_area, block_size, c_val, psm_mode
):
    # Hàm xử lý chính: Deskew -> Denoise -> Threshold -> Smart Filter -> OCR -> Eval
    if image is None:
        return None, "", ""

    # 1. Deskew
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if do_deskew:
        angle = get_skew_angle(img)
        if abs(angle) > 0.5:
            img = rotate_image(img, angle)

    # 2. Grayscale & Basic Denoise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if denoise_mode == "Median Blur (Khuyên dùng)":
        gray = cv2.medianBlur(gray, 3)
    elif denoise_mode == "Gaussian Blur":
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    elif denoise_mode == "FastNLMeans":
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # 3. Adaptive Threshold
    if block_size % 2 == 0:
        block_size += 1
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_val
    )

    # 4. Smart Noise Filter
    final_img = smart_noise_filter(binary, min_area=min_area)

    # 5. OCR & Evaluation
    config = f"--oem 3 --psm {psm_mode}"
    text = pytesseract.image_to_string(final_img, config=config, lang="vie")
    metrics = calculate_accuracy(ground_truth, text)

    return final_img, text, metrics


# --- GIAO DIỆN NGƯỜI DÙNG (GRADIO) ---
with gr.Blocks(title="Vietnamese OCR Project", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🇻🇳 Đồ án OCR Tiếng Việt Nâng Cao")

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Ảnh đầu vào", type="numpy")
            txt_ground_truth = gr.TextArea(
                label="Văn bản gốc (Ground Truth)",
                placeholder="Nhập văn bản chuẩn...",
                lines=3,
            )

            with gr.Accordion("Cấu hình tham số", open=True):
                chk_deskew = gr.Checkbox(label="Deskew (Xoay ảnh)", value=True)
                dd_denoise = gr.Dropdown(
                    [
                        "None",
                        "Median Blur (Khuyên dùng)",
                        "Gaussian Blur",
                        "FastNLMeans",
                    ],
                    value="Median Blur (Khuyên dùng)",
                    label="Denoise Method",
                )
                slider_area = gr.Slider(
                    0, 20, value=5, step=1, label="Smart Filter Area (px)"
                )
                slider_block = gr.Slider(
                    3, 51, value=15, step=2, label="Threshold Block Size"
                )
                slider_c = gr.Slider(0, 20, value=10, label="Threshold C Constant")
                dd_psm = gr.Dropdown(["3", "6"], value="6", label="PSM Mode")

            btn = gr.Button("CHẠY XỬ LÝ", variant="primary")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Ảnh nhị phân"):
                    output_img = gr.Image(label="Processed Image", type="numpy")
                with gr.TabItem("Kết quả Text"):
                    output_text = gr.Textbox(
                        label="OCR Output", lines=15, show_copy_button=True
                    )
                with gr.TabItem("Đánh giá"):
                    output_metrics = gr.Textbox(label="Metrics", lines=5)

    btn.click(
        fn=process_pipeline,
        inputs=[
            input_img,
            txt_ground_truth,
            chk_deskew,
            dd_denoise,
            slider_area,
            slider_block,
            slider_c,
            dd_psm,
        ],
        outputs=[output_img, output_text, output_metrics],
    )

if __name__ == "__main__":
    demo.launch()
