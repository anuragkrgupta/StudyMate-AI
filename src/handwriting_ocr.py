import cv2
import numpy as np
import pymupdf
import streamlit as st
import torch

from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


MODEL_NAME = "microsoft/trocr-base-handwritten"


@st.cache_resource
def load_handwriting_model():
    """Load a matching TrOCR processor and model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(
        MODEL_NAME,
        use_fast=False,
    )
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    # Use half precision on the GTX 1650 to reduce VRAM usage.
    if device == "cuda":
        model = model.half()

    model = model.to(device)
    model.eval()

    return processor, model, device


def detect_text_lines(page_image):
    """Detect approximate handwritten text-line crops."""
    image = np.array(page_image.convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Convert ink to white pixels on a black background.
    binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    # Remove long horizontal notebook ruling lines.
    kernel_width = max(30, image.shape[1] // 25)
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_width, 1)
    )
    horizontal_rules = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, horizontal_kernel
    )
    cleaned = cv2.subtract(binary, horizontal_rules)

    # Find rows containing enough ink to belong to text.
    row_ink = np.count_nonzero(cleaned, axis=1)
    active_rows = row_ink > max(3, int(image.shape[1] * 0.002))

    # Group nearby active rows into text lines.
    indices = np.flatnonzero(active_rows)
    if len(indices) == 0:
        return []

    groups = []
    start = previous = int(indices[0])

    for value in indices[1:]:
        value = int(value)

        if value - previous > 15:
            groups.append((start, previous + 1))
            start = value

        previous = value

    groups.append((start, previous + 1))

    height, width = gray.shape
    crops = []

    for y1, y2 in groups:
        if y2 - y1 < 8:
            continue

        # Include a little vertical context around each line.
        top = max(0, y1 - 8)
        bottom = min(height, y2 + 8)

        # Find horizontal bounds of the ink in this line.
        line_ink = cleaned[top:bottom, :]
        columns = np.flatnonzero(np.any(line_ink > 0, axis=0))

        if len(columns) == 0:
            continue

        left = max(0, int(columns[0]) - 10)
        right = min(width, int(columns[-1]) + 11)

        if right - left >= 20:
            crop = image[top:bottom, left:right]
            crops.append(Image.fromarray(crop).convert("RGB"))

    return crops


def recognize_handwritten_pdf(pdf_bytes, max_pages=50, page_number_offset=0):
    """
    Extract handwritten text from a scanned PDF locally.

    Args:
        pdf_bytes (bytes): Raw bytes of the PDF to recognize.
        max_pages (int, optional): Maximum number of pages allowed for OCR. Defaults to 50.
        page_number_offset (int, optional): Zero-based offset added to page
            numbers in the output (e.g., page_number_offset=3 labels the first page
            in pdf_bytes as Page 4). Defaults to 0.

    Returns:
        str: Extracted and recognized text from all processed pages.
    """
    processor, model, device = load_handwriting_model()
    results = []

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        if len(document) > max_pages:
            raise ValueError(
                f"PDF has {len(document)} pages. "
                f"Please upload a PDF with at most {max_pages} pages."
            )

        for page_number, page in enumerate(document, start=1 + page_number_offset):
            # Render at 200 DPI for handwriting recognition.
            pixmap = page.get_pixmap(dpi=200, alpha=False)
            page_image = Image.frombytes(
                "RGB",
                (pixmap.width, pixmap.height),
                pixmap.samples,
            )

            lines = detect_text_lines(page_image)
            del pixmap, page_image
            print(
                f"Page {page_number}: "
                f"detected {len(lines)} candidate text lines"
            )
            recognized_lines = []

            for line_image in lines:
                inputs = processor(
                    images=line_image,
                    return_tensors="pt",
                ).pixel_values.to(device)

                if device == "cuda":
                    inputs = inputs.half()

                with torch.inference_mode():
                    generated_ids = model.generate(
                        inputs,
                        max_new_tokens=128,
                    )

                text = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )[0].strip()

                if text:
                    print(f"Recognized: {text[:100]}")
                    recognized_lines.append(text)

            del lines

            if recognized_lines:
                results.append(
                    f"[Page {page_number}]\n"
                    + "\n".join(recognized_lines)
                )

    return "\n\n".join(results)
