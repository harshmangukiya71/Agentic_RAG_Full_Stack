"""
ocr.py - native text extraction plus optional OCR fallback.

PaddleOCR is preferred when installed. Tesseract is used as a fallback. If neither
OCR engine is installed, scanned pages are reported with empty text instead of
crashing the whole ingestion job.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import fitz

from app.models import OCRBlock, OCRPage

logger = logging.getLogger(__name__)


class OCRProcessor:
    def __init__(self, min_native_chars_per_page: int = 40, render_dpi: int = 220) -> None:
        self.min_native_chars_per_page = min_native_chars_per_page
        self.render_dpi = render_dpi
        self._paddle: Any | None = None
        self._paddle_checked = False
        self._tesseract_checked = False
        self._has_tesseract = False

    def extract_document(self, path: Path | str) -> list[OCRPage]:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf(path)
        if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
            return [self._ocr_image_file(path, document_id=path.name, page_number=1)]
        raise ValueError(f"Unsupported file type for OCR ingestion: {suffix}")

    def _extract_pdf(self, path: Path) -> list[OCRPage]:
        pages: list[OCRPage] = []
        with fitz.open(str(path)) as doc:
            for page_number, page in enumerate(doc, start=1):
                native = page.get_text("text").strip()
                blocks = self._native_blocks(page)
                if len(native) >= self.min_native_chars_per_page:
                    pages.append(OCRPage(
                        document_id=path.name,
                        page_number=page_number,
                        raw_text=native,
                        ocr_confidence=1.0,
                        blocks=blocks,
                        extraction_method="native",
                    ))
                    continue

                pix = page.get_pixmap(dpi=self.render_dpi, alpha=False)
                image_bytes = pix.tobytes("png")
                ocr_page = self._ocr_image_bytes(image_bytes, path.name, page_number)
                pages.append(ocr_page)
        return pages

    def _native_blocks(self, page: fitz.Page) -> list[OCRBlock]:
        native_blocks: list[OCRBlock] = []
        for block in page.get_text("blocks"):
            if len(block) < 5:
                continue
            text = str(block[4]).strip()
            if text:
                native_blocks.append(OCRBlock(
                    text=text,
                    bbox=[float(block[0]), float(block[1]), float(block[2]), float(block[3])],
                    confidence=1.0,
                ))
        return native_blocks

    def _ocr_image_file(self, path: Path, document_id: str, page_number: int) -> OCRPage:
        data = path.read_bytes()
        return self._ocr_image_bytes(data, document_id, page_number)

    def _ocr_image_bytes(self, image_bytes: bytes, document_id: str, page_number: int) -> OCRPage:
        paddle_page = self._try_paddle(image_bytes, document_id, page_number)
        if paddle_page is not None:
            return paddle_page

        tesseract_page = self._try_tesseract(image_bytes, document_id, page_number)
        if tesseract_page is not None:
            return tesseract_page

        logger.warning("No OCR backend available for %s page %d", document_id, page_number)
        return OCRPage(
            document_id=document_id,
            page_number=page_number,
            raw_text="",
            ocr_confidence=0.0,
            blocks=[],
            extraction_method="ocr_unavailable",
        )

    def _try_paddle(self, image_bytes: bytes, document_id: str, page_number: int) -> OCRPage | None:
        try:
            if not self._paddle_checked:
                from paddleocr import PaddleOCR  # type: ignore

                self._paddle = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
                self._paddle_checked = True
            if self._paddle is None:
                return None

            import io

            import numpy as np
            from PIL import Image

            image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
            result = self._paddle.ocr(image, cls=True)
            blocks: list[OCRBlock] = []
            texts: list[str] = []
            confidences: list[float] = []
            for page_result in result or []:
                for item in page_result or []:
                    bbox = item[0] or []
                    text, conf = item[1]
                    flat_bbox = [float(v) for point in bbox for v in point]
                    confidence = max(0.0, min(1.0, float(conf)))
                    blocks.append(OCRBlock(text=text, bbox=flat_bbox, confidence=confidence))
                    texts.append(text)
                    confidences.append(confidence)
            if not texts:
                return None
            return OCRPage(
                document_id=document_id,
                page_number=page_number,
                raw_text="\n".join(texts),
                ocr_confidence=sum(confidences) / len(confidences),
                blocks=blocks,
                extraction_method="paddleocr",
            )
        except Exception as exc:
            if not self._paddle_checked:
                self._paddle_checked = True
                logger.info("PaddleOCR is not available: %s", exc)
            else:
                logger.warning("PaddleOCR failed on %s page %d: %s", document_id, page_number, exc)
            self._paddle = None
            return None

    def _try_tesseract(self, image_bytes: bytes, document_id: str, page_number: int) -> OCRPage | None:
        try:
            if not self._tesseract_checked:
                import pytesseract  # type: ignore

                self._has_tesseract = True
                self._tesseract_checked = True
            if not self._has_tesseract:
                return None

            import io

            from PIL import Image
            import pytesseract  # type: ignore

            image = Image.open(io.BytesIO(image_bytes))
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config="--psm 6")
            blocks: list[OCRBlock] = []
            texts: list[str] = []
            confidences: list[float] = []
            for i, text in enumerate(data.get("text", [])):
                text = str(text).strip()
                if not text:
                    continue
                raw_conf = float(data["conf"][i])
                confidence = 0.0 if raw_conf < 0 else min(1.0, raw_conf / 100.0)
                bbox = [
                    float(data["left"][i]),
                    float(data["top"][i]),
                    float(data["left"][i] + data["width"][i]),
                    float(data["top"][i] + data["height"][i]),
                ]
                blocks.append(OCRBlock(text=text, bbox=bbox, confidence=confidence))
                texts.append(text)
                confidences.append(confidence)
            if not texts:
                return None
            return OCRPage(
                document_id=document_id,
                page_number=page_number,
                raw_text=" ".join(texts),
                ocr_confidence=sum(confidences) / len(confidences),
                blocks=blocks,
                extraction_method="tesseract",
            )
        except Exception as exc:
            if not self._tesseract_checked:
                self._tesseract_checked = True
                logger.info("Tesseract OCR is not available: %s", exc)
            else:
                logger.warning("Tesseract failed on %s page %d: %s", document_id, page_number, exc)
            self._has_tesseract = False
            return None