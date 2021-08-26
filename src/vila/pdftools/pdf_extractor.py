import pdf2image

from .pdfplumber_extractor import PDFPlumberTokenExtractor


class PDFExtractor:
    """PDF Extractor will load both images and layouts for PDF documents for downstream processing."""

    def __init__(self, pdf_extractor_name, **kwargs):

        self.pdf_extractor_name = pdf_extractor_name.lower()

        if self.pdf_extractor_name == PDFPlumberTokenExtractor.NAME:
            self.pdf_extractor = PDFPlumberTokenExtractor(**kwargs)
        else:
            raise NotImplementedError(
                f"Unknown pdf_extractor_name {pdf_extractor_name}"
            )

    def load_tokens_and_image(
        self, pdf_path: str, resize_image=False, resize_layout=False, dpi=72, **kwargs
    ):

        pdf_tokens = self.pdf_extractor(pdf_path, **kwargs)

        page_images = pdf2image.convert_from_path(pdf_path, dpi=dpi)

        assert not (
            resize_image and resize_layout
        ), "You could not resize image and layout simultaneously."

        if resize_layout:
            for image, page in zip(page_images, pdf_tokens):
                width, height = image.size
                resize_factor = width / page.width, height / page.height
                page.tokens = page.tokens.scale(resize_factor)
                page.image_height = height
                page.image_width = width

        elif resize_image:
            page_images = [
                image.resize((int(page.width), int(page.height)))
                if page.width != image.size[0]
                else image
                for image, page in zip(page_images, pdf_tokens)
            ]

        return pdf_tokens, page_images
