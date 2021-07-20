from abc import ABC, abstractmethod


class BasePDFTokenExtractor(ABC):
    """PDF token extractors will load all the *tokens* and save using pdfstructure service."""

    def __call__(self, pdf_path: str):
        return self.extract(pdf_path)

    @abstractmethod
    def extract(self, pdf_path: str):
        """Extract PDF Tokens from the input pdf_path
        Args:
            pdf_path (str):
                The path to a PDF file
        Returns:
        """
        pass