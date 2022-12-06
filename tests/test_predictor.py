import layoutparser as lp

from vila.pdftools.pdf_extractor import PDFExtractor
from vila.predictors import HierarchicalPDFPredictor, LayoutIndicatorPDFPredictor


def test_return_type():

    pdf_extractor = PDFExtractor("pdfplumber")
    page_tokens, page_images = pdf_extractor.load_tokens_and_image(
        f"tests/fixtures/regular.pdf"
    )

    vision_model = lp.EfficientDetLayoutModel("lp://PubLayNet")
    pdf_predictors = [
        HierarchicalPDFPredictor.from_pretrained(
            "allenai/hvila-row-layoutlm-finetuned-docbank"
        ),
        LayoutIndicatorPDFPredictor.from_pretrained(
            "allenai/ivila-block-layoutlm-finetuned-docbank"
        ),
    ]

    for pdf_predictor in pdf_predictors:
        for idx, page_token in enumerate(page_tokens):
            blocks = vision_model.detect(page_images[idx])
            page_token.annotate(blocks=blocks)
            pdf_data = page_token.to_pagedata().to_dict()
            predicted_tokens = pdf_predictor.predict(
                pdf_data, page_token.page_size, return_type="layout"
            )
            predicted_types = pdf_predictor.predict(
                pdf_data, page_token.page_size, return_type="list"
            )
            assert predicted_types == [l.type for l in predicted_tokens]


def test_vila_paper():
    pdf_extractor = PDFExtractor("pdfplumber")
    page_tokens, page_images = pdf_extractor.load_tokens_and_image(
        f"tests/fixtures/vila-test.pdf"
    )
    pdf_predictor = HierarchicalPDFPredictor.from_pretrained(
        "allenai/hvila-row-layoutlm-finetuned-docbank"
    )

    for idx, page_token in enumerate(page_tokens):
        pdf_data = page_token.to_pagedata().to_dict()
        predicted_tokens = pdf_predictor.predict(pdf_data, page_token.page_size)
