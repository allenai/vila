import layoutparser as lp
from vila.pdftools.pdf_extractor import PDFExtractor
from vila.predictors import LayoutIndicatorPDFPredictor

def test_large_pdf():
    pdf_extractor = PDFExtractor("pdfplumber")
    page_tokens, page_images = pdf_extractor.load_tokens_and_image(f"tests/fixtures/large.pdf")

    vision_model = lp.EfficientDetLayoutModel("lp://PubLayNet") 
    pdf_predictor = LayoutIndicatorPDFPredictor.from_pretrained("allenai/ivila-block-layoutlm-finetuned-docbank")

    for idx, page_token in enumerate(page_tokens[:1]):
        blocks = vision_model.detect(page_images[idx])
        page_token.annotate(blocks=blocks)
        pdf_data = page_token.to_pagedata().to_dict()
        predicted_tokens = pdf_predictor.predict(pdf_data, page_token.page_size)