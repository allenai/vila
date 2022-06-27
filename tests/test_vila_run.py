import layoutparser as lp # For visualization 

from vila.pdftools.pdf_extractor import PDFExtractor
from vila.predictors import HierarchicalPDFPredictor, LayoutIndicatorPDFPredictor

def test_hvila_run():

    pdf_extractor = PDFExtractor("pdfplumber")
    page_tokens, page_images = pdf_extractor.load_tokens_and_image(f"tests/fixtures/large.pdf")

    vision_model = lp.EfficientDetLayoutModel("lp://PubLayNet") 
    pdf_predictor = HierarchicalPDFPredictor.from_pretrained("allenai/hvila-row-layoutlm-finetuned-docbank")

    for idx, page_token in enumerate(page_tokens):
        blocks = vision_model.detect(page_images[idx])
        page_token.annotate(blocks=blocks)
        pdf_data = page_token.to_pagedata().to_dict()
        predicted_tokens = pdf_predictor.predict(pdf_data, page_token.page_size)

def test_ivila_run():
    pdf_extractor = PDFExtractor("pdfplumber")
    page_tokens, page_images = pdf_extractor.load_tokens_and_image(f"tests/fixtures/large.pdf")

    vision_model = lp.EfficientDetLayoutModel("lp://PubLayNet") 
    pdf_predictor = LayoutIndicatorPDFPredictor.from_pretrained("allenai/ivila-block-layoutlm-finetuned-docbank")

    for idx, page_token in enumerate(page_tokens):
        blocks = vision_model.detect(page_images[idx])
        page_token.annotate(blocks=blocks)
        pdf_data = page_token.to_pagedata().to_dict()
        predicted_tokens = pdf_predictor.predict(pdf_data, page_token.page_size)
