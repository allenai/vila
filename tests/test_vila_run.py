import pytest
import layoutparser as lp

from vila.pdftools.pdf_extractor import PDFExtractor
from vila.predictors import HierarchicalPDFPredictor, LayoutIndicatorPDFPredictor


def test_hvila_run():

    pdf_extractor = PDFExtractor("pdfplumber")
    page_tokens, page_images = pdf_extractor.load_tokens_and_image(
        f"tests/fixtures/regular.pdf"
    )

    vision_model = lp.EfficientDetLayoutModel("lp://PubLayNet")
    pdf_predictor = HierarchicalPDFPredictor.from_pretrained(
        "allenai/hvila-row-layoutlm-finetuned-docbank"
    )

    for idx, page_token in enumerate(page_tokens):

        # Method 1
        predicted_tokens1 = pdf_predictor.predict_page(
            page_token, page_image=page_images[idx], visual_group_detector=vision_model
        )

        # Method 2
        blocks = vision_model.detect(page_images[idx])
        page_token.annotate(blocks=blocks)
        pdf_data = page_token.to_pagedata().to_dict()
        predicted_tokens2 = pdf_predictor.predict(pdf_data, page_token.page_size)

        assert predicted_tokens1 == predicted_tokens2


def test_ivila_run():
    pdf_extractor = PDFExtractor("pdfplumber")
    page_tokens, page_images = pdf_extractor.load_tokens_and_image(
        f"tests/fixtures/regular.pdf"
    )

    vision_model = lp.EfficientDetLayoutModel("lp://PubLayNet")
    pdf_predictor = LayoutIndicatorPDFPredictor.from_pretrained(
        "allenai/ivila-block-layoutlm-finetuned-docbank"
    )

    for idx, page_token in enumerate(page_tokens):

        # Method 1
        predicted_tokens1 = pdf_predictor.predict_page(
            page_token, page_image=page_images[idx], visual_group_detector=vision_model
        )
        assert page_token.blocks == []

        # Method 2
        blocks = vision_model.detect(page_images[idx])
        page_token.annotate(blocks=blocks)
        pdf_data = page_token.to_pagedata().to_dict()
        predicted_tokens2 = pdf_predictor.predict(pdf_data, page_token.page_size)

        assert predicted_tokens1 == predicted_tokens2


def test_vila_run_with_special_unicode_inputs():

    pdf_data = {
        "words": ["\uf02a", "\uf02a\u00ad", "Modalities"],
        "bbox": [
            [82.806, 70.34515579999993, 123.4487846, 84.6913558],
            [127.0353346, 70.34515579999993, 191.9949282, 84.6913558],
            [195.5814782, 70.34515579999993, 262.26261580000005, 84.6913558],
        ],
        "block_ids": [0, 0, 0],
    }

    pdf_predictor = LayoutIndicatorPDFPredictor.from_pretrained(
        "allenai/ivila-block-layoutlm-finetuned-docbank"
    )

    pdf_predictor.predict(pdf_data, (596, 842))

    with pytest.raises(AssertionError):
        pdf_predictor.predict(pdf_data, (596, 842), replace_empty_unicode=False)


def test_vila_run_bbox():

    pdf_data = {
        "words": ["\uf02a", "New", "Modalities"],
        "block_ids": [0, 0, 0],
    }

    pdf_predictor = LayoutIndicatorPDFPredictor.from_pretrained(
        "allenai/ivila-block-layoutlm-finetuned-docbank"
    )

    # Case 1: Good boxes 
    bbox =  [
        [82, 70, 123, 84],
        [127, 70, 191, 84],
        [195, 70, 262, 84],
    ]
    pdf_data["bbox"] = bbox
    pdf_predictor.predict(pdf_data, (596, 800))

    # Case 2: Good boxes -- float
    bbox =  [
        [82.806, 70.34515579999993, 123.4487846, 84.6913558],
        [127.0353346, 70.34515579999993, 191.9949282, 84.6913558],
        [195.5814782, 70.34515579999993, 262.26261580000005, 84.6913558],
    ]
    pdf_data["bbox"] = bbox
    pdf_predictor.predict(pdf_data, (596, 800))

    # Case 3: Large Pages - height 

    bbox =  [
        [82.806, 70.34515579999993, 123.4487846, 84.6913558],
        [127.0353346, 70.34515579999993, 191.9949282, 84.6913558],
        [195.5814782, 70.34515579999993, 262.26261580000005, 84.6913558],
    ]
    pdf_data["bbox"] = bbox
    pdf_predictor.predict(pdf_data, (596, 1200))

    # Case 4: Large Pages - width 

    bbox =  [
        [82.806, 70.34515579999993, 123.4487846, 84.6913558],
        [127.0353346, 70.34515579999993, 191.9949282, 84.6913558],
        [195.5814782, 70.34515579999993, 262.26261580000005, 84.6913558],
    ]
    pdf_data["bbox"] = bbox
    pdf_predictor.predict(pdf_data, (1200, 596))

    # Case 5: Large Pages - Both 

    bbox =  [
        [82.806, 70.34515579999993, 123.4487846, 84.6913558],
        [127.0353346, 70.34515579999993, 191.9949282, 84.6913558],
        [195.5814782, 70.34515579999993, 262.26261580000005, 84.6913558],
    ]
    pdf_data["bbox"] = bbox
    pdf_predictor.predict(pdf_data, (1200, 1200))

    # C Case 6: Incorrect Bbox (x1>x2)

    bbox =  [
        [82.806, 70.34515579999993, 123.4487846, 84.6913558],
        [191.9949282, 70.34515579999993, 127.0353346, 84.6913558],
        [296, 70.34515579999993, 262.26261580000005, 84.6913558],
    ]
    pdf_data["bbox"] = bbox
    pdf_predictor.predict(pdf_data, (1200, 1200))