from vila.predictors import normalize_bbox, unnormalize_bbox
from vila.utils import replace_unicode_tokens
from vila.constants import UNICODE_CATEGORIES_TO_REPLACE


def test_normalize_bbox():

    # fmt: off
    assert normalize_bbox([128, 256, 256, 512], 1000, 1000) == (128, 256, 256, 512)
    
    assert normalize_bbox([128, 256, 256, 512], 1000, 1024) == (125.0, 250.0, 250.0, 500.0)
    assert normalize_bbox([128, 256, 256, 512], 1024, 1000) == (125.0, 250.0, 250.0, 500.0)
    
    assert normalize_bbox([128, 256, 256, 512], 1024, 1024) == (125.0, 250.0, 250.0, 500.0)
    assert normalize_bbox([256, 256, 128, 512], 1024, 1024) == (125.0, 250.0, 250.0, 500.0)

    assert unnormalize_bbox((128, 256, 256, 512), 1000, 1000) == (128, 256, 256, 512)
    
    assert unnormalize_bbox((125.0, 250.0, 250.0, 500.0), 1000, 1024) == (128, 256, 256, 512)
    assert unnormalize_bbox((125.0, 250.0, 250.0, 500.0), 1024, 1000) == (128, 256, 256, 512)
    
    assert unnormalize_bbox((125.0, 250.0, 250.0, 500.0), 1024, 1024) == (128, 256, 256, 512)
    # fmt: on


def test_replace_unicode_tokens():

    words = ["\uf02a", "\uf02a\u00ad", "Modalities\uf02a"]

    out = replace_unicode_tokens(
        words,
        UNICODE_CATEGORIES_TO_REPLACE,
        "[UNK]",
    )

    assert out == ["[UNK]", "[UNK]", "Modalities\uf02a"]
