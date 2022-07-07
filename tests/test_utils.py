from vila.predictors import normalize_bbox


def test_normalize_bbox():

    # fmt: off
    assert normalize_bbox([128, 256, 256, 512], 1000, 1000) == (128, 256, 256, 512)
    assert normalize_bbox([128, 256, 256, 512], 1000, 1024) == (128, 250.0, 256, 500.0)
    assert normalize_bbox([128, 256, 256, 512], 1024, 1000) == (125, 256, 250, 512)
    assert normalize_bbox([128, 256, 256, 512], 1024, 1024) == (125.0, 250.0, 250.0, 500.0)
    assert normalize_bbox([256, 256, 128, 512], 1024, 1024) == (125.0, 250.0, 250.0, 500.0)
    # fmt: on
