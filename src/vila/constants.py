DEFAULT_SPECIAL_TOKEN_BOXES = {
    "[UNK]": [0, 0, 0, 0],
    "[PAD]": [0, 0, 0, 0],
    "[CLS]": [0, 0, 0, 0],
    "[MASK]": [0, 0, 0, 0],
    "[SEP]": [1000, 1000, 1000, 1000],
}

MAX_LINE_PER_PAGE = 200
MAX_TOKENS_PER_LINE = 25
MAX_BLOCK_PER_PAGE = 40
MAX_TOKENS_PER_BLOCK = 100
MAX_2D_POSITION_EMBEDDINGS = 1024

MODEL_PDF_WIDTH = 1000
MODEL_PDF_HEIGHT = 1000

UNICODE_CATEGORIES_TO_REPLACE = ["Cc", "Cf", "Co", "Cs", "Mn", "Zl", "Zp", "Zs", "So"]
# Based on the rules in the bert Normalizer in the HF tokenizers 
# https://github.com/huggingface/tokenizers/blob/adf90dcd722cbc20af930441988b317a19815878/tokenizers/src/normalizers/bert.rs#L77