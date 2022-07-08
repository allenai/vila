from typing import List, Union, Dict, Any, Tuple
import logging
import unicodedata
from copy import deepcopy

import layoutparser as lp

logger = logging.getLogger(__name__)


def union_box(blocks) -> List:
    if len(blocks) == 0:
        logger.warning("The length of blocks is 0!")
        return [0, 0, 0, 0]

    x1, y1, x2, y2 = float("inf"), float("inf"), float("-inf"), float("-inf")
    for bbox in blocks:
        x1 = min(x1, bbox[0])
        y1 = min(y1, bbox[1])
        x2 = max(x2, bbox[2])
        y2 = max(y2, bbox[3])
    return [int(x1), int(y1), int(x2), int(y2)]


def union_lp_box(blocks: List[lp.TextBlock]) -> List:

    x1, y1, x2, y2 = float("inf"), float("inf"), float("-inf"), float("-inf")

    for bbox in blocks:
        _x1, _y1, _x2, _y2 = bbox.coordinates
        x1 = min(x1, _x1)
        y1 = min(y1, _y1)
        x2 = max(x2, _x2)
        y2 = max(y2, _y2)

    return lp.TextBlock(lp.Rectangle(x1, y1, x2, y2))


def is_in(block_a, block_b):
    """A rewrite of the lp.LayoutElement.is_in function.
    We will use a soft_margin and center function by default.
    """
    return block_a.is_in(
        block_b,
        soft_margin={"top": 1, "bottom": 1, "left": 1, "right": 1},
        center=True,
    )


def assign_tokens_to_blocks(
    blocks: List, tokens: List, keep_empty_blocks=False
) -> Tuple[List, List]:
    """It will assign the token to the corresponding blocks,
    sort the blocks based on the token ids (the blocks are
    ordered based on the minimum id of the contained tokens),
    and then assign the correspinding block_id to the tokens
    within the block.

    It will return a Tuple for blocks (in correct order) and
    tokens (with block_id assigned and in the original order).
    """
    blocks = deepcopy(blocks)
    tokens = deepcopy(tokens)
    left_tokens_last = tokens

    for idx, token in enumerate(tokens):
        token.id = idx

    all_token_groups = []

    for block in blocks:
        token_group = []
        remaining_tokens = []
        for token in left_tokens_last:
            if is_in(token, block):
                token_group.append(token)
            else:
                remaining_tokens.append(token)
        if len(token_group) > 0 or keep_empty_blocks:
            all_token_groups.append((block, token_group))

        left_tokens_last = remaining_tokens

    remaining_tokens = [token for token in left_tokens_last]
    sorted_token_groups = sorted(
        all_token_groups,
        key=lambda ele: min(tok.id for tok in ele[1])
        if len(ele[1]) > 0  # Avoid empty blocks
        else float("-inf"),
    )

    blocks, tokens = [], []
    for bid, (block, gp_tokens) in enumerate(sorted_token_groups):
        block.id = bid
        blocks.append(block)
        for token in gp_tokens:
            token.block_id = bid
        tokens.extend(gp_tokens)

    for token in remaining_tokens:
        token.block_id = None
    tokens.extend(remaining_tokens)

    return blocks, sorted(tokens, key=lambda ele: ele.id)


def replace_unicode_tokens(
    tokens: List[str], unicode_categories: List[str], replace_token: str
) -> List[str]:
    """Replace certain unicode tokens that are in the specified categories
    with the replace_token.

    Args:
        tokens (List[str]): a list of PDF tokens
        unicode_categories (List[str]): the unicode categories to be replaced
        replace_token (str): the token to replace the unicode tokens
    """
    tokens = tokens.copy()
    for idx in range(len(tokens)):
        if all(unicodedata.category(ch) in unicode_categories for ch in tokens[idx]):
            logging.debug(f"Replacing special unicode tokens {tokens[idx]} with {replace_token}")
            tokens[idx] = replace_token
    
    return tokens