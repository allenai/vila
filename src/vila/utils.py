from typing import List, Union, Dict, Any, Tuple
import logging

import layoutparser as lp

def union_box(blocks) -> List:
    if len(blocks) == 0:
        logging.warning("The length of blocks is 0!")
        return [0, 0, 0, 0]

    x1, y1, x2, y2 = float("inf"), float("inf"), float("-inf"), float("-inf")
    for bbox in blocks:
        x1 = min(x1, bbox[0])
        y1 = min(y1, bbox[1])
        x2 = max(x2, bbox[2])
        y2 = max(y2, bbox[3])
    return [int(x1), int(y1), int(x2), int(y2)]

def union_lp_box(blocks:List[lp.TextBlock]) -> List:

    x1, y1, x2, y2 = float("inf"), float("inf"), float("-inf"), float("-inf")
    
    for bbox in blocks:
        _x1, _y1, _x2, _y2 = bbox.coordinates
        x1 = min(x1, _x1)
        y1 = min(y1, _y1)
        x2 = max(x2, _x2)
        y2 = max(y2, _y2)

    return lp.TextBlock(lp.Rectangle(x1, y1, x2, y2))