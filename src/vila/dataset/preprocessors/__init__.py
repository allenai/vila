from .config import VILAPreprocessorConfig
from .base import SimplePDFDataPreprocessor
from .grouping import RowGroupingPDFDataPreprocessor, BlockGroupingPDFDataPreprocessor
from .hierarchical_modeling import (
    RowLevelHierarchicalPDFDataPreprocessor,
    BlockLevelHierarchicalPDFDataPreprocessor,
)
from .layout_indicator import (
    SentenceLayoutIndicatorPDFDataPreprocessor,
    RowLayoutIndicatorPDFDataPreprocessor,
    BlockLayoutIndicatorPDFDataPreprocessor,
)


def instantiate_dataset_preprocessor(style, tokenizer, config):

    if style == "hierarchical_modeling":
        if config.agg_level == "row":
            return RowLevelHierarchicalPDFDataPreprocessor(tokenizer, config)
        elif config.agg_level == "block":
            return BlockLevelHierarchicalPDFDataPreprocessor(tokenizer, config)
        else:
            raise ValueError(
                f"Invalid (style, agg_level pair): {(style, config.agg_level)}"
            )

    elif style == "layout_indicator":
        if config.agg_level == "row":
            return RowLayoutIndicatorPDFDataPreprocessor(tokenizer, config)
        elif config.agg_level == "block":
            return BlockLayoutIndicatorPDFDataPreprocessor(tokenizer, config)
        elif config.agg_level == "sentence":
            return SentenceLayoutIndicatorPDFDataPreprocessor(tokenizer, config)

    elif style == "grouping":
        if config.agg_level == "row":
            return RowGroupingPDFDataPreprocessor(tokenizer, config)
        elif config.agg_level == "block":
            return BlockGroupingPDFDataPreprocessor(tokenizer, config)

    elif style == "base":
        return SimplePDFDataPreprocessor(tokenizer, config)