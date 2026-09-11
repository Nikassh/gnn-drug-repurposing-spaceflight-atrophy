import pandas as pd
import numpy as np

class DirectionAwareDEGParser:
    """
    Direction-Aware Differential Gene Expression Parser for NASA GeneLab Datasets.
    Corrects Log2FoldChange signs based on contrast string orientation.
    """
    def __init__(self, q_value_threshold: float = 0.05):
        self.q_value_threshold = q_value_threshold

    def parse_contrast(self, df: pd.DataFrame, contrast_col: str, logfc_col: str, qval_col: str) -> pd.DataFrame:
        """
        Parses contrast column and normalizes Log2FC orientation relative to Ground Control.
        """
        df = df.copy()
        mask_reversed = df[contrast_col].str.contains("Control.*v.*Flight", case=False, regex=True)
        df.loc[mask_reversed, logfc_col] = -df.loc[mask_reversed, logfc_col]
        
        # Filter significant DEGs
        sig_degs = df[df[qval_col] < self.q_value_threshold]
        return sig_degs
