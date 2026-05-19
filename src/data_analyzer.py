"""
SciVizKit — Data Analyzer
Detects column types and determines compatible chart types for a given DataFrame.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Any

from .chart_registry import CHART_REGISTRY


class DataAnalyzer:
    """Profiles a DataFrame and maps it to compatible chart types."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.profile = self._analyze()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyze(self) -> Dict[str, Any]:
        df = self.df
        col_types: Dict[str, str] = {}
        numeric_cols: List[str] = []
        categorical_cols: List[str] = []
        datetime_cols: List[str] = []
        binary_cols: List[str] = []
        high_cardinality_cols: List[str] = []

        for col in df.columns:
            series = df[col].dropna()
            n_unique = series.nunique()

            # Try to parse as datetime if object type
            if pd.api.types.is_datetime64_any_dtype(series):
                col_types[col] = "datetime"
                datetime_cols.append(col)
            elif pd.api.types.is_numeric_dtype(series):
                if n_unique == 2:
                    col_types[col] = "binary"
                    binary_cols.append(col)
                    numeric_cols.append(col)
                else:
                    col_types[col] = "numeric"
                    numeric_cols.append(col)
            elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
                # Try datetime parse
                if series.dtype == object:
                    try:
                        pd.to_datetime(series.head(20))
                        col_types[col] = "datetime"
                        datetime_cols.append(col)
                        continue
                    except Exception:
                        pass
                if n_unique == 2:
                    col_types[col] = "binary"
                    binary_cols.append(col)
                    categorical_cols.append(col)
                elif n_unique <= 50:
                    col_types[col] = "categorical"
                    categorical_cols.append(col)
                else:
                    col_types[col] = "high_cardinality"
                    high_cardinality_cols.append(col)
            else:
                col_types[col] = "other"

        return {
            "col_types": col_types,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "datetime_cols": datetime_cols,
            "binary_cols": binary_cols,
            "high_cardinality_cols": high_cardinality_cols,
            "columns": list(df.columns),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def col_types(self) -> Dict[str, str]:
        return self.profile["col_types"]

    @property
    def numeric_cols(self) -> List[str]:
        return self.profile["numeric_cols"]

    @property
    def categorical_cols(self) -> List[str]:
        return self.profile["categorical_cols"]

    @property
    def datetime_cols(self) -> List[str]:
        return self.profile["datetime_cols"]

    def get_compatible_charts(self, domain: str = "General") -> List[str]:
        """Return list of chart IDs compatible with current data and domain."""
        p = self.profile
        n_num = len(p["numeric_cols"])
        n_cat = len(p["categorical_cols"])
        n_dt = len(p["datetime_cols"])
        n_rows = p["n_rows"]

        compatible = []
        for chart_id, meta in CHART_REGISTRY.items():
            req = meta.get("requires", {})
            domains = meta.get("domains", ["General"])

            # Domain check — "General" charts appear in all domains
            if domain not in domains and "General" not in domains:
                continue

            # Requirements check
            if n_num < req.get("min_numeric", 0):
                continue
            if n_cat < req.get("min_categorical", 0):
                continue
            if n_dt < req.get("min_datetime", 0):
                continue
            if n_rows < req.get("min_rows", 0):
                continue

            compatible.append(chart_id)

        return compatible

    def summary_text(self) -> str:
        p = self.profile
        lines = [
            f"**Rows:** {p['n_rows']} | **Columns:** {p['n_cols']}",
            f"**Numeric:** {len(p['numeric_cols'])} | **Categorical:** {len(p['categorical_cols'])} | "
            f"**Datetime:** {len(p['datetime_cols'])} | **Binary:** {len(p['binary_cols'])}",
        ]
        return "\n".join(lines)
