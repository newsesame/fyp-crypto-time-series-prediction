import pandas as pd


def apply_pandas_finta_compatibility() -> None:
    """Apply compatibility fix for pandas>=2 and finta library
    
    pandas>=2 removed iteritems; finta still uses it
    This function adds the iteritems method back to Series for compatibility
    """
    if not hasattr(pd.Series, "iteritems"):
        pd.Series.iteritems = pd.Series.items  # type: ignore[attr-defined]


__all__ = ["apply_pandas_finta_compatibility"]


