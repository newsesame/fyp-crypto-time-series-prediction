import pandas as pd


def apply_pandas_finta_compatibility() -> None:
    # pandas>=2 移除了 iteritems；finta 仍使用它
    if not hasattr(pd.Series, "iteritems"):
        pd.Series.iteritems = pd.Series.items  # type: ignore[attr-defined]


__all__ = ["apply_pandas_finta_compatibility"]


