"""Simple feature fusion utilities for combining historical & exogenous features."""
from typing import Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


def simple_concatenate(historical_seq: np.ndarray, exogenous_seq: np.ndarray) -> np.ndarray:
    """
    Concatenate along features axis.
    historical_seq: (N, seq_len, n_hist)
    exogenous_seq: (N, seq_len, n_exog)
    returns: (N, seq_len, n_hist + n_exog)
    """
    try:
        return np.concatenate([historical_seq, exogenous_seq], axis=-1)
    except Exception:
        logger.exception("Failed to concatenate sequences for fusion.")
        raise
