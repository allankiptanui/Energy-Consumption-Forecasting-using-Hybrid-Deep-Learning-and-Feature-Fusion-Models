 
from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
except Exception as e:
    raise ImportError("statsmodels is required for ARIMAModel. Install with `pip install statsmodels`.") from e


OrderType = Union[Tuple[int, int, int], int, None]


class ARIMAModel:
    """Statsmodels ARIMA wrapper suitable for use in training workflows."""

    def __init__(self, order: OrderType = (3, 1, 2), *args, **kwargs):
        """
        Parameters
        ----------
        order : tuple(int,int,int) or int or None
            - Preferred: (p,d,q)
            - If an int is passed positionally (common mistake when build_model forwards input_dim),
              the wrapper will WARN and use default (3,1,2).
            - If an int is passed explicitly as order=5, it's interpreted as p and converted to (5,1,0).
        *args, **kwargs:
            tolerated for flexible constructor calls from build_model helper.
        """
        # If user passed order through kwargs, prefer that
        if "order" in kwargs:
            provided = kwargs.get("order")
        else:
            provided = order

        # If the provided value is a tuple/list of length 3 -> use it
        if isinstance(provided, (tuple, list)) and len(provided) == 3:
            self.order = tuple(int(x) for x in provided)
        # If provided is an int via kwargs: interpret as p -> (p,1,0)
        elif isinstance(provided, int):
 
            self.order = (int(provided), 1, 0)
            warnings.warn(
                f"ARIMAModel received an integer for `order` ({provided}). Interpreting as (p,d,q)=({provided},1,0). "
                "If this was an accidental positional `input_dim`, call ARIMAModel() with no args or with order=(p,d,q).",
                UserWarning,
            )
        else:
            # If provided is something unexpected (e.g. input_dim passed positionally), fall back to default
            self.order = (3, 1, 2)
            # But if args contains a single int and order was left as default param value changed by caller,
            # that's the common case we saw (build_model forwarded input_dim positionally).
            if args and isinstance(args[0], int):
                warnings.warn(
                    "ARIMAModel was constructed with a positional int (likely `input_dim`) — "
                    "ignoring it and using default ARIMA order=(3,1,2). To pick a custom order, pass order=(p,d,q).",
                    UserWarning,
                )
            else:
                # Generic fallback warning
                warnings.warn(
                    "ARIMAModel constructor received unexpected `order` input; using default (3,1,2).",
                    UserWarning,
                )

        self.model_fit: Optional[ARIMAResults] = None
        self.is_fitted: bool = False

    def fit(self, y_train):
        """
        Fit ARIMA to a univariate series.

        y_train may be pandas Series/DataFrame or numpy array or list.
        """
        # coerce to 1-D numpy array
        if isinstance(y_train, pd.DataFrame):
            # if DataFrame has multiple columns, try to find a single numeric column;
            # otherwise take the first column.
            if y_train.shape[1] > 1:
                warnings.warn(
                    "ARIMAModel.fit received a DataFrame with multiple columns; using the first column."
                )
            y_train = y_train.iloc[:, 0].values
        elif isinstance(y_train, pd.Series):
            y_train = y_train.values
        elif isinstance(y_train, (list, tuple)):
            y_train = np.array(y_train)
        elif not isinstance(y_train, np.ndarray):
            y_train = np.asarray(y_train)

        y_train = y_train.squeeze().astype(float)

        if y_train.ndim != 1:
            raise ValueError("ARIMAModel.fit requires a 1-D series-like input (shape (n,)).")

        try:
            model = ARIMA(y_train, order=self.order)
            self.model_fit = model.fit()
            self.is_fitted = True
        except Exception as e:
            raise RuntimeError(f"ARIMA fitting failed (order={self.order}): {e}") from e

    def forecast(self, steps: int = 1):
        """Forecast next `steps` values. Returns numpy array shape (steps,)."""
        if not self.is_fitted or self.model_fit is None:
            raise RuntimeError("ARIMAModel must be fitted before forecasting.")
        preds = self.model_fit.forecast(steps=steps)
        return np.asarray(preds)

    def summary(self):
        if not self.is_fitted or self.model_fit is None:
            return "ARIMAModel: not fitted"
        return self.model_fit.summary()

    def save(self, path: str):
        """Save fitted ARIMA results to disk (statsmodels' .save())."""
        if not self.is_fitted or self.model_fit is None:
            raise RuntimeError("Fit the model before saving.")
        self.model_fit.save(path)

    def load(self, path: str):
        """Load fitted ARIMA results from disk."""
        # ARIMAResults can be loaded using statsmodels' Results.load
        from statsmodels.tsa.arima.model import ARIMAResults

        self.model_fit = ARIMAResults.load(path)
        self.is_fitted = True


 





