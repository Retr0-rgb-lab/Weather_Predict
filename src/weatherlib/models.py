"""Model factory. R1: linear models only (Ridge, Lasso). Trees land in R2,
MLP in R3 (spec v1.1 section 2.3). Every model is a sklearn Pipeline whose
StandardScaler is fit on the training slice only -> no leakage.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-2, 4, 25)


def fit_ridge(Xtr: np.ndarray, ytr: np.ndarray, alphas=ALPHAS):
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    model.fit(Xtr, ytr)
    return model


def fit_lasso(Xtr: np.ndarray, ytr: np.ndarray, alphas=ALPHAS, max_iter=5000, cv=5, random_state=0):
    model = make_pipeline(StandardScaler(), LassoCV(alphas=alphas, max_iter=max_iter, cv=cv, random_state=random_state))
    model.fit(Xtr, ytr)
    return model
