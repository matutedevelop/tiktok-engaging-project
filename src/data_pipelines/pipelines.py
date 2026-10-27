import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter

from utils import parsear_challenges


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Reemplaza cada categoría por log(1 + veces que aparece en train).

    No usa el target. Categorías no vistas en train reciben 0.
    """

    def fit(self, X, y=None):
        _X = pd.DataFrame(X)
        self.conteos_ = [_X[c].value_counts() for c in _X.columns]
        self.n_features_in_ = _X.shape[1]
        return self

    def transform(self, X):
        _X = pd.DataFrame(X)
        _salida = [
            np.log1p(_X[c].map(_conteo).fillna(0).to_numpy(dtype=float))
            for c, _conteo in zip(_X.columns, self.conteos_)
        ]
        return np.column_stack(_salida)


class HashtagFrequency(BaseEstimator, TransformerMixin):
    """Popularidad de los hashtags de cada video según cuántas veces aparecen en train.

    Devuelve, por video: media y máximo de log(1 + frecuencia) de sus hashtags y la
    fracción de sus hashtags que no aparecen en train. No usa el target.
    """

    def fit(self, X, y=None):
        _conteo = Counter()
        for _valor in np.asarray(X).ravel():
            _conteo.update(parsear_challenges(_valor))
        self.conteos_ = _conteo
        return self

    def transform(self, X):
        _filas = []
        for _valor in np.asarray(X).ravel():
            _hashtags = parsear_challenges(_valor)
            if not _hashtags:
                _filas.append((0.0, 0.0, 0.0))
                continue
            _frec = np.log1p([self.conteos_.get(h, 0) for h in _hashtags])
            _filas.append((_frec.mean(), _frec.max(), float(np.mean(_frec == 0))))
        return np.asarray(_filas, dtype=float)


