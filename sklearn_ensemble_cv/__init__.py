__all__ = ['ensemble_cv']

from sklearn_ensemble_cv.cross_validation import comp_empirical_ecv, cross_validation_ecv
from sklearn_ensemble_cv.ensemble import Ensemble
from sklearn_ensemble_cv.utils import make_grid


__license__ = "MIT"
__version__ = "0.1.0"
__author__ = "Jin-Hong Du"
__email__ = "jinhongd@andraw.cmu.edu"
__maintainer__ = "Jin-Hong Du"
__maintainer_email__ = "jinhongd@andraw.cmu.edu"
__description__ = "Ensemble Cross-validation is a Python package for performing specialized cross-validation on ensemble models, such as extrapolated cross-validation (ECV), generalized cross-validation (GCV), and etc. The implementation of ensemble models are based on scikit-learn."
