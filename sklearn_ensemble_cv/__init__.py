__all__ = ['ECV', 'GCV', 'splitCV', 'KFoldCV', 'reset_random_seeds', 'Ensemble', 'generate_data']

from sklearn_ensemble_cv.cross_validation import ECV, GCV, splitCV, KFoldCV
from sklearn_ensemble_cv.ensemble import Ensemble
from sklearn_ensemble_cv.utils import reset_random_seeds
from sklearn_ensemble_cv.simu_data import generate_data


__license__ = "MIT"
__version__ = "0.2.0"
__author__ = "Jin-Hong Du"
__email__ = "jinhongd@andrew.cmu.edu"
__maintainer__ = "Jin-Hong Du"
__maintainer_email__ = "jinhongd@andrew.cmu.edu"
__description__ = ("Ensemble Cross-validation is a Python package for performing specialized " 
    "cross-validation on ensemble models, such as extrapolated cross-validation (ECV), "
    "generalized cross-validation (GCV), and etc. The implementation of ensemble models are "
    "based on scikit-learn."
    )
