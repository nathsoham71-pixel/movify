"""
Loads your trained HybridRecommender (backend/recom_train/hybrid_model.pkl) ONCE per process.

Why the custom unpickler?
    `python model.py train` runs model.py as __main__, so the pickle remembers the class as
    `__main__.HybridRecommender`. Inside Django there is no such __main__ class, so a plain
    pickle.load() fails with "Can't get attribute 'HybridRecommender'". We import model.py under
    the name `recom_model` and redirect `__main__` to it while unpickling.
"""
import importlib.util
import pickle
import sys
from functools import lru_cache

from django.conf import settings

_MODULE_NAME = "recom_model"


def _import_model_module():
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    path = settings.RECOMMENDER_DIR / "model.py"
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


class _RedirectingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = _MODULE_NAME
        return super().find_class(module, name)


@lru_cache(maxsize=1)
def get_model():
    """Returns the HybridRecommender instance (cached for the life of the process)."""
    _import_model_module()
    with open(settings.RECOMMENDER_MODEL_PATH, "rb") as f:
        return _RedirectingUnpickler(f).load()
