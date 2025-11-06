# Optimization/framework/src/optfw/samplers/__init__.py
from .lhs import LHSSampler
from .sobol import SobolSampler

SAMPLERS = {
    "lhs": LHSSampler,
    "sobol": SobolSampler,
    "random": None,  # optional später
}
