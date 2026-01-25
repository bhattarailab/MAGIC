from .fiedler import FiedlerValue
from .mst import get_mst_normalized_wt

CONNECTIVITY_MAP = {
	"fiedler": FiedlerValue.using_torch,
	"mst": get_mst_normalized_wt,
}

def get_connectivity_function(id):
	return CONNECTIVITY_MAP[id]