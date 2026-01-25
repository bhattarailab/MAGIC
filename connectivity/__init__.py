from .fiedler import FiedlerValue
from .mst import get_mst_normalized_wt
from .basics import compute_basic_measures

CONNECTIVITY_MAP = {
	"fiedler": FiedlerValue.using_torch,
	"mst": get_mst_normalized_wt,
	"basics": compute_basic_measures,
}

def get_connectivity_function(id):
	return CONNECTIVITY_MAP[id]