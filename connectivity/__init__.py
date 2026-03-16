from .fiedler import FiedlerValue
from .mst import get_mst_normalized_wt
from .basics import compute_basic_measures
from .lookback import compute_lookback_lens

CONNECTIVITY_MAP = {
	"fiedler": FiedlerValue.using_torch,
	"mst": get_mst_normalized_wt,
	"basics": compute_basic_measures,
	"toha": None,
	"lookback": compute_lookback_lens,
}

def get_connectivity_function(id):
	return CONNECTIVITY_MAP[id]