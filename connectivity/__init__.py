from .fiedler import FiedlerValue

CONNECTIVITY_MAP = {
	"fiedler": FiedlerValue,
}

def get_connectivity_function(id):
	return CONNECTIVITY_MAP[id]