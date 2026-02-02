import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

from utils.toha import compute_normalized_connectivity_diff, select_heads, prediction
from utils.logger import Logger

parser = argparse.ArgumentParser(description="Based on connectivity select the best heads and inferences on the test set to report the metics.")
parser.add_argument("--train", help="Path to training pickle file output from extract-connectivity.py")
parser.add_argument("--test", help="Path to testing pickle file output from extract-connectivity.py")
parser.add_argument("--name", "-n", help="name to be given for the experiment", default="exp")
parser.add_argument("--visual-weight", "-v", help="weight for visual connectivity value", default=-1, type=float)
parser.add_argument("--text-weight", "-t", help="weight for visual connectivity value", default=1, type=float)
parser.add_argument("--type", help="type of the conectivity mesure to be used")
parser.add_argument("--max-heads", help="maximum number of heads", default=10, type=int)
parser.add_argument("--n-train", help="Number of training samples", type=int, default=200)
parser.add_argument("--log", "-l", help="Log the connectivity measure", action="store_true")

args = parser.parse_args()
train_connectivity_path, test_connectivity_path = args.train, args.test
exp_name = args.name
visual_wt, text_wt = args.visual_weight, args.text_weight
connectivity_type = args.type
max_heads = args.max_heads
n_train = args.n_train

logger = Logger(f"/outputs/{exp_name}.json") if args.log else None
logger.add("connectivity_measure", connectivity_type)
logger.add("visual_weight", visual_wt)
logger.add("text_weight", text_wt)
logger.add("files", {
	"train": train_connectivity_path,
	"test": test_connectivity_path,
})

with open(train_connectivity_path, "rb") as fp:
	# each sample contains "connectivity" with dict of keys "layer_0_head_0" with "text" and "visual"
	train_connectivity = pickle.load(fp)
	n_train = min(n_train, len(train_connectivity))
	train_connectivity = train_connectivity[:n_train]

with open(test_connectivity_path, "rb") as fp:
	# each sample contains "connectivity" with dict of keys "layer_0_head_0" with "text" and "visual"
	test_connectivity = pickle.load(fp)

print(f"Total number of train samples = {len(train_connectivity)}")
print(f"Total number of test samples = {len(test_connectivity)}\n")

# for the train sets; COmputing the heads
hallucinating_samples = [m for m in train_connectivity if m["is_hallucinating"]]
non_hallucinating_samples = [m for m in train_connectivity if not m["is_hallucinating"]]
print(f"Total {len(hallucinating_samples)} and {len(non_hallucinating_samples)} of hallucinating and non-hallucinating samples.")

# logs the current experiment number of train and test samples
if logger:
	logger.add("n_train_samples", {
		"total": len(train_connectivity),
		"hallucinating": len(hallucinating_samples),
		"non_hallucinating": len(non_hallucinating_samples),
	})

	logger.add("n_test_samples", {
		"total": len(test_connectivity),
		"hallucinating": sum([True if m["is_hallucinating"] else False for m in test_connectivity]),
		"non_hallucinating": sum([False if m["is_hallucinating"] else True for m in test_connectivity]),
	})

hallucinating_fiedlers_diff = compute_normalized_connectivity_diff(hallucinating_samples, t_weight=text_wt, v_weight=visual_wt, type=connectivity_type)
non_hallucinating_fiedlers_diff = compute_normalized_connectivity_diff(non_hallucinating_samples, t_weight=text_wt, v_weight=visual_wt, type=connectivity_type)

divergence_scores = hallucinating_fiedlers_diff-non_hallucinating_fiedlers_diff

selected_heads, train_auroc = select_heads(divergence_scores, train_connectivity, max_heads=max_heads, t_weight=text_wt, v_weight=visual_wt, type=connectivity_type)

if logger:
	logger.add("selected_heads", selected_heads)

# creating the grid for divergence score visualization
ax = sns.heatmap(divergence_scores)
plt.xlabel("heads")
plt.ylabel("layers")
plt.savefig(f"./outputs/{exp_name}.png")
plt.close()

test_auroc = prediction(selected_heads, test_connectivity, t_weight=text_wt, v_weight=visual_wt, type=connectivity_type)

if logger:
	logger.add("auroc", {
		"train": train_auroc,
		"test": test_auroc,
	})
