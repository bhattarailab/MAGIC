import pickle
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--old")
parser.add_argument("--new")

args = parser.parse_args()
old_path, new_path = args.old, args.new

with open(old_path, "rb") as fp:
	old_data = pickle.load(fp)

with open(new_path, "rb") as fp:
	new_data = pickle.load(fp)

# assertion to see if total number of data are consistent
assert(len(new_data) == len(old_data))

incorrect_total_question_mismatch = 0
incorrect_prediction_for_matched = 0
incorrect_prediction_hallucination_not_matched = 0
correct_prediction_hallucination_not_matched = 0
total_matched = 0
for old_index, old_sample in enumerate(old_data):
	for new_index, new_sample in enumerate(new_data):
		# find the correct one to one mapping
		if old_sample["visual_subgraphs_file"].split("/")[-1] == new_sample["visual_subgraphs_file"].split("/")[-1]:
			# compare the question and ground truth to see such that the sample is correct
			if old_sample["question"] == new_sample["question"] and old_sample["ground_truth"] == new_sample["ground_truth"]:
				# check if the prediction is differnt or not
				if old_sample["prediction"] != new_sample["prediction"]:
					incorrect_prediction_for_matched+=1

					if old_sample["is_hallucinating"] != new_sample["is_hallucinating"]:
						incorrect_prediction_hallucination_not_matched += 1
				else:
					total_matched+=1
					if old_sample["is_hallucinating"] != new_sample["is_hallucinating"]:
						correct_prediction_hallucination_not_matched += 1
			else:
				incorrect_total_question_mismatch+=1
			break
	else:
		raise Exception("Not found error")

print(f"Total samples = {len(old_data)}")
print(f"Total matches = {total_matched}")
print(f"Total incorrect question mismatch but attention file name match = {incorrect_total_question_mismatch}")
print(f"Total question and attention name matched but precition doesnot matches = {incorrect_prediction_for_matched}")
print(f"Total answer correct but hallucination mismsatch = {correct_prediction_hallucination_not_matched}")
print(f"Total incorrect prediction plus hallucination mismatch = {incorrect_prediction_hallucination_not_matched}")
