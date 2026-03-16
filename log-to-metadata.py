import pickle
import re
import json

"""
Processing sample 3947/3956
Question: What type of polyp is identified in the gastrointestinal tract?
Ground truth: polyp consistent with paris iia classification
The image shows a sessile polyp in the gastrointestinal tract. 
Elapsed time: 2.124650 seconds
Model answer: The image shows a sessile polyp in the gastrointestinal tract.
"""
# regexes
processing_regex = r"Processing sample (\d+)/(\d+)"
question_regex = r"Question:\s(.*)"
ground_truth_regex = r"Ground truth:\s(.*)"
prediction_regex = r"Model answer:\s(.*)"

log_path = "outputs/58966558.log"

in_one_sample = False    # is set to true if processing sample text is appeared if it does then, dumps existing and continue new ones
save_folder = "/scratch/noli1/graph_hallucination_detection/results/llava-med-v1.5-mistral-7b_test_Feb17_Kvasir-VQA-x1_from_12000_to_16000"

metadata = []

with open(log_path, "r") as fp:
	# looping each line to line
	sample_dict = {}
	for line in fp:
		line = line.strip()

		# see if the line has a given format
		if re.match(processing_regex, line):
			# if there was previous than dump it
			if in_one_sample:
				if all([t in sample_dict.keys() for t in ["question", "ground_truth", "prediction", "visual_subgraphs_file", "text_subgraphs_file", "attention_file"]]):
					metadata.append(sample_dict.copy())
					sample_dict = {}
				else:
					raise Exception(f"Exception encountered, not enough keys. {sample_dict}")
			
			in_one_sample = True

			# getting the index and creatign the atttention and visual and text file path
			idx = int(re.match(processing_regex, line).groups()[0])-1+12000

			sample_dict["attention_file"]=f"{save_folder}/attention_{idx}.npz"
			sample_dict["visual_subgraphs_file"] = f"{save_folder}/visual_subgraphs_{idx}.npz"
			sample_dict["text_subgraphs_file"] = f"{save_folder}/text_subgraphs_{idx}.npz"

		if re.match(question_regex, line):
			sample_dict["question"] = re.match(question_regex, line).groups()[0]

		if re.match(ground_truth_regex, line):
			sample_dict["ground_truth"] = re.match(ground_truth_regex, line).groups()[0]

		if re.match(prediction_regex, line):
			sample_dict["prediction"] = re.match(prediction_regex, line).groups()[0]


	metadata.append(sample_dict)
        
with open("./copied_metadata.json", "w") as fp:
	json.dump(metadata, fp, indent=2)