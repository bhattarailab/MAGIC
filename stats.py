import pickle

path = "./outputs/basics_train_all.pkl"
with open(path, "rb") as fp:
	data = pickle.load(fp)

total = len(data)
hallucinating = sum([True if d["is_hallucinating"] else False for d in data])
print(total)
print(hallucinating)
print(total-hallucinating)