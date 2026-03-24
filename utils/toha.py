from sklearn.metrics import roc_auc_score, average_precision_score

def select_heads(df, max_heads=25):
	hallucinating_df, non_hallucinating_df = df[df["label"]].drop(columns=["split", "label"]), df[~df["label"]].drop(columns=["split", "label"])

	# compute the distance scrore of train set for all the heads
	train_distances = hallucinating_df.mean() - non_hallucinating_df.mean()
	descending_distances_series = train_distances.sort_values(ascending=False)
	descending_keys = descending_distances_series.index

	n_heads_max = max_heads

	N, N_optimal = 1, 1
	heads = list()

	p_s = {s: 0 for s in range(len(df))}
	auc_max = 0.0

	while N<=n_heads_max:
		heads.append(descending_keys[N-1])

		for i in range(len(df)):
			s = df.iloc[i]
			key = heads[-1]

			p_s[i] = (N-1)/N * p_s[i] + 1/N * s[key]

		y_s = df["label"].values
		auc = roc_auc_score(y_s, list(p_s.values()))

		if auc > auc_max:
			auc_max = auc
			N_optimal = N

		N = N+1

	print(f"Total {N_optimal} heads selected with maximum AUC of {auc_max}")
	
	return heads[:N_optimal], auc_max

def prediction(heads, df):
	y_s = df["label"].values
	n_heads = len(heads)

	p_s = []
	for i in range(len(df)):
		s = df.iloc[i]
		p_s.append(s[heads].values.sum()/n_heads)

	auc_score = roc_auc_score(y_s, p_s)
	aupr_score = average_precision_score(y_s, p_s)
	print(f"AUC score = {auc_score}, AUPR score = {aupr_score}")

	return auc_score, aupr_score