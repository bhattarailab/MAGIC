import pandas as pd
import argparse

from utils.toha import select_heads, prediction

parser = argparse.ArgumentParser(description="Based on connectivity select the best heads and inferences on the test set to report the metics.")
parser.add_argument("--csv", required=True, help="CSV of created dataset")
parser.add_argument("--max-heads", help="maximum number of heads", default=25, type=int)

args = parser.parse_args()
max_heads = args.max_heads
csv = args.csv

# extracting the csv file
df = pd.read_csv(csv)

n_hallucinating_train = len(df[(df["split"] == "train") & (df["label"])])
n_hallucinating_test = len(df[(df["split"] == "test") & (df["label"])])

print(f"Total number of train samples = {(df['split'] == 'train').sum()} with hallucinating = {n_hallucinating_train}")
print(f"Total number of test samples = {(df['split'] == 'test').sum()} with hallucinating = {n_hallucinating_test}\n")

# divide into train and test samples
train_df, test_df = df[df["split"] == "train"], df[df["split"] == "test"]

# select the heads using toha approach
selected_heads, train_auroc = select_heads(train_df, max_heads=max_heads)
test_auroc = prediction(selected_heads, test_df)
