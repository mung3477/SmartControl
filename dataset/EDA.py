import json

import matplotlib.pyplot as plt
import pandas as pd


def parse_dataset(fp: str):
	with open(fp, "r") as dataset_json:
		dataset = json.load(dataset_json)
		dataset = pd.DataFrame(dataset)["source"].to_list()
		subjects = [tuple(data.split("/")[-1].split("_")[1 : 3]) for data in dataset]

		items = pd.Series(subjects).value_counts().items()
		dataset_df = pd.DataFrame([[source, dest, num] for ((source, dest), num) in items], columns=["source", "target", "numbers"])
		# print(dataset_df.set_index(["source", "target"]).sort_index().to_string())
		print(dataset_df.set_index(["source"]).groupby("source").sum("numbers").sort_values("numbers", ascending=False).to_string())

parse_dataset("./seg.json")
