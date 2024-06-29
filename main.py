from datasets import load_dataset

dataset = load_dataset("HOWRY/Vietnamese-text-recognition")
dataset.push_to_hub("HOWRY/Vietnamese-text-recognition")

