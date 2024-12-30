import matplotlib.pyplot as plt
import pandas as pd
import torch


def visualize_unet_similarities(csv_filepath, target_block: str):
	"""
	Visualizes cosine similarity scores for target blocks from a CSV file.

	Args:
		csv_filepath: Path to the CSV file.
	"""
	try:
		df = pd.read_csv(csv_filepath, delimiter=' ')
	except FileNotFoundError:
		print(f"Error: File not found at {csv_filepath}")
		return

	# Filter data for target blocks
	df_unet = df[df['model'] == target_block]

	# Reverse the order of timesteps for x-axis flipping
	df_unet = df_unet.sort_values('timestep', ascending=False)

	# Create the visualization
	plt.figure(figsize=(10, 6))  # Adjust figure size as needed
	plt.plot(df_unet['timestep'], df_unet['similarity'], marker='o', linestyle='None')
	plt.xlabel('Timestep')
	plt.ylabel('Cosine Similarity')
	plt.title(f'Cosine Similarity of ControlNet\'s {target_block} Over Timesteps')
	plt.grid(True)
	plt.ylim(0, 1)
	plt.gca().invert_xaxis() # Invert x-axis
	plt.savefig("similarity.png")

def compare_latent(l1_path: str, l2_path: str):
    """
    Loads two tensors from specified filepaths, calculates and prints their cosine similarity.

    Args:
        l1_path: Path to the first tensor file (.pt).
        l2_path: Path to the second tensor file (.pt).
    """
    try:
        tensor1 = torch.load(l1_path)
        tensor2 = torch.load(l2_path)
    except FileNotFoundError:
        print(f"Error: One or both files not found.")
        return
    except Exception as e:
        print(f"An error occurred during loading: {e}")
        return

    similarity = torch.nn.functional.cosine_similarity(tensor1, tensor2) #Cosine similarity is 1 - cosine distance
    print(f"Cosine similarity between tensors: {similarity.mean().item()}")

# Example usage:
visualize_unet_similarities('./similarity.csv', "UNetMidBlock2DCrossAttn")
# compare_latent("./final_latent_ControlNet", "./final_latent_smartControl")

