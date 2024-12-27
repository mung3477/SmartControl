import pandas as pd
import matplotlib.pyplot as plt

def visualize_unet_similarities(csv_filepath):
	"""
	Visualizes cosine similarity scores for CrossAttnUpBlock2D blocks from a CSV file.

	Args:
		csv_filepath: Path to the CSV file.
	"""
	try:
		df = pd.read_csv(csv_filepath, delimiter=' ')
	except FileNotFoundError:
		print(f"Error: File not found at {csv_filepath}")
		return

	# Filter data for CrossAttnUpBlock2D blocks
	df_unet = df[df['model'] == 'CrossAttnUpBlock2D']

	# Reverse the order of timesteps for x-axis flipping
	df_unet = df_unet.sort_values('timestep', ascending=False)

	# Create the visualization
	plt.figure(figsize=(10, 6))  # Adjust figure size as needed
	plt.plot(df_unet['timestep'], df_unet['similarity'], marker='o', linestyle='None')
	plt.xlabel('Timestep')
	plt.ylabel('Cosine Similarity')
	plt.title('Cosine Similarity of SmartControl\'s CrossAttnUpBlock2D Blocks Over Timesteps')
	plt.grid(True)
	plt.gca().invert_xaxis() # Invert x-axis
	plt.savefig("similarity.png")

# Example usage:
visualize_unet_similarities('./log/latents/similarity.csv')

