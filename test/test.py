from .test_set import seeds, human_subjects, human_prompts, animal_subjects, animal_prompts

def test_no_conflict():
	for seed in seeds:
		for subject in human_subjects:
			for prompt in human_prompts:
				prompt = prompt.format(subject=subject)
				reference = f"/root/Desktop/workspace/SmartControl/assets/{subject}/{" ".join(prompt.split(" ")[1:])}"
				# run sampling

		for subject in animal_subjects:
			for prompt in animal_prompts:
				prompt = prompt.format(subject=subject)
				reference = f"/root/Desktop/workspace/SmartControl/assets/{subject}/{" ".join(prompt.split(" ")[1:])}"
				# run sampling

	# CLIP score
	# ImageReward
	# Picscore

def test_mild_conflict():
	for seed in seeds:
		for subject in human_subjects:
			for subject2 in human_subjects:
				if subject == subject2:
					continue
				for prompt in human_prompts:
					prompt = prompt.format(subject=subject)
					reference = f"/root/Desktop/workspace/SmartControl/assets/{subject2}/{" ".join(prompt.split(" ")[1:])}"

					# run sampling

		for subject in animal_subjects:
			for subject2 in animal_subjects:
				if subject == subject2:
					continue
				for prompt in animal_prompts:
					prompt = prompt.format(subject=subject)
					reference = f"/root/Desktop/workspace/SmartControl/assets/{subject2}/{" ".join(prompt.split(" ")[1:])}"
					# run sampling

	# CLIP score
	# ImageReward
	# Picscore

def test_significant_conflict():
	for seed in seeds:
		for subject in human_subjects:
			for subject2 in animal_subjects:
				for prompt in animal_prompts:
					prompt = prompt.format(subject=subject)
					reference = f"/root/Desktop/workspace/SmartControl/assets/{subject2}/{" ".join(prompt.split(" ")[1:])}"
					# run sampling

		for subject in animal_subjects:
			for subject2 in human_subjects:
				for prompt in human_prompts:
					prompt = prompt.format(subject=subject)
					reference = f"/root/Desktop/workspace/SmartControl/assets/{subject2}/{" ".join(prompt.split(" ")[1:])}"
					# run sampling

	# CLIP score
	# ImageReward
	# Picscore
