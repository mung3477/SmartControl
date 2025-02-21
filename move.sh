#!/bin/bash

old_dir="mild_conflict"
new_dir="significant_conflict"

animals=("dog" "cat" "teddy" "turtle" "racoon")
humans=("baby" "child" "woman" "grandpa" "spiderman")

for dir_name in ./test/output/mild_conflict/SmartControl/*
do
	# Extract the second word after '-'
	word_after_dash=$(echo "$dir_name" | awk -F' - ' '{print $2}' | awk '{print $2}')

	# Extract the second word after 'with'
	word_after_with=$(echo "$dir_name" | awk -F' with ' '{print $2}' | awk '{print $2}')

	# Function to check if a word is in an array
	is_in_group() {
		local word=$1
		shift
		local group=("$@")

		for item in "${group[@]}"; do
			if [[ "$word" == "$item" ]]; then
				return 0  # Found in group
			fi
		done
		return 1  # Not found in group
	}

	# Compare the extracted words
	# if [ "$word_after_dash" != "$word_after_with" ]; then
	if (is_in_group "$word_after_dash" "${animals[@]}" && is_in_group "$word_after_with" "${humans[@]}") || \
   		(is_in_group "$word_after_dash" "${humans[@]}" && is_in_group "$word_after_with" "${animals[@]}"); then
		new_path=$(echo $dir_name | sed -E "s#/${old_dir}/#/${new_dir}/#")
		mv "$dir_name" "$new_path"
		echo "moved $dir_name to $new_path"
	else
		continue
	fi

done
