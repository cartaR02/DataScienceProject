import numpy as np
from collections import Counter



file_name = "sequences_training.txt"


feature_list = []
label_list = []

try: 
	with open(file_name, 'r') as file:

		for line in file:

			DNA,classification = line.split(',')
			dnaLength = len(DNA)
			feature_list.append(DNA)
			label_list.append(classification)

			print(f"{dnaLength}: {classification}")

except FileNotFoundError:
	print(f"Error no file")

counts = Counter(label_list)
print(len(feature_list))
print(len(label_list))
for cls, count in counts.items():
	print(f"{cls}: {count}")

training_cycles = 8
patterns = 1
augmented_inputs = 3
learning_constant = .1

patterns = feature_list # TODO make a np array
text_lables = ("DNA", "RNA", "nonDRNA", "DRNA")
desired_output = ([1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1])

