import numpy as np
from collections import Counter



file_name = "sequences_training.txt"


feature_list = []
label_list = []
DNAsum = []
RNAsum = []
nonDRNAsum = []
DRNAsum = []
def getAverage(dnas_list):
	length = len(dnas_list)
	print(f"Length {length}")
	total = 0
	for d in dnas_list:
		total += d

	return total / length
try: 
	with open(file_name, 'r') as file:

		for line in file:

			DNA,classification = line.split(',')
			dnaLength = len(DNA)
			feature_list.append(DNA)
			classification = classification.strip()
			label_list.append(classification)
			match classification:
				case "DNA":

					DNAsum.append(dnaLength)
				case "RNA":
					RNAsum.append(dnaLength)
				case "nonDRNA":
					nonDRNAsum.append(dnaLength)
				case "DRNA":
					DRNAsum.append(dnaLength)

except FileNotFoundError:
	print(f"Error no file")


dna_average = getAverage(DNAsum)
print(f"DNA Average: {dna_average}")
rna_average = getAverage(RNAsum)
print(f"RNA Average: {rna_average}")
nondrna_average = getAverage(nonDRNAsum)
print(f"nonDRNA Average: {nondrna_average}")
drna_average = getAverage(DRNAsum)
print(f"DRNA Average: {drna_average}")
training_cycles = 8
patterns = 1
augmented_inputs = 3
learning_constant = .1

patterns = feature_list # TODO make a np array
text_lables = ("DNA", "RNA", "nonDRNA", "DRNA")
desired_output = ([1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1])

