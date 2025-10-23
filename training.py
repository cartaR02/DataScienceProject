file_name = "sequences_training.txt"

feature_list = []
label_list = []

try: 
	with open(file_name, 'r') as file:

		for line in file:

			DNA,classification = line.split(',')
			dnaLength = len(DNA)
			
			print(f"{dnaLength}: {classification}")

except FileNotFoundError:
	print(f"Error no file")
