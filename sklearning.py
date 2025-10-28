# 1. Import necessary modules
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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
			match classification:
				case "DNA":
					print("DNA")

					DNAsum.append(dnaLength)
				case "RNA":
					RNAsum.append(dnaLength)
				case "nonDRNA":
					nonDRNAsum.append(dnaLength)
				case "DRNA":
					DRNAsum.append(dnaLength)

except FileNotFoundError:
	print(f"Error no file")
# 2. Load a dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target # Target variable

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Initialize and train a model
knn = KNeighborsClassifier(n_neighbors=3) # Create a K-Nearest Neighbors classifier
knn.fit(X_train, y_train) # Train the model on the training data

# 5. Make predictions on the test set
y_pred = knn.predict(X_test)

# 6. Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
