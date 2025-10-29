from sklearn.model_selection import train_test_split
import csv
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from datetime import datetime
feature_list = []
label_list = []
DNAsum = []
RNAsum = []
nonDRNAsum = []
DRNAsum = []
file_name = "sequences_training.txt"
modified_training = "modified_training.txt"
try: 
	with open(modified_training, 'r') as file:

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

# Used for typical scoring accuracy 
NRangeModalOutput = []
def runNRangeModal(nrange_lower, nrange_upper, feature_limit):
    
    information_list = []

    X_data = feature_list
    Y_data = label_list
    start = datetime.now()
    # build the modal
    text_classification_pipeline = Pipeline([
        # Vectorize strings
        ('tfidf',TfidfVectorizer(ngram_range=(3,10), analyzer='char',max_features=feature_limit)),

        # classifier learns to map nubmers to a label
        #('clf', LogisticRegression(max_iter=1000))
        ('clf', DecisionTreeClassifier(random_state=0))
    ])

    print(f"NLower: {nrange_lower} NUpper: {nrange_upper} Max Feature: {feature_limit}")
    information_list.append(nrange_lower)
    information_list.append(nrange_upper)
    information_list.append(feature_limit)

    #getScores(text_classification_pipeline)
    cross_validated_scores = cross_val_score(text_classification_pipeline, X_data, Y_data, cv=5)
    print(cross_validated_scores)
    for score in cross_validated_scores:
        information_list.append(score)

    end = datetime.now()
    elapsed = end - start
    elapsed_time = str(elapsed).split(".")[0]
    print(f"Elapsed Time {elapsed_time}")
    information_list.append(elapsed_time)
    NRangeModalOutput.append(information_list)
# start of main
NRangeModalOutput.append(["NRange_Upper", "NRange_Lower", "Feature_Limit", "Score1", "Score2", "Score3", "Score4", "Score5", "Time Elapsed"])
nLower = 6
nUpper = 20
lowerFeature = 50000
upperFeature = 200000
for n in range (nLower,nUpper):
    for max_feat in range (lowerFeature, upperFeature, 5000):
        runNRangeModal(nLower, n, max_feat)

with open('NRangeDecisionTree.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(NRangeModalOutput)
