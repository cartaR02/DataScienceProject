from sklearn.model_selection import train_test_split
import csv
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

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

def runTFIDDecisionTree():
  
    NRangeModalOutput = []
    # start of main
    NRangeModalOutput.append(["NRange_Upper", "NRange_Lower", "Feature_Limit", "Max_depth", "Score1", "Score2", "Score3", "Score4", "Score5", "Time Elapsed"])
    nLower = 3
    nUpper = 10
    lowerFeature = 5000
    upperFeature = 20000
    print("--------------Decision Tree-----------------")
    for n in range (nLower,nUpper):
        for max_feat in range (lowerFeature, upperFeature, 5000):
            for max_depth in range(10, 20):
                NRangeModalOutput.append(runNRangeModal(nLower, n, max_feat, max_depth))
    with open('NDecisionTree.csv', 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(NRangeModalOutput)

def runNRangeModal(nrange_lower, nrange_upper, feature_limit, depth):
    
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
        ('clf', DecisionTreeClassifier(random_state=0,
                                       max_depth=depth))

    ])

    print(f"NLower: {nrange_lower} NUpper: {nrange_upper} Max Feature: {feature_limit} Max Depth: {depth}")
    information_list.append(nrange_lower)
    information_list.append(nrange_upper)
    information_list.append(feature_limit)
    information_list.append(depth)

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
    return information_list

def runTFIDLogisticRegression():
    LogisticOutput = []
    nLower = 3
    nUpper = 10
    lowerFeature = 5000
    upperFeature = 55000
    
    LogisticOutput.append(["NRange_Upper", "NRange_Lower", "Feature_Limit", "Score1", "Score2", "Score3", "Score4", "Score5", "Time Elapsed"])
    print("--------------Logistic Regression-----------------")
    for n in range (nLower, nUpper):
        print("**********New nLower and nUpper************")
        for max_feat in range (lowerFeature, upperFeature, 10000):
            LogisticOutput.append(runLogisticModal(nLower, n, max_feat))
    with open('NLogisticOutput.csv', 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(LogisticOutput)

def runLogisticModal(nrange_lower, nrange_upper, feature_limit):
    
    information_list = []

    X_data = feature_list
    Y_data = label_list
    start = datetime.now()
    # build the modal
    text_classification_pipeline = Pipeline([
        # Vectorize strings
        ('tfidf',TfidfVectorizer(ngram_range=(nrange_lower, nrange_upper), analyzer='char',max_features=feature_limit)),

        ('clf', LogisticRegression(
                                   max_iter=20000,
                                   solver='saga',
                                   class_weight='balanced',
                                   C=.1,
                                   n_jobs=-1
        ))
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
    return information_list


def wrapperTFIDRandomForest():
    RandomForestOutput = []
    nLower = 3
    nUpper = 10
    lowerFeature = 5000
    upperFeature = 55000
    
    RandomForestOutput.append(["NRange_Upper", "NRange_Lower", "Feature_Limit", "Score1", "Score2", "Score3", "Score4", "Score5", "Time Elapsed"])
    print("--------------Random Tree-----------------")
    for n in range (nLower, nUpper):
        print("**********New nLower and nUpper************")
        for max_feat in range (lowerFeature, upperFeature, 10000):
            RandomForestOutput.append(runRandomForestModal(nLower, n, max_feat))
    with open('NRandomForest.csv', 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(RandomForestOutput)

def runRandomForestModal(nrange_lower, nrange_upper, feature_limit):
    
    information_list = []

    X_data = feature_list
    Y_data = label_list
    start = datetime.now()
    # build the modal
    text_classification_pipeline = Pipeline([
        # Vectorize strings
        ('tfidf',TfidfVectorizer(ngram_range=(nrange_lower, nrange_upper), analyzer='char',max_features=feature_limit)),

        ('clf', RandomForestClassifier(
                                       n_estimators=300,
                                       max_depth=None,
                                       min_samples_split=2,
                                       min_samples_leaf=1,
                                       max_features='sqrt',
                                       n_jobs=1,
                                       random_state=42))
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
    return information_list
def wrapperSGDCClassifier():
    SClassifier= []
    nLower = 2
    nUpper = 8
    lowerFeature = 15000
    upperFeature = 20000
    increaseBy = 2000
    TestingOptions = False
    SClassifier.append(["NRange_Upper", "NRange_Lower", "Feature_Limit", "Score1", "Score2", "Score3", "Score4", "Score5", "Time Elapsed"])
    print("--------------SGDC Classifier-----------------")
    if TestingOptions:

        for n in range (nLower, nUpper):
            print("**********New nLower and nUpper************")
            for max_feat in range (lowerFeature, upperFeature, increaseBy):
                SClassifier.append(runSClassifier(nLower, n, max_feat))
    else:
        # current 
        SClassifier.append(runSClassifier(2,7,19000))
    with open('SGDCClassifier.csv', 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(SClassifier)

def runSClassifier(nrange_lower, nrange_upper, feature_limit):
    
    information_list = []

    X_data = feature_list
    Y_data = label_list
    X_train, X_test, Y_train, Y_test = train_test(X_data,
                                                  Y_data,
                                                  test_size = .2
                                                  random_state=42
                                                  )

    start = datetime.now()
    # build the modal
    text_classification_pipeline = Pipeline([
        # Vectorize strings
        ('tfidf',TfidfVectorizer(ngram_range=(nrange_lower, nrange_upper), analyzer='char',max_features=feature_limit)),

        ('clf',SGDClassifier (
                                loss='log_loss',
                                class_weight='balanced',
                                max_iter=4000,
                                alpha=1e-4,
                                n_jobs=-1
                                ))
    ])
    print(f"NLower: {nrange_lower} NUpper: {nrange_upper} Max Feature: {feature_limit}")
    information_list.append(nrange_lower)
    information_list.append(nrange_upper)
    information_list.append(feature_limit)
    crossFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    #getScores(text_classification_pipeline)
    cross_validated_pred = cross_val_predict(text_classification_pipeline, X_train, Y_train, cv=crossFold, n_jobs=-1)
    #print(cross_validated_scores)
    for score in cross_validated_pred:
        information_list.append(score)


    text_classification_pipeline.fit(X_data, Y_data)
    confusion = confusion_matrix(Y_train, cross_validated_pred)
    feature_names = text_classification_pipeline.named_steps['tfidf'].get_feature_names_out()
    print(f"Feature names\n{feature_names}")
    print(confusion)
    
    end = datetime.now()
    elapsed = end - start
    elapsed_time = str(elapsed).split(".")[0]
    print(f"Elapsed Time {elapsed_time}")
    information_list.append(elapsed_time)
    return information_list


def main():

    #runTFIDDecisionTree()
    #runTFIDLogisticRegression()
    #wrapperTFIDRandomForest()
    wrapperSGDCClassifier()
	# TODO work on getting information out of this and getting the accuracy in  the confusion matrix
if __name__ == "__main__":
    main()
