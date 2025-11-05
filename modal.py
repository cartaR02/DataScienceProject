from sklearn.model_selection import train_test_split
import csv
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier

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
    X_train, X_test, Y_train, Y_test = train_test_split(X_data,
                                                  Y_data,
                                                  test_size = .2,
                                                  random_state=42
                                                  )

    start = datetime.now()
    # build the modal
    text_classification_pipeline = Pipeline([
        # Vectorize strings
        ('tfidf',TfidfVectorizer(ngram_range=(nrange_lower, nrange_upper), analyzer='char',max_features=feature_limit)),

        ('clf',SGDClassifier (
                                loss='log_loss',
                                max_iter=4000,
                                class_weight='balanced',
                                alpha=1e-4,
                                n_jobs=-1
                                ))
    ])
    print(f"NLower: {nrange_lower} NUpper: {nrange_upper} Max Feature: {feature_limit}")
    information_list.append(nrange_lower)
    information_list.append(nrange_upper)
    information_list.append(feature_limit)

    
    # --- 1. CROSS-VALIDATION on 80% TRAINING SET ---
    print("Running 5-fold cross-validation on training data...")
    crossFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cross_validated_pred = cross_val_predict(text_classification_pipeline, X_train, Y_train, cv=crossFold, n_jobs=-1)

    # --- 2. GET CV SCORES ---
    cv_accuracy = accuracy_score(Y_train, cross_validated_pred)
    cv_precision, cv_recall, cv_f1, _ = precision_recall_fscore_support(Y_train, cross_validated_pred, average='weighted')
    
    # <-- FIX: Append the summary scores, not the thousands of predictions
    information_list.append(cv_accuracy)
    information_list.append(cv_precision)
    information_list.append(cv_recall)
    information_list.append(cv_f1)
    
    print("\n--- CROSS-VALIDATION RESULTS (ESTIMATE) ---")
    print(f"CV Accuracy: {cv_accuracy:.4f}")
    print(f"CV Precision: {cv_precision:.4f}")
    print(f"CV Recall: {cv_recall:.4f}")
    print(f"CV F1-Score: {cv_f1:.4f}")
    
    # <-- FIX: This is the CV confusion matrix
    cv_confusion = confusion_matrix(Y_train, cross_validated_pred) 
    print("CV Confusion Matrix:")
    print(cv_confusion)


    # --- 3. TRAINING FINAL MODEL on 80% TRAINING SET ---
    print("\nTraining the final model on all training data...")
    text_classification_pipeline.fit(X_train, Y_train)

    # Get class names
    feature_names = text_classification_pipeline.named_steps['clf'].classes_


    # --- 4. TESTING FINAL MODEL on 20% TEST SET ---
    print("Testing the final model on unseen test data...")
    final_predictions = text_classification_pipeline.predict(X_test)
    
    # <-- FIX: Typo was 'presicion'
    final_precision, final_recall, final_f1, _ = precision_recall_fscore_support(Y_test, final_predictions, average='weighted') 
    
    # <-- FIX: Typo was 'accuracy_scores' (plural)
    final_accuracy = accuracy_score(Y_test, final_predictions) 

    print("\n--- FINAL MODEL TEST RESULTS ---")
    
    # <-- FIX: Corrected the print labels
    print(f"Final Accuracy: {final_accuracy:.4f}")
    print(f"Final Precision: {final_precision:.4f}")
    print(f"Final Recall: {final_recall:.4f}")
    print(f"Final F1-Score: {final_f1:.4f}")
    
    # <-- FIX: This is the FINAL confusion matrix
    final_confusion = confusion_matrix(Y_test, final_predictions)
    
    print(f"\nClass names (in order): {feature_names}")
    print("Final Confusion Matrix:")
    print(final_confusion)


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
if __name__ == "__main__":
    main()
