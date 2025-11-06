from sklearn.model_selection import train_test_split
import numpy
import csv
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, matthews_corrcoef
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

def getMetrics(confusion_matrix, classNames):
    metrics = {}

    FP = confusion_matrix.sum(axis=0) - numpy.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - numpy.diag(confusion_matrix)
    TP = numpy.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP+FN+TP)

    for i, classNames in enumerate(classNames):
        sensitivity = TP[i] / (TP[i] + FN[i]) if (TP[i] + FN[i]) > 0 else 0

        specificity = TN[i] / (TN[i] + FP[i]) if (TN[i] + FP[i]) > 0 else 0
        accuracy = (TP[i] + TN[i]) / confusion_matrix.sum()

        denom = numpy.sqrt((TP[i] + FP[i]) * (TP[i] + FN[i]) * (TN[i] + FP[i]) * (TN[i] + FN[i]))
        mcc = ((TP[i] * TN[i]) - (FP[i] * FN[i])) / denom if denom > 0 else 0
        metrics[classNames] = {
                "Sensitivity": sensitivity,
                "Specificity": specificity,
                "Accuracy": accuracy,
                "MCC": mcc
                }
    return metrics
def wrapperSGDCClassifier():
    nLower = 2
    nUpper = 8
    lowerFeature = 15000
    upperFeature = 20000
    increaseBy = 2000
    TestingOptions = False
    print("--------------SGDC Classifier-----------------")
    if TestingOptions:

        for n in range (nLower, nUpper):
            print("**********New nLower and nUpper************")
            for max_feat in range (lowerFeature, upperFeature, increaseBy):
                runSClassifier(nLower, n, max_feat)
    else:
        # current 
        runSClassifier(2,7,19000)

def runSClassifier(nrange_lower, nrange_upper, feature_limit):
    

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

    
    # --- 1. CROSS-VALIDATION on 80% TRAINING SET ---
    print("Running 5-fold cross-validation on training data...")
    crossFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cross_validated_pred = cross_val_predict(text_classification_pipeline, X_train, Y_train, cv=crossFold, n_jobs=-1)

    # --- 2. GET CV SCORES ---
    cv_accuracy = accuracy_score(Y_train, cross_validated_pred)
    cv_precision, cv_recall, cv_f1, _ = precision_recall_fscore_support(Y_train, cross_validated_pred, average='weighted', zero_division=0)

    cv_mcc = matthews_corrcoef(Y_train, cross_validated_pred)
    print(f"CV MCC score (Estimate) {cv_mcc:.4f}")
    
    # <-- FIX: Append the summary scores, not the thousands of predictions
    
    print("\n--- CROSS-VALIDATION RESULTS (ESTIMATE) ---")
    print(f"CV Accuracy: {cv_accuracy:.4f}")
    print(f"CV Precision: {cv_precision:.4f}")
    print(f"CV Recall: {cv_recall:.4f}")
    print(f"CV F1-Score: {cv_f1:.4f}")
    
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
    
    final_precision, final_recall, final_f1, _ = precision_recall_fscore_support(Y_test, final_predictions, average='weighted') 
    final_mcc = matthews_corrcoef(Y_test, final_predictions)
    final_accuracy = accuracy_score(Y_test, final_predictions) 

    print("\n--- FINAL MODEL TEST RESULTS ---")
    
    # <-- FIX: Corrected the print labels
    print(f"Final Accuracy: {final_accuracy:.4f}")
    print(f"Final Precision: {final_precision:.4f}")
    print(f"Final Recall: {final_recall:.4f}")
    print(f"Final F1-Score: {final_f1:.4f}")
    print(f"Final MCC: {final_mcc:.4f}")
    
    # <-- FIX: This is the FINAL confusion matrix
    final_confusion = confusion_matrix(Y_test, final_predictions)
    
    print(f"\nClass names (in order): {feature_names}")
    print("Final Confusion Matrix:")
    print(final_confusion)

    cv_class_metrics = getMetrics(final_confusion, feature_names)
    print("\n Per class metrics")
    for class_name, metrics in cv_class_metrics.items():
        print(f"\nClass: {class_name}")
        for metric, value in metrics.items():
            print(f" {metric}: {value:.4f}")

    end = datetime.now()
    elapsed = end - start
    elapsed_time = str(elapsed).split(".")[0]
    print(f"Elapsed Time {elapsed_time}")


def main():

    #runTFIDDecisionTree()
    #runTFIDLogisticRegression()
    #wrapperTFIDRandomForest()
    wrapperSGDCClassifier()
if __name__ == "__main__":
    main()
