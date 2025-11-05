
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
    return information_list
