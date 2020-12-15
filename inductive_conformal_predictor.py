def inductive_conformal_predictor(training_proper_X, training_proper_y, test_X, test_y, significance_level):
    
    my_regressor = Lasso(alpha=0.001,max_iter=1000).fit(training_proper_X, training_proper_y)
    alphas = []
    preds = []
    for i in range(len(test_X)):
        pred = my_regressor.predict([test_X[i]])
        preds.append(pred[0])
        alpha = abs(test_y[i] - pred)
        alphas.append(alpha)
    
    alphas = sorted(alphas)
    K = math.ceil((1-significance_level) * (len(test_X)+1))
    c = alphas[K-1]
    
    prediction_set = []
    
    for pred in preds:
        interval = [pred-c, pred+c]
        prediction_set.append(interval)
    
    return prediction_set
        