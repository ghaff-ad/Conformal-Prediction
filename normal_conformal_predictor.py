def my_conformal_predictor(training_samples, training_labels, samples, labels):
    
    labeled_samples = []
    for sample, label in zip(training_samples, training_labels):
        l_sample = np.append(sample, label)
        labeled_samples.append(l_sample)
    
    labels = set(labels) 
    

    #Pre-processing training set and storing euclidian distance calculations for later use to reduce computational complexity
    eucl_calcs = [] #eucl_calcs = [[same_class, diff_clas],[],[],[],[]]
     
    for i in range(len(labeled_samples)):
        eucl_calcs.append([float('inf'),float('inf')])
        
        new_labeled_samples = labeled_samples[:i] + labeled_samples[i+1:]
       
        for labeled_sample in new_labeled_samples:
            j = 0
            eucl = 0
            while j < len(labeled_sample) - 1:
                eucl += (labeled_samples[i][j] - labeled_sample[j])**2
                j += 1
                        
            eucl = eucl ** 0.5
                    
            if labeled_samples[i][-1] == labeled_sample[-1] and eucl < eucl_calcs[i][0]:
                eucl_calcs[i][0] = eucl
            elif labeled_samples[i][-1] != labeled_sample[-1] and eucl < eucl_calcs[i][1]:
                eucl_calcs[i][1] = eucl
     
    #Processing Samples
    p_values_table = []           
    for sample in samples:
        p_values = []
        for label in labels:
            postulated = np.append(sample,label) 
            new_eucl_calcs = copy.deepcopy(eucl_calcs)
            new_eucl_calcs.append([float('inf'),float('inf')])
           
            #euclidian distance calculation
            for i in range(len(labeled_samples)):
                j = 0
                eucl = 0
                while j < len(labeled_samples[i]) - 1:
                    eucl += (labeled_samples[i][j] - postulated[j])**2
                    j += 1    
                eucl = eucl ** 0.5
                
                #This checks to see if the euclidian distance computed is less then the minimum euclidian distance computed so--
                #--far for the test sample for both same class and different class and updates the minimum euclidian distances--
                #--computed for the test sample so far with this newly computed euclidian distance if true
                if labeled_samples[i][-1] == postulated[-1] and eucl < new_eucl_calcs[-1][0]:
                     new_eucl_calcs[-1][0] = eucl 
                elif labeled_samples[i][-1] != postulated[-1] and eucl < new_eucl_calcs[-1][1]:
                    new_eucl_calcs[-1][1] = eucl
                 
                #This checks to see if the euclidian distance computed is less then the existing minimum euclidian distance--
                #-- computed for the training sample for both same class and different class and edits the minimum euclidian--
                #-- distances computed for the training sample with the newly computed euclidian distance if true
                if labeled_samples[i][-1] == postulated[-1] and eucl < new_eucl_calcs[i][0]:
                    new_eucl_calcs[i][0] = eucl
                elif labeled_samples[i][-1] != postulated[-1] and eucl < new_eucl_calcs[i][1]:
                    new_eucl_calcs[i][1] = eucl
                
            #This computes conformity scores based on (nearest sample in different class/nearest sample in same class)
            conformity_scores = []     
            for eucl_arr in new_eucl_calcs:
                if eucl_arr[0]  == 0:
                    conformity_scores.append(float('inf'))
                elif eucl_arr[0] == 0  and eucl_arr[1] == 0:
                    conformity_scores.append(0)
                else:
                    conformity_scores.append(eucl_arr[1]/eucl_arr[0])
            
            #computing the rank of conformity score of the sample in question
            rank = 0
            for score in conformity_scores:
                if conformity_scores[-1] >= score:
                    rank += 1
                    
            p_value = rank/(len(labeled_samples) + 1) #Computing the p-value
            p_values.append(p_value)
        
        p_values_table.append(p_values)
    
    #This computes the average false p-value for each sample and stores it in the avg_p_values list
    copy_p_values_table = p_values_table.copy()
    avg_p_values = []
    for values in  copy_p_values_table:
        values.remove(max(values))
        avg = sum(values)/len(values)
        avg_p_values.append(avg)
        
    avg_false_p_value = sum(avg_p_values)/len(avg_p_values) #computing overall average false p-value
    return avg_false_p_value 