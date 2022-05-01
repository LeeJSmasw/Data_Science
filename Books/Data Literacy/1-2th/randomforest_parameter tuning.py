trial = 3



from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
import Orange.regression
import Orange.evaluation
import copy
import random
import numpy as np

data = copy.copy(in_data)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 5, stop = 200, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
#


best_model = None
best_MAE = None
best_parameter = None

for i in range(trial) : 
    __n_estimators = random.choice(n_estimators)
    __max_features = random.choice(max_features)
    __max_depth = random.choice(max_depth)
    __min_samples_split = random.choice(min_samples_split)
    __min_samples_leaf = random.choice(min_samples_leaf)
    __bootstrap = random.choice(bootstrap)
    
    print(__n_estimators, __max_features, __max_depth, __min_samples_split, __min_samples_leaf, __bootstrap)
    
    RF = Orange.regression.RandomForestRegressionLearner(n_estimators=__n_estimators,
                                                     max_features=__max_features,
                                                     max_depth=__max_depth,
                                                     min_samples_split=__min_samples_split,
                                                     min_samples_leaf=__min_samples_leaf,
                                                     bootstrap=__bootstrap)
    
    model = RF(data)
    cv_result = Orange.evaluation.testing.CrossValidation(data=data, learners=[RF])
    mae = Orange.evaluation.scoring.MAE(cv_result)

    print(mae)
    
    if best_MAE == None or best_MAE > mae[0] :
        best_MAE = mae[0]
        best_model = model
        best_parameter = {
            "n_estimators": __n_estimators,
            "max_features": __max_features,
            "max_depth": __max_depth,
            "min_samples_split": __min_samples_split,
            "min_samples_leaf": __min_samples_leaf,
            "bootstrap": __bootstrap
        }
        
print("BEST")

print(best_MAE)
print(best_parameter)

out_classifier=best_model

