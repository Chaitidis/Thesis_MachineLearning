from sklearn.naive_bayes import BernoulliNB
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.naive_bayes import CategoricalNB, MultinomialNB, GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate




    # ! BernoulliNB FOR 10 MODELS


# Initialize variables to store metrics for each shuffle
training_errors = []
model_scores=[]
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

num_shuffles=10
for shuffle in range(num_shuffles):

    train_data = pd.read_csv(f'training_MODIFIED_data_shuffle_{shuffle}.csv')
    test_data = pd.read_csv(f'testing_MODIFIED_data_shuffle_{shuffle}.csv')
    # train_data = pd.read_csv(f'DECISIONTREEDATASPECIALEDITION_{shuffle}.csv')
    # test_data = pd.read_csv(f'DECISIONTREEDATASPECIALEDITION_TESTING_{shuffle}.csv')
    
    train_data = train_data.drop(columns=['Index'])  # Adjust 'target' to your actual target column name
    test_data = test_data.drop(columns=['Index'])  # Adjust 'target' to your actual target column name
    Indexes = pd.read_csv(f'testing_MODIFIED_data_shuffle_{shuffle}.csv')

    # ageaccuratetra=pd.read_csv(f'training_data_shuffle_{shuffle}.csv')
    # ageaccuratetest=pd.read_csv(f'testing_data_shuffle_{shuffle}.csv')

    # train_data['AGE']=ageaccuratetra['AGE']
    # test_data['AGE']=ageaccuratetest['AGE']


    # Extract the features (X) and the target (y) for the training set
    # X_train = train_data.drop(columns=['RESULT', 'AREA' ])  # Adjust 'target' to your actual target column name
    X_train = train_data.drop(columns=['RESULT'])  # Adjust 'target' to your actual target column name
    # X_train = train_data.drop(columns=['RESULT'])  # Adjust 'target' to your actual target column name

    y_train = train_data['RESULT']

    # Extract the features (X) and the target (y) for the testing set
    # X_test = test_data.drop(columns=['RESULT', 'AREA'])    # Adjust 'target' to your actual target column name
    X_test = test_data.drop(columns=['RESULT'])    # Adjust 'target' to your actual target column name

    y_test = test_data['RESULT']


    random_seed=42
    # random_seed=11
    np.random.seed(random_seed)


    # Define the hyperparameters and their possible values
    param_grid = {
        'var_smoothing': [1e-09, 1e-07, 1e-08, 1e-05, 0.0001, 0.0005, 0.001, 0.004, 0.005, 0.01, 0.02, 0.03, 
                          0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15, 0.2, 0.25, 0.29, 0.3, 
                          0.31, 0.32, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1, 1.1], # Adjust the range of var_smoothing values

    }
    # param_grid = {
        # 'var_smoothing': [4, 5, 6 , 7, 8, 9, 10], # Adjust the range of var_smoothing values
    #     'var_smoothing': [0], # Adjust the range of var_smoothing values

    # }

    # Create a Bernoulli Naive Bayes classifier
    bnb = GaussianNB()

    # Create a custom scorer for GridSearchCV (use the scoring method you prefer)
    custom_scorer = make_scorer(accuracy_score)

    # Create a GridSearchCV instance
    grid_search = GridSearchCV(bnb, param_grid, scoring=custom_scorer, cv=7)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)
    # Get the best model with the optimal hyperparameters
    best_bnb = grid_search.best_estimator_

    # Use the best model to make predictions on the test set
    y_pred = best_bnb.predict(X_test)
    
    
    
    # k_best_selector = SelectKBest(score_func=f_classif, k=10)  # Adjust 'k' to the desired number of features
    # X_train_selected = k_best_selector.fit_transform(X_train, y_train)
    # selected_feature_indices = k_best_selector.get_support(indices=True)
    # X_train_selected, X_test_selected = X_train.iloc[:, selected_feature_indices], X_test.iloc[:, selected_feature_indices]
    # selected_feature_names = X_train.columns[selected_feature_indices]
    # print("Selected Feature Names:")
    # print(selected_feature_names)





    
    # Calculate training error 
    y_train_pred = best_bnb.predict(X_train)  
    training_error = 1 - accuracy_score(y_train, y_train_pred)

    # Calculate the model name
    model_name = f'Model_{shuffle}'

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Shuffle :", shuffle )
    print("Models accuracy:", accuracy)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Store metrics in the respective lists
    model_scores.append(model_name)
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    training_errors.append(training_error)
    y_pred_prob = best_bnb.predict_proba(X_test)  # Predict class probabilities






    #                                  CREATING THE DATA PROBABILITIES FOR EACH MODEL                                  #
    
    
    df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    # Assuming y_pred_prob is a list of lists where each inner list contains probabilities for class 0 and class 1
    # Extract the probabilities for class 0 and class 1
    class_0_probs = [prob[0] for prob in y_pred_prob]
    class_1_probs = [prob[1] for prob in y_pred_prob]
    # Reset the index for the DataFrame
    # Create a new DataFrame for the probabilities
    probs = pd.DataFrame(y_pred_prob, columns=['Class_0_Prob', 'Class_1_Prob'])
    def format_float(x):
        return "{:.2f}%".format(x * 100)  # Assumes two decimal places for the percentage

    # Apply the custom function to the entire column
    probs['Class_0_Prob'] = probs['Class_0_Prob'].apply(format_float)
    probs['Class_1_Prob'] = probs['Class_1_Prob'].apply(format_float)
    # Concatenate the original DataFrame and the probabilities DataFrame
    result_df = pd.concat([df, probs], axis=1)
    result_df = result_df.reset_index(drop=True)
    result_df['Index'] = Indexes['Index']
    last_column = result_df.pop(df.columns[-1])
    result_df.insert(0, last_column.name, last_column)
    result_df['Classified']= result_df['y_test']==result_df['y_pred']
    # result_df.to_csv(f'BernoulliNB_MODEL_PROBS_{shuffle}.csv', index=False)
    result_df.to_csv(f'MultinomialNB_MODEL_PROBS_{shuffle}.csv', index=False)

    
    #                                  CREATING THE DATA PROBABILITIES FOR EACH MODEL                                  #
    
    
    # !                                       FEATURE IMPORTANCE                                    #
    X = X_test
    from sklearn.inspection import permutation_importance

    # Calculate permutation importance
    perm_importance = permutation_importance(best_bnb, X_test, y_test, n_repeats=30, random_state=42)

    # Get the feature importances
    importances = perm_importance.importances_mean

    # # Manually scale the importances to the range [0, 1]
    # importances = (importances - importances.min()) / (importances.max() - importances.min())

    # # Create a bar plot to visualize the scaled feature importances
    # plt.barh(range(X.shape[1]), importances)
    # plt.yticks(range(X.shape[1]), X.columns)
    # plt.xlabel('Scaled Permutation Importance')
    # plt.show()
    # #                                 #    FEATURE IMPORTANCE                                    #
    
        # Create a DataFrame to store importances, one column per dataset
    col_name = f'Dataset_{shuffle}'
    if shuffle == 0:
        importance_df = pd.DataFrame()
        importance_df[col_name] = importances
    else:
        importance_df[col_name] = importances

    # # Calculate statistics (e.g., mean and standard deviation)
    # mean_importance = importance_df.mean(axis=1)
    # std_importance = importance_df.std(axis=1)
    # !                                       FEATURE IMPORTANCE                                    #



    
# END OF LOOP
importance_df.to_csv('importances_BernoulliNB_df.csv')
weighted_importances = importance_df.mul(accuracy_scores, axis=1)
 
# print(total_acc)
# weighted_importances.to_csv('weighted_avg_importances_DTC.csv')
weighted_importances.to_csv('importance_df_weighted_NB.csv')


# Calculate the average of every metric
avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
avg_precision = sum(precision_scores) / len(precision_scores)
avg_recall = sum(recall_scores) / len(recall_scores)
avg_f1 = sum(f1_scores) / len(f1_scores)
avg_training_error = sum(training_errors) / len(training_errors)

# Add the averages to the lists
model_scores.append('Average')
accuracy_scores.append(avg_accuracy)
precision_scores.append(avg_precision)
recall_scores.append(avg_recall)
f1_scores.append(avg_f1)
training_errors.append(avg_training_error) 

# Create a Pandas DataFrame to store and analyze the metrics
metrics_df = pd.DataFrame({
    'Model Name': model_scores,
    'Accuracy': accuracy_scores,
    'Precision': precision_scores,
    'Recall': recall_scores,
    'F1 Score': f1_scores,
    'Training Error': training_errors,

})

# You can also calculate statistics or plot the metrics as needed
# ...
columns_to_round = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Training Error'] 
# Round only the specified columns
metrics_df[columns_to_round] = metrics_df[columns_to_round].apply(lambda x: round(x, 4))

# Save the metrics for reference
metrics_df.to_csv('shuffle_metrics_BernoulliNB.csv')


