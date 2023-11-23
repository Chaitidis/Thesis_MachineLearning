from sklearn.model_selection import train_test_split, GridSearchCV,learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,explained_variance_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc


    # ! DECISION TREE MODEL FOR 10 MODELS


# Initialize variables to store metrics for each shuffle
accuracy_scores = []
model_scores=[]
training_errors = []
precision_scores = []
recall_scores = []
f1_scores = []
accuracies=[]
auc_values=[]

num_shuffles=10
for shuffle in range(num_shuffles):

    
    
    Indexes = pd.read_csv(f'testing_MODIFIED_data_shuffle_{shuffle}.csv')

    # train_data = pd.read_csv(f'training_MODIFIED_data_shuffle_{shuffle}.csv')
    # test_data = pd.read_csv(f'testing_MODIFIED_data_shuffle_{shuffle}.csv')
    train_data = pd.read_csv(f'DECISIONTREEDATASPECIALEDITION_{shuffle}.csv')
    test_data = pd.read_csv(f'DECISIONTREEDATASPECIALEDITION_TESTING_{shuffle}.csv')
    ageaccuratetra=pd.read_csv(f'training_data_shuffle_{shuffle}.csv')
    ageaccuratetest=pd.read_csv(f'testing_data_shuffle_{shuffle}.csv')

    train_data['AGE']=ageaccuratetra['AGE']
    test_data['AGE']=ageaccuratetest['AGE']

    train_data = train_data.drop(columns=['Index'])  # Adjust 'target' to your actual target column name
    test_data = test_data.drop(columns=['Index'])  # Adjust 'target' to your actual target column name

    # # This is done to drop the indexing column from Alex's specific split of dataframe
    # train_data = train_data.iloc[:, 1:]  # Drop the first column
    # test_data = test_data.iloc[:, 1:]  # Drop the first column

    # Extract the features (X) and the target (y) for the training set
    # X_train = train_data.drop(columns=['RESULT'])  # Adjust 'target' to your actual target column name
    # X_train = train_data.drop(columns=['RESULT'])  # Adjust 'target' to your actual target column name
    # X_train = train_data.drop(columns=['RESULT', 'SEX','ETHNICITY', 'VIOLATION','SEARCH' ])  # Adjust 'target' to your actual target column name
    X_train = train_data.drop(columns=['RESULT','AREA'])  # Adjust 'target' to your actual target column name


    y_train = train_data['RESULT']

    # Extract the features (X) and the target (y) for the testing set
    X_test = test_data.drop(columns=['RESULT','AREA'])    # Adjust 'target' to your actual target column name
    # X_test = test_data.drop(columns=['RESULT', 'SEX','ETHNICITY', 'VIOLATION','SEARCH'])    # Adjust 'target' to your actual target column name
    # X_test = test_data.drop(columns=['RESULT', 'AREA','VIOLATION', 'CONDITION'])    # Adjust 'target' to your actual target column name


    y_test = test_data['RESULT']
    # print(X_train.head(5))
    # print(X_train.shape)

    random_seed=42
    # random_seed=11

    # Create a Decision Tree classifier
    tree_classifier = DecisionTreeClassifier(random_state=random_seed)
    # Define a grid of hyperparameters to search
    param_grid = {
        'criterion': ['gini'],    # Split criterion (measures of impurity)
        # 'max_depth': [None, 5, 10, 15],     # Maximum depth of the tree
        # 'max_depth': [None,1],     # Maximum depth of the tree
        'max_depth': [None,1],     # Maximum depth of the tree

        # 'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15],    # Minimum samples required to split an internal node
        # 'min_samples_split': [2],    # Minimum samples required to split an internal node
        'min_samples_split': [2, 3, 4, 5, 6],    # Minimum samples required to split an internal node

        # 'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 10],       # Minimum samples required to be in a leaf node
        # 'min_samples_leaf': [1],       # Minimum samples required to be in a leaf node
        'min_samples_leaf': [1, 2, 3],       # Minimum samples required to be in a leaf node

        # 'max_features': [ 2, 3, 4, 5, 6, 7, 8, 9]
        'max_features': [2,3]

    }

    # Create a grid search with cross-validation
    grid_search = GridSearchCV(tree_classifier, param_grid, cv=4, scoring='accuracy')

    # Fit the grid search to your data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)
    # Best Hyperparameters: {'criterion': 'gini', 'max_depth': None, 'max_features': 2, 'min_samples_leaf': 4, 'min_samples_split': 5}
    # Best Hyperparameters: {'criterion': 'gini', 'max_depth': None, 'max_features': 2, 'min_samples_leaf': 1, 'min_samples_split': 5}

    # Use the best model for prediction
    best_tree_classifier = grid_search.best_estimator_
    y_pred = best_tree_classifier.predict(X_test)

    
    y_train_pred = best_tree_classifier.predict(X_train)  # Assuming l1_model is your model
    training_error = 1 - accuracy_score(y_train, y_train_pred)
     # Calculate the model name
    model_name = f'Model_{shuffle}'
    
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Store metrics in the respective lists
    # model_name = f'Model_{shuffle}'
    model_scores.append(f'Model_{shuffle}')
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    training_errors.append(training_error)
    y_pred_prob = best_tree_classifier.predict_proba(X_test)  # Predict class probabilities

    print("Model ", shuffle, "--> Accuracy : " ,accuracy)

    
    
        #                                  CREATING THE DATA PROBABILITIES FOR EACH MODEL                                  #
    
    
    df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    # Assuming y_pred_prob is a list of lists where each inner list contains probabilities for class 0 and class 1
    # Extract the probabilities for class 0 and class 1
    class_0_probs = [prob[0] for prob in y_pred_prob]
    class_1_probs = [prob[1] for prob in y_pred_prob]
    # Reset the index for the DataFrame
    # Create a new DataFrame for the probabilities
    probs = pd.DataFrame(y_pred_prob, columns=['Class_0_Prob', 'Class_1_Prob'])
    def format_float(x): # Function to format the probabilities.
        return "{:.2f}%".format(x * 100)  # Two decimal places for the percentage %

    # Apply the custom function to the entire column
    probs['Class_0_Prob'] = probs['Class_0_Prob'].apply(format_float)
    probs['Class_1_Prob'] = probs['Class_1_Prob'].apply(format_float)
    # Concatenate the original DataFrame and the probabilities DataFrame and the Index Dataframe
    result_df = pd.concat([df, probs], axis=1)
    result_df = result_df.reset_index(drop=True)
    result_df['Index'] = Indexes['Index']
    last_column = result_df.pop(df.columns[-1])
    result_df.insert(0, last_column.name, last_column)
    result_df['Classified']= result_df['y_test']==result_df['y_pred']
    result_df.to_csv(f'DecisionTree_MODEL_PROBS_{shuffle}.csv', index=False)
    
    
    #                                  CREATING THE DATA PROBABILITIES FOR EACH MODEL                                  #


    #                                  CREATING FEATURE IMPORTANCES                                  #       
                               
    importances = best_tree_classifier.feature_importances_
    # print("Feature importance: ",importances)
    
        # Create a DataFrame to store importances, one column per dataset
    col_name = f'Dataset_{shuffle}'
    if shuffle == 0:
        importance_df = pd.DataFrame()
        importance_df[col_name] = importances
    else:
        importance_df[col_name] = importances

    # Calculate statistics (e.g., mean and standard deviation)
    # mean_importance = importance_df.mean(axis=1)
    # print("mean_importance: ",mean_importance)
    importance_df.to_csv('importances_DTC_df.csv')
    
    #                                  CREATING FEATURE IMPORTANCES                                  #                                  
    accuracies.append(accuracy)
    




    # # Visualize the Decision Tree. Plotting the Decison tree
    # plt.figure(figsize=(12, 8))
    # plot_tree(best_tree_classifier, feature_names=X_train.columns, filled=True, rounded=True, class_names=True)
    # # Define the text
    # text = "Decision Tree Classifier Data Modified"
    # # Add the text to the top-right corner
    # plt.text(0.25, 0.95, text, fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right')

    # plt.show()
    
    
    
    
    # average_importances = importances
    # feature_names=X_train.columns
    # # Sort the features and importances by importance in descending order
    # sorted_features = X_train.columns[np.argsort(average_importances)[::-1]]
    # sorted_importances = average_importances[np.argsort(average_importances)[::-1]]

    # print(X_train.columns) 
    # print(average_importances)
    # Create a bar plot to visualize the average feature importances
#    plt.barh(range(len(average_importances)), average_importances)
#    plt.yticks(range(len(feature_names)), feature_names)
#    plt.xlabel('Average Feature Importance 10 Models (DTC)')

#      plt.show()


##!                                             AUC OVERAL CURVE AND VALUE                                              !##               
#     y_pred = best_tree_classifier.predict(X_test)
        
#     # Calculate ROC and AUC for the current model
#     fpr, tpr, thresholds = roc_curve(y_test, y_pred)
#     roc_auc = auc(fpr, tpr)
        
#     # Append the AUC value to the list
#     auc_values.append(roc_auc)

# # Plot the ROC curve for the last model in the list (or a specific model)
# print("AUC VALUES: ", auc_values)
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc='lower right')
# plt.show()
# # Calculate and plot the overall AUC score
# overall_auc = sum(auc_values) / len(auc_values)
# print(f'Overall AUC: {overall_auc:.2f}')
    
# END OF LOOP
    
    
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
# Specify which columns to round (numeric columns)
columns_to_round = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Training Error']  # Replace with your column names

# Round only the specified columns
metrics_df[columns_to_round] = metrics_df[columns_to_round].apply(lambda x: round(x, 4))
metrics_df.to_csv('shuffle_metrics_DecisionTree.csv')

# total_acc=sum(accuracies)
# weights = [acc / total_acc for acc in accuracies]
# Average_importances=importance_df.div(10)
importance_df.to_csv('importance_df_AVERAGE_DTC.csv')

# weighted_importances = importance_df.mul(accuracies, axis=1)
# weighted_importances[columns_to_round] = weighted_importances[columns_to_round].apply(lambda x: round(x, 4))

# weighted_importances.to_csv('importance_df_weighted_DTC.csv')
