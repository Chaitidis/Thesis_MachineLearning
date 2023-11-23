from sklearn.model_selection import GridSearchCV
from sklearn import svm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import random
from mpl_toolkits.mplot3d import Axes3D


    # ! SVM's FOR 10 MODELS


# Initialize variables to store metrics for each shuffle
training_errors = []
model_scores=[]
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
auc_values = []
   
num_shuffles=10
for shuffle in range(num_shuffles):

    # train_data = pd.read_csv(f'training_MODIFIED_data_shuffle_{shuffle}.csv')
    # test_data = pd.read_csv(f'testing_MODIFIED_data_shuffle_{shuffle}.csv')
    
    Indexes = pd.read_csv(f'testing_MODIFIED_data_shuffle_{shuffle}.csv')
    train_data = pd.read_csv(f'DECISIONTREEDATASPECIALEDITION_{shuffle}.csv')
    test_data = pd.read_csv(f'DECISIONTREEDATASPECIALEDITION_TESTING_{shuffle}.csv')
    ageaccuratetra = pd.read_csv(f'training_data_shuffle_{shuffle}.csv')
    ageaccuratetest = pd.read_csv(f'testing_data_shuffle_{shuffle}.csv')

    # Drop the 'Index' column
    train_data = train_data.drop(columns=['Index'])
    test_data = test_data.drop(columns=['Index'])

    # Extract the 'AGE' column for scaling
    train_data['AGE'] = ageaccuratetra['AGE']
    test_data['AGE'] = ageaccuratetest['AGE']

    # Initialize a MinMaxScaler
    scaler = MinMaxScaler()
    train_data['AGE'] = scaler.fit_transform(train_data[['AGE']])
    test_data['AGE'] = scaler.fit_transform(test_data[['AGE']])

    # # Fit and transform the 'AGE' column for both training and testing data
    # train_data['AGE'] = scaler.fit_transform(train_data['AGE'])
    # test_data['AGE'] = scaler.transform(test_data['AGE'])  # Use transform for the testing set

    # # Now, scale the entire dataframes, excluding the 'AGE' column
    # columns_to_scale = [col for col in train_data.columns]
    # train_data[columns_to_scale] = scaler.fit_transform(train_data[columns_to_scale])
    # test_data[columns_to_scale] = scaler.fit_transform(test_data[columns_to_scale])

    # print(train_data.head())
    # print(test_data)

    X_train = train_data.drop(columns=['RESULT', 'AREA'])  # Adjust 'target' to your actual target column name


    y_train = train_data['RESULT']

    # Extract the features (X) and the target (y) for the testing set
    X_test = test_data.drop(columns=['RESULT','AREA'])    # Adjust 'target' to your actual target column name
    # X_test = test_data.drop(columns=['RESULT'])    # Adjust 'target' to your actual target column name

    y_test = test_data['RESULT']

    
    
    
    
    
    
    # Choose the number of components you want to retain
    # n_components = 2  # decent results AVG ACC= 0.8133 , visualization decent semi
    # n_components = 3  # decent results AVG ACC= 0.7933 , visualization decent semi
    # n_components = 4  # decent results AVG ACC= 0.7867 , visualization decent. AVG ACC=0.8163 with normalized age
    # n_components = 5  # decent results AVG ACC= 0.7467 , visualization decent AVG ACC=0.8067 with normalized age
    # n_components = 6  # decent results AVG ACC= 0.7267 , visualization semi AVG ACC=0.8200 with normalized age
    # n_components = 7  # decent results AVG ACC= 0.7400 , visualization semi
    # n_components = 8  # decent results AVG ACC= 0.7533 , visualization decent
    # n_components = 9  # decent results AVG ACC= 0.7467 , visualization good AVG ACC=0.8133 with normalized age
    # n_components = 10  # decent results AVG ACC= 0.7800 , visualization semi AVG ACC=0.8067 with normalized age
    # n_components = 11  # decent results AVG ACC= 0.7533 , visualization semi
    # n_components = 12  # decent results AVG ACC= 0.8000 , visualization decent
    # n_components = 13  # decent results AVG ACC= 0.7200 , visualization decent
    # n_components = 14  # decent results AVG ACC= 0.7667 , visualization decent
    # n_components = 15  # decent results AVG ACC= 0.7733 , visualization semi AVG ACC=0.7800 with normalized age
    # n_components = 16  # decent results AVG ACC= 0.7533 , visualization decent
    # n_components = 17  # decent results AVG ACC= 0.7533 , visualization semi
    # n_components = 18  # decent results AVG ACC= 0.7400 , visualization semi
    # n_components = 19  # decent results AVG ACC= 0.7400 , visualization semi
    # n_components = 20  # decent results AVG ACC= 0.7400 
    # n_components = 21  # decent results AVG ACC= 0.7400 , visualization semi

    
    # # Create a PCA instance and fit it to your training data
    # pca = PCA(n_components=n_components)
    # X_train = pca.fit_transform(X_train)  # X_train is your training feature data
    # X_test = pca.transform(X_test)  # X_test is your testing feature data





    random_seed=42
    random_seed=42
    np.random.seed(15)
    random.seed(15)
    tf.random.set_seed(15)

    # random_seed=11
    
    # ? GRIDSEARCH WITH SVM'S 
    print("\n")
    param_grid = [
        # {
        #     'kernel': ['poly'],
        #     # 'C': [0.001],
        #     'C': [0.5,  1,1.5,  2, 2.5, 3, 4, 5, 6, 8, 9, 10, 11], # BEST MODEL
        #     # 'C': [0.001, 0.01, 0.05, 0.1, 0.5,  1,1.5,  2, 2.5, 3, 4, 5, 6, 6.5, 7, 7.5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 25],
        #     # 'C': [0.001, 0.01, 0.05, 0.1,0.5,  1,1.5,  2, 2.5, 3, 4, 5, 6, 6.5, 7, 7.5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 25],
        #     # 'gamma': [0.00001]
        # },
        # {
        #     # 'kernel': ['sigmoid', 'linear'],
        #     'kernel': ['linear'],
        #     # 'C': [0.001, 0.01, 0.05, 0.1, 0.5,  1,1.5,  2, 2.5, 3, 4, 5, 6, 6.5, 7, 7.5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 25],
        #     # 'C': [5, 6, 6.5, 7, 7.5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 25],
        #     'C': [0.001, 0.01, 0.05, 0.1], # BEST MODEL
        # },
        # {
        #     'kernel': ['rbf'],
        #     # 'C': [0.001, 0.01, 0.05, 0.1, 1, 2, 2.5, 3, 4, 5, 6, 6.5, 7, 7.5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 25, 26,30 ,40, 100],
        #     # 'C': [0.001, 0.01, 0.05, 0.1],
        #     'C': [ 1,1.5,  2, 2.5, 3, 3.5  , 4 , 4.5 , 5, 6], # BEST MODEL

        #     # 'gamma': [ 0.001, 0.004, 0.005, 0.006, 0.01, 0.015, 0.02, 0.025] # Great results with 'C': [ 1,1.5,  2, 2.5, 3, 4, 5, 6], # good results
        #     'gamma': [ 0.001, 0.004, 0.005, 0.006, 0.01, 0.015, 0.02, 0.025] # BEST MODEL
        # }
    ]


    # Create an SVM classifier
    svm_classifier = svm.SVC(random_state=random_seed)

    # Create a grid search object
    grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=4, scoring='accuracy')

    
    
    
    # Fit the grid search to your training data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    # best_score = grid_search.best_score_
    print(best_params)
    # Create an SVM classifier with the best hyperparameters
    best_svm_classifier = svm.SVC(**best_params, probability=True, random_state=random_seed)

    # Train the model with the best hyperparameters on your training data
    best_svm_classifier.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = best_svm_classifier.predict(X_test)

  
    y_train_pred = best_svm_classifier.predict(X_train)  
    training_error = 1 - accuracy_score(y_train, y_train_pred)
    

    # Calculate the model name
    model_name = f'Model_{shuffle}'
   
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
    y_pred_prob = best_svm_classifier.predict_proba(X_test)  # Predict class probabilities

    print("Model ", shuffle, "--> Accuracy : " ,accuracy)
    print(15-accuracy*15, "/15 : misclassified " )
    print( "Training Error : "   ,training_error)

    #                                  CREATING THE DATA PROBABILITIES FOR EACH MODEL                                  #
    
    
    df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    # Assuming y_pred_prob is a list of lists where each inner list contains probabilities for class 0 and class 1
    # Extract the probabilities for class 0 and class 1
    # class_0_probs = [prob[0] for prob in y_pred_prob]
    # class_1_probs = [prob[1] for prob in y_pred_prob]
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
    result_df.to_csv(f'MultinomialNB_MODEL_PROBS_{shuffle}.csv', index=False)
    
    
    #                                  CREATING THE DATA PROBABILITIES FOR EACH MODEL                                  #




    
    
    
    # # This code will show the decision boundary based on the model's predictions on the testing set, helping you visualize how the model performs on unseen data.
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X_test)  # Use the testing set X_test
    # best_svm_classifier.fit(X_pca, y_test)  # Fit the model on the testing set

    # x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    # y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    # # Z = best_svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    # # Z = Z.reshape(xx.shape)
    # if Z.shape[0] - xx.shape[0] > 0 or Z.shape[1] - xx.shape[1] > 0:
    #     row_padding = max(0, abs(Z.shape[0] - xx.shape[0]))
    #     col_padding = max(0, abs(Z.shape[1] - xx.shape[1]))
    #     # Z = np.pad(Z, ((0, row_padding), (0, col_padding)), mode='constant')
    #     xx = np.pad(xx, ((0, row_padding), (0, col_padding)), mode='constant')
    #     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))



    # if Z.shape[0] - yy.shape[0] > 0 or Z.shape[1] - yy.shape[1] > 0:
    #     row_padding = max(0, abs(Z.shape[0] - xx.shape[0]))
    #     col_padding = max(0, abs(Z.shape[1] - xx.shape[1]))
    #     # Z = np.pad(Z, ((0, row_padding), (0, col_padding)), mode='constant')
    #     yy = np.pad(yy, ((0, row_padding), (0, col_padding)), mode='constant')
      
    
    # row_padding = max(0, abs(Z.shape[0] - xx.shape[0]))
    # col_padding = max(0, abs(Z.shape[1] - xx.shape[1]))
    # xx = np.pad(xx, ((0, row_padding), (0, col_padding)), mode='constant')

    # row_padding = max(0, abs(Z.shape[0] - yy.shape[0]))
    # col_padding = max(0, abs(Z.shape[1] - yy.shape[1]))
    # yy = np.pad(yy, ((0, row_padding), (0, col_padding)), mode='constant')


    # # Pad x with zeros to match the shape of z

    # plt.contourf(xx+10, yy+10, Z, alpha=1)
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, edgecolors='k', cmap=plt.cm.Paired)
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.show()
#     # # This code will show the decision boundary based on the model's predictions on the testing set, helping you visualize how the model performs on unseen data.
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X_train)  # Use the testing set X_test
#     best_svm_classifier.fit(X_pca, y_train)  # Fit the model on the testing set

#     # Create a meshgrid that covers your feature space
#     x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
#     y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

#     # Use your trained SVM model to predict the class labels for these points
#     Z = best_svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)

#     # Plot the decision boundary
#     plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

#     # Plot the data points
#     plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')

#     # Add labels and title
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.title('SVM Decision Boundary on Training Set')

#     # Show the plot
#     plt.show()
    

# #  #! PLOTS 
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X_test)  # Use the testing set X_test
#     best_svm_classifier.fit(X_pca, y_test)  # Fit the model on the testing set

#     x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
#     y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
#     Z = best_svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     plt.contourf(xx, yy, Z, alpha=0.8)
    
    
#     y_test_pred = best_svm_classifier.predict(X_pca)  # Predict based on the original X_test data

#     misclassified_indices = np.where(y_test != y_test_pred)
#     plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, edgecolors='k', cmap=plt.cm.coolwarm)
#     plt.scatter(X_pca[misclassified_indices, 0], X_pca[misclassified_indices, 1], c='red', marker='x', s=100, label='Misclassified')
#     kernel_type = best_params['kernel']

#     # Add this line after plotting your data
#     plt.title(f'SVM Decision Boundary on Testing Set with {kernel_type} Kernel')

#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.show()    
#      #! PLOTS 


    
    
    
    # # ! 3D PRINTING SEMIIIIIIII
    # pca = PCA(n_components=3)
    # X_pca = pca.fit_transform(X_test)  # Use the testing set X_test
    # best_svm_classifier.fit(X_pca, y_test)  # Fit the model on the testing set

    # n_components = 3

    # n_components = 2

    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X_test)  # Use the testing set X_test
    # best_svm_classifier.fit(X_pca, y_test)  # Fit the model on the testing set


    # # Fit the model with the first two principal components
    # best_svm_classifier.fit(X_pca, y_test)

    # # Create a meshgrid that covers your feature space in the two principal components
    # x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    # y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # # Use your trained SVM model to predict the class labels for these points
    # Z = best_svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot the decision boundary in 3D
    # ax.contourf(xx, yy, Z, zdir='z', offset=-2, cmap=plt.cm.coolwarm, alpha=0.8)

    # # Plot the data points
    # ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 0], c=y_test, cmap=plt.cm.coolwarm, edgecolors='k')

    # # Add labels and title
    # ax.set_xlabel('Feature 1')
    # ax.set_ylabel('Feature 2')
    # ax.set_zlabel('Feature 3')
    # ax.set_title('3D Decision Boundary Plot')

    # # Show the 3D plot
    # plt.show()
    
    
    
    
    
    
    #  # ! semiiii 3D PRINTING WWEEWW
    # pca = PCA(n_components=3)
    # X_pca = pca.fit_transform(X_test)

    # support_vectors = best_svm_classifier.support_vectors_

    # # Extract the coefficients (weights) of the support vectors
    # weights = best_svm_classifier.dual_coef_

    # # Calculate the intercept (bias) term
    # intercept = best_svm_classifier.intercept_

    # # Define the equation of the separating plane in 3D
    # z = lambda x, y: (-intercept[0] - weights[0, 0] * x - weights[0, 1] * y) / weights[0, 2]

    # # Create a meshgrid for 3D plotting
    # tmp = np.linspace(-5, 5, 30)
    # x, y = np.meshgrid(tmp, tmp)
    # z_values = z(x, y)

    # # Create a 3D figure
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot the data points
    # ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y_test, cmap=plt.cm.coolwarm, edgecolors='k')

    # # Plot the decision boundary as a 3D surface
    # ax.plot_surface(x, y, z_values, alpha=0.5, cmap=plt.cm.coolwarm)

    # # Set labels and title
    # ax.set_xlabel('Principal Component 1')
    # ax.set_ylabel('Principal Component 2')
    # ax.set_zlabel('Principal Component 3')
    # ax.set_title('3D Decision Boundary Plot')

    # # Show the 3D plot
    # plt.show()
        # Initialize a list to store AUC values for all models
 


    y_pred = best_svm_classifier.predict(X_test)
        
    # Calculate ROC and AUC for the current model
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
        
    # Append the AUC value to the list
    auc_values.append(roc_auc)

# Plot the ROC curve for the last model in the list (or a specific model)
print("AUC VALUES: ", auc_values)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
# Calculate and plot the overall AUC score
overall_auc = sum(auc_values) / len(auc_values)
print(f'Overall AUC: {overall_auc:.2f}')

    
    
    
    
    

    


    
    
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
columns_to_round = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Training Error']  # Replace with your column names
# Round only the specified columns
metrics_df[columns_to_round] = metrics_df[columns_to_round].apply(lambda x: round(x, 4))
# Save the metrics to a CSV file for future reference
metrics_df.to_csv('shuffle_metrics_SVM.csv')
