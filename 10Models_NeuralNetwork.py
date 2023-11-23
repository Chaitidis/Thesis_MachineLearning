from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc




    # ! NeuralNetowrks MODEL FOR 10 MODELS



# Initialize variables to store metrics for each shuffle
training_errors = []
model_scores = []
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []





num_shuffles = 10
for shuffle in range(num_shuffles):

    # train_data = pd.read_csv(f'training_MODIFIED_data_shuffle_{shuffle}.csv')
    # test_data = pd.read_csv(f'testing_MODIFIED_data_shuffle_{shuffle}.csv')
    
    Indexes = pd.read_csv(f'testing_MODIFIED_data_shuffle_{shuffle}.csv')
    train_data = pd.read_csv(f'DECISIONTREEDATASPECIALEDITION_{shuffle}.csv')
    test_data = pd.read_csv(f'DECISIONTREEDATASPECIALEDITION_TESTING_{shuffle}.csv')
    ageaccuratetra=pd.read_csv(f'training_data_shuffle_{shuffle}.csv')
    ageaccuratetest=pd.read_csv(f'testing_data_shuffle_{shuffle}.csv')
    train_data = train_data.drop(columns=['Index'])  # Adjust 'target' to your actual target column name
    test_data = test_data.drop(columns=['Index'])  # Adjust 'target' to your actual target column name

    train_data['AGE']=ageaccuratetra['AGE']
    test_data['AGE']=ageaccuratetest['AGE']


    # # This is done to drop the indexing column from Alex's specific split of dataframe
    # train_data = train_data.iloc[:, 1:]  # Drop the first column
    # test_data = test_data.iloc[:, 1:]  # Drop the first column

    # Extract the features (X) and the target (y) for the training set
    # X_train = train_data.drop(columns=['RESULT'])  # Adjust 'target' to your actual target column name
    # X_train = train_data.drop(columns=['RESULT'])  # Adjust 'target' to your actual target column name
    # X_train = train_data.drop(columns=['RESULT', 'SEX','ETHNICITY', 'VIOLATION','SEARCH' ])  # Adjust 'target' to your actual target column name
    X_train = train_data.drop(columns=['RESULT', 'AREA'])  # Adjust 'target' to your actual target column name


    y_train = train_data['RESULT']

    # Extract the features (X) and the target (y) for the testing set
    X_test = test_data.drop(columns=['RESULT','AREA'])    # Adjust 'target' to your actual target column name
    # X_test = test_data.drop(columns=['RESULT'])    # Adjust 'target' to your actual target column name

    y_test = test_data['RESULT']

    random_seed=42
    np.random.seed(15)
    random.seed(15)
    tf.random.set_seed(15)

    # Create a neural network model
    model = Sequential()
    # DENSE 23 INPUT 23, DENSE 1 (SIGMOID) AVG ACC=87.3%
    model.add(Dense(23, activation='sigmoid', input_dim=23)) # Input layer
    model.add(Dense(23, activation='sigmoid')) # Sigmoid function for binary classification problems
    # model.add(Dense(12, activation='sigmoid'))
    # model.add(Dense(23, activation='relu'))

    model.add(Dense(1, activation='sigmoid')) # Output 1 

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    # binary_crossentropy is a loss function specifically log loss.It's the error between predicted and actual class. 
    # Well Suited for binary classifications
    # Optimizer=adam, adam is a popular optimization algorithm, responsible for adjusting the weights and biases during
    # Τraining to minimize the lossfunction
    
    # Train the model, Python has built-in backpropagation algorithm that is used during mode.fit(...)
    model.fit(X_train, y_train, epochs=200, batch_size=7) # epochs=92 decent results 
    # model.fit(X_train, y_train, epochs=92, batch_size=1) # epochs=92 decent results



    # Evaluate the model on the testing data
    loss, accuracy = model.evaluate(X_test, y_test)

    # Calculate training error
    y_train_pred = (model.predict(X_train) > 0.5).astype(int)
    training_error = 1 - accuracy_score(y_train, y_train_pred)

    # Calculate the model name
    model_name = f'Model_{shuffle}'
    y_pred_prob = model.predict(X_test)  # Predict class probabilities
    class_1_probs = 1 - y_pred_prob

    # Concatenate the original probabilities and the complementary probabilities
    y_pred_prob = np.concatenate((class_1_probs,y_pred_prob), axis=1)
    y_pred_prob = pd.DataFrame(data=y_pred_prob)

    print(y_pred_prob.head(5))
    # columns = [f'Class_{i}' for i in range(y_pred_prob.shape[1])]
    # predictions_df = pd.DataFrame(data=y_pred_prob, columns=columns)

    # Calculate evaluation metrics on the test data
    # y_pred = model.predict_classes(X_test)

    y_pred = (model.predict(X_test) > 0.5).astype(int)
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

    print("Model ", shuffle, "--> Test Accuracy: ", accuracy)
    # y_pred_probs contains the probabilities for class 1; you can get class 0 probabilities by subtracting from 1
    # y_pred_probs_class0 = 1 - y_pred_prob
    # Create a DataFrame to store the class probabilities
    
        #                                  CREATING THE DATA PROBABILITIES FOR EACH MODEL                                  #
    # print("y_test", y_test.shape) 
    # print("y_pred", y_pred.shape) 
    y_pred = y_pred.squeeze()
    df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    # probs = pd.DataFrame(data=y_pred_prob, columns=columns)


    # ! THIS WORKS TO PLOT THE PREDICTED PROBABILITIES
    def format_float(x):
        if isinstance(x, str):
            return float(x.strip('%')) / 100  # Convert string to float
        elif isinstance(x, float):
            return x  # If it's already a float, leave it as is

    # Apply the custom function to the entire DataFrame
    y_pred_prob = y_pred_prob.applymap(format_float)

    # Create the bar chart
    n_samples, n_classes = y_pred_prob.shape
    x = range(n_samples)

    bottom = [0] * n_samples

    for i in range(n_classes):
        class_probabilities = y_pred_prob.iloc[:, i]
        plt.bar(x, class_probabilities, label=f'Class {i}', bottom=bottom)

        for j, prob in enumerate(class_probabilities):
            plt.text(j, prob / 2 + bottom[j], f'{prob:.2%}', ha='center', va='center', color='white')

        bottom = [sum(x) for x in zip(bottom, class_probabilities)]

    sample_indices = Indexes['Index'].values  # Use this list for labeling the X-axis

    for i in range(n_samples):
        actual = df['y_test'].iloc[i]
        predicted = df['y_pred'].iloc[i]
        print(f"Sample {sample_indices[i]} - Actual: {actual}, Predicted: {predicted}")

    # # plt.show()
    # plt.xticks(x, sample_indices)  # Set the tick labels on the X-axis to be the sample indices
    # plt.xlabel('Sample')
    # plt.ylabel('Probability')
    # plt.title(f'Predicted Probabilities for Each Sample Model {shuffle}')
    # plt.legend()
    # plt.show()
     # ! THIS WORKS TO PLOT THE PREDICTED PROBABILITIES

    
    
    
    
    
    
    
    
    
    # #! TRANSPOSED PROBABILITIES PLOT
    # sample_indices = Indexes['Index'].values  # Use this list for labeling the X-axis

    # def format_float(x):
    #     if isinstance(x, str):
    #         return float(x.strip('%')) / 100  # Convert string to float
    #     elif isinstance(x, float):
    #         return x  # If it's already a float, leave it as is

    # # Apply the custom function to the entire DataFrame
    # y_pred_prob = y_pred_prob.applymap(format_float)

    # # Create the bar chart with inverted axes
    # n_samples, n_classes = y_pred_prob.shape
    # y = range(n_samples)  # Use 'y' for the sample indices
    # fig, ax = plt.subplots()  # Create a figure and axis
    # bottom = [0] * n_samples

    # for i in range(n_classes):
    #     class_probabilities = y_pred_prob.iloc[:, i]
    #     ax.barh(y, class_probabilities, label=f'Class {i}', left=bottom)  # Use barh for horizontal bar chart

    #     for j, prob in enumerate(class_probabilities):
    #         ax.text(prob / 2 + bottom[j], j, f'{prob:.2%}', va='center', ha='center', color='white')  # Adjust text positions

    #     bottom = [sum(x) for x in zip(bottom, class_probabilities)]

    # # Set the tick labels on the y-axis to be the sample indices
    # ax.set_yticks(y)
    # ax.set_yticklabels(sample_indices)

    # plt.xlabel('Probability')
    # plt.ylabel('Sample')  # Adjust the labels
    # plt.title(f'Predicted Probabilities for Each Sample Model {shuffle}')

    # # Print the actual and predicted results
    # for i in range(n_samples):
    #     actual = df['y_test'].iloc[i]
    #     predicted = df['y_pred'].iloc(i)
    #     print(f"Sample {sample_indices[i]} - Actual: {actual}, Predicted: {predicted}")

    # plt.show()
    
    #                                  CREATING THE DATA PROBABILITIES FOR EACH MODEL                                  #
    
    # ROC and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    # Plot ROC curve
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
    
    
    # Initialize a list to store AUC values for all models
    auc_values = []

    # Assuming you have a loop for training and evaluating multiple models
    for shuffle in range(num_shuffles):  # Replace 'models' with your list of models
        # Fit the model and make predictions
        y_pred = model.predict(X_test)
        
        # Calculate ROC and AUC for the current model
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # Append the AUC value to the list
        auc_values.append(roc_auc)

    # Plot the ROC curve for the last model in the list (or a specific model)
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


# Save the metrics to a CSV file for future reference
columns_to_round = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Training Error']  # Replace with your column names
# Round only the specified columns
metrics_df[columns_to_round] = metrics_df[columns_to_round].apply(lambda x: round(x, 4))
metrics_df.to_csv('shuffle_metrics_NeuralNetwork4.csv')
