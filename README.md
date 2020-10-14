# Home-Credit-Predictions-with-Machine-Learning-Methods-Kaggle-

The main goal in this project is to identify the people who can repay a loan and who can
not, from the unbanked population. Initially, we’ve examined the dataframes from .csv files
and gained some inferences. We split the data into two different groups; numerical and
categorical. In order to detect abnormalities in our data, we’ve checked the min-max values
of all numerical data and replaced all max values which were inf with NaN values on
DAYS_EMPLOYED column. We tried to find the columns with more than reasonable number
of missing values. We started with Read_dataframes() function, and included data_bureau,
data_installments_payments, data_previous_application, data_application_train,
data_application_test, credit_card_balance, pos_cash_balance dataframes to the project.
Then, with the use of Merge_Dataframes() function, we’ve calculated mean, std, min and
max values of numerical columns and mode values for categorical columns of the
dataframes. We’ve merged these dataframes with train and test data. (we’ve removed ID
and target columns from train data and ID column from test data). Lastly, with the usage of
columns that related to each other, we’ve created eight new columns.
With Correlation() function, we evaluated the correlation between the columns (with corr()
function) and for every column pair which has a bigger correlation value than the threshold
we removed one of these columns from our dataframe. We created Find_Missing_Values()
function to detect columns with high number of missing values and removed the columns
which has a bigger missing value percentage than the threshold we’ve indicated. We’ve used
Normalization_Encoding() function for normalization of numerical columns and used one hot
encoding method for categorical columns. For these applications, we used minmaxscaler
function to make normalization and get_dummies function to make one hot encoding. Inside
the Feature_importance() function, we did 2-fold cross validation. With LGBMClassifier
function we started to create our model and found the features with zero importance and
removed them. Inside the Learning_Model()function, we did 5-fold croos validation with the
data returned from the previous function. We applied our model on the test set and the final
result was given by the average scores. We’ve printed the scores on the submission file
which consist of a header, 2 columns; one with SK_ID_CURR (the ID of each person) and the
other with TARGET (the predicted probability that the person with this ID will repay the
loan).
After that, we executed our model and started to examine the scores and in order to make
our score better we decreased the missing value percentage by %5 and increased the
correlation threshold by %10. Finally, we observed the impact of every dataframe on our
model and decided to use credit_card_balance ve pos_cash_balance dataframes. With all of
these changes, our kaggle score became 0.78230.
