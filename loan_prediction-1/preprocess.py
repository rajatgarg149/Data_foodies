# Importing pandas
import pandas as pd
# Importing training data set
X_train=pd.read_csv('X_train.csv')
Y_train=pd.read_csv('Y_train.csv')
# Importing testing data set
X_test=pd.read_csv('X_test.csv')
Y_test=pd.read_csv('Y_test.csv')
print (X_train.head())

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

X_train[X_train.dtypes[(X_train.dtypes=="float64")|(X_train.dtypes=="int64")].index.values].hist(figsize=[11,11])
plt.show()

# Initializing and Fitting a k-NN model
#from sklearn.neighbors import KNeighborsClassifier
#knn=KNeighborsClassifier(n_neighbors=5)
#knn.fit(X_train[['ApplicantIncome', 'CoapplicantIncome','LoanAmount','Loan_Amount_Term', 'Credit_History']],Y_train)
# Checking the performance of our model on the testing data set
from sklearn.metrics import accuracy_score
#accuracy_score(Y_test,knn.predict(X_test[['ApplicantIncome', 'CoapplicantIncome','LoanAmount', 'Loan_Amount_Term', 'Credit_History']]))

Y_train.Target.value_counts()/Y_train.Target.count()
#Y_test.Target.value_counts()/Y_test.Target.count()

# Importing MinMaxScaler and initializing it
#from sklearn.preprocessing import MinMaxScaler
#min_max=MinMaxScaler()
# Scaling down both train and test data set
#X_train_minmax=min_max.fit_transform(X_train[['ApplicantIncome', 'CoapplicantIncome','LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
#X_test_minmax=min_max.fit_transform(X_test[['ApplicantIncome', 'CoapplicantIncome','LoanAmount', 'Loan_Amount_Term', 'Credit_History']])

#knn=KNeighborsClassifier(n_neighbors=5)
#knn.fit(X_train_minmax,Y_train)
# Checking the model's accuracy
#accuracy_score(Y_test,knn.predict(X_test_minmax))

#applying logistic regression
from sklearn.linear_model import LogisticRegression
log=LogisticRegression(penalty='l2',C=0.01)
log.fit(X_train[['ApplicantIncome', 'CoapplicantIncome','LoanAmount','Loan_Amount_Term', 'Credit_History']],Y_train)
accuracy_score(Y_test,log.predict(X_test[['ApplicantIncome', 'CoapplicantIncome','LoanAmount','Loan_Amount_Term', 'Credit_History']]))
#After scaling
#log.fit(X_train_minmax,Y_train)
# Checking the model's accuracy
#accuracy_score(Y_test,log.predict(X_test_minmax))

# Standardizing the train and test data
from sklearn.preprocessing import scale
X_train_scale=scale(X_train[['ApplicantIncome', 'CoapplicantIncome','LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
X_test_scale=scale(X_test[['ApplicantIncome', 'CoapplicantIncome','LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
# Fitting logistic regression on our standardized data set
from sklearn.linear_model import LogisticRegression
log=LogisticRegression(penalty='l2',C=.01)
log.fit(X_train_scale,Y_train)
# Checking the model's accuracy
accuracy_score(Y_test,log.predict(X_test_scale))

#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import scale
#from sklearn.metrics import accuracy_score
#from sklearn.linear_model import LogisticRegression
#le=LabelEncoder()
# Iterating over all the common columns in train and test
#for col in X_test.columns.values:
#Encoding only categorical variables
#	if X_test[col].dtypes=='object':
	# Using whole data to form an exhaustive list of levels
#    		data=X_train[col].append(X_test[col])
#    		le.fit(data.values)
#    		X_train[col]=le.transform(X_train[col])
#    		X_test[col]=le.transform(X_test[col])

# Standardizing the features
#X_train_scale=scale(X_train)
#X_test_scale=scale(X_test)
# Fitting the logistic regression model
#log=LogisticRegression(penalty='l2',C=.01)
#log.fit(X_train_scale,Y_train)
# Checking the models accuracy
#accuracy_score(Y_test,log.predict(X_test_scale))

corr2 = X_train.corr(method='pearson', min_periods=1)
print corr2

X_Y=pd.concat([X_train,Y_train],axis=1)
X_Y['Target']=X_Y['Target']=='Y'
X_Y['Target']=X_Y['Target'].astype(int)
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))
sns.countplot(x='Target', hue="Gender", data=X_Y, order=[1,0],ax=axis2)
#gender_perc = X_Y[["Gender", "Target"]].groupby(['Gender'],as_index=False).mean()
#sns.barplot(x='Gender', y='Target', data=gender_perc,order=['Male','Female'],ax=axis3)
sns.barplot(x="Gender", y="Target", hue="Education", data=X_Y)
plt.show()


from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder(sparse=False)
X_train_1=X_train
X_test_1=X_test
columns=['Gender', 'Married', 'Dependents', 'Education','Self_Employed','Credit_History', 'Property_Area']
for col in columns:
	# creating an exhaustive list of all possible categorical values
    data=X_train[[col]].append(X_test[[col]])
    enc.fit(data)
    # Fitting One Hot Encoding on train data
    temp = enc.transform(X_train[[col]])
    # Changing the encoded features into a data frame with new column names
    temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col].value_counts().index])
    # In side by side concatenation index values should be same
    # Setting the index values similar to the X_train data frame
    temp=temp.set_index(X_train.index.values)
    # adding the new One Hot Encoded varibales to the train data frame
    X_train_1=pd.concat([X_train_1,temp],axis=1)
    # fitting One Hot Encoding on test data
    temp = enc.transform(X_test[[col]])
    # changing it into data frame and adding column names
    temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col].value_counts().index])
	# Setting the index for proper concatenation
    temp=temp.set_index(X_test.index.values)
    # adding the new One Hot Encoded varibales to test data frame
    X_test_1=pd.concat([X_test_1,temp],axis=1)


# Standardizing the data set
X_train_scale=scale(X_train_1)
X_test_scale=scale(X_test_1)
# Fitting a logistic regression model
log=LogisticRegression(penalty='l2',C=.01)
log.fit(X_train_scale,Y_train)
# Checking the model's accuracy
accuracy_score(Y_test,log.predict(X_test_scale))











