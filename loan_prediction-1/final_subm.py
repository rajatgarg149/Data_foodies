import pandas as pd
import numpy as np
# Importing training data set
X_train=pd.read_csv('X_train.csv')
#Y_train=pd.read_csv('Y_train.csv')
# Importing testing data set
X_test=pd.read_csv('X_test.csv')
#Y_test=pd.read_csv('Y_test.csv')
print (X_train.head())

from numpy import corrcoef, sum, log, arange
from pylab import pcolor, show, colorbar, xticks, yticks
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

X_train[X_train.dtypes[(X_train.dtypes=="float64")|(X_train.dtypes=="int64")].index.values].hist(bins=50)
plt.show()

X_train.head(10)
X_train.describe()

#import plotly.plotly as py
#from plotly.tools import FigureFactory as FF 

#table = FF.create_table(X_train)
#py.iplot(table, filename='simple_table')

from tabulate import tabulate
print tabulate(X_train.head(10))
X_full=pd.concat([X_train,X_test],axis=0,ignore_index=True)
print tabulate(X_train.describe())

X_full.boxplot(column='ApplicantIncome', by = 'Gender')

#from pandas.tools.plotting import table
#ax = plt.subplot(111, frame_on=False) # no visible frame
#ax.xaxis.set_visible(False)  # hide the x axis
#ax.yaxis.set_visible(False)  # hide the y axis
#table(ax, X_train.head())  # where X_full is your data frame
#plt.savefig('mytable.png')

temp1 = X_full['Credit_History'].value_counts(ascending=True)
temp2 = X_train.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
print 'Frequency Table for Credit History:' 
print temp1

print '\nProbility of getting loan for each Credit History class:' 
print temp2

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")

temp3 = pd.crosstab(X_train['Dependents'], X_train['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

temp3 = pd.crosstab(X_train['Married'], X_train['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

temp3 = pd.crosstab(X_train['Credit_History'], X_train['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

temp3 = pd.crosstab(X_train['Gender'], X_train['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

temp3 = pd.crosstab(X_train['Property_Area'], X_train['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

temp3 = pd.crosstab(X_train['Education'], X_train['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

temp3 = pd.crosstab(X_train['Self_Employed'], X_train['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


#temp2 = X_train.pivot_table(values='Loan_Status',index=['Credit_History','Gender'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
#temp2.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

X_full.apply(lambda x: sum(x.isnull()),axis=0) 
X_full.boxplot(column='LoanAmount',by=['Education','Self_Employed'])
plt.show()
#X_full['LoanAmount'].fillna(X_full['LoanAmount'].mean(), inplace=True)
X_full['Self_Employed'].fillna('No',inplace=True)
table = X_full.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
# Define function to return value of this pivot_table
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]


# Replace missing values
X_full['LoanAmount'].fillna(X_full[X_full['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)

X_full['LoanAmount_log'] = np.log(X_full['LoanAmount'])
X_full['LoanAmount_log'].hist(bins=20)

X_full['TotalIncome'] = X_full['ApplicantIncome'] + X_full['CoapplicantIncome']
X_full['TotalIncome_log'] = np.log(X_full['TotalIncome'])
X_full['TotalIncome_log'].hist(bins=20) 

#temp3 = pd.crosstab(X_train['Dependents'], X_train['Loan_Status'])
temp2 = X_train.pivot_table(values='Loan_Status',index=['Dependents'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
print temp2
temp2 = X_train.pivot_table(values='Loan_Status',index=['Gender'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
print temp2
temp2 = X_train.pivot_table(values='Loan_Status',index=['Married'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
print temp2

print X_full.Gender.value_counts()
X_full['Gender'].fillna('Male', inplace='True')

print X_full.Dependents.value_counts()
X_full['Dependents'].fillna(0, inplace='True')

print X_full.Married.value_counts()
X_full['Married'].fillna('Yes', inplace='True')

print X_full.Credit_History.value_counts()
X_full['Credit_History'].fillna(1, inplace='True')

X_full['Loan_Amount_Term'].fillna(X_full['Loan_Amount_Term'].mean(), inplace='True')

from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
le = LabelEncoder()
for i in var_mod:
    X_full[i] = le.fit_transform(X_full[i])



print X_full.dtypes 

X_train = X_full[:614]
X_test = X_full[614:]
X_train['Loan_Status']=le.fit_transform(X_train['Loan_Status'])

#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

#sns.scatterplot

#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print "Accuracy : %s" % "{0:.3%}".format(accuracy)
  #Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0], n_folds=5)
  error = []
  for train, test in kf:
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
  print "Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error))
  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome]) 


outcome_var = 'Loan_Status'
model = LogisticRegression()
predictor_var = ['Credit_History']
classification_model(model, X_train,predictor_var,outcome_var)
#We can try different combination of variables:
predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']
classification_model(model, X_train,predictor_var,outcome_var)

log=LogisticRegression(penalty='l2',C=0.01)
log.fit(X_train[['ApplicantIncome', 'CoapplicantIncome','LoanAmount','Loan_Amount_Term', 'Credit_History','Property_Area','Intercept']],X_train['Loan_Status'])
accuracy_score(X_train['Loan_Status'],log.predict(X_train[['ApplicantIncome', 'CoapplicantIncome','LoanAmount','Loan_Amount_Term', 'Credit_History','Property_Area']]))

X_train['Intercept'] = 1.0




#####Plotting
sns.swarmplot(x="Gender", y="Loan_Status", data=X_train)

#X_train.drop(['Loan_ID','Loan_Status'], axis=1,inplace=True)
#X_train.to_csv('de.csv')


import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

logit = sm.Logit(X_train['Loan_Status'], X_train[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term', 'Credit_History','Property_Area','Gender','Dependents','Education','Self_Employed','Intercept']])
result = logit.fit()
result.summary()


#results = smf.ols('Loan_Status ~ Credit_History + ApplicantIncome + CoapplicantIncome + LoanAmount + Loan_Amount_Term', data=X_train).fit()


X_train['Intercept'] = 1.0

logit = sm.Logit(X_train['Loan_Status'], X_train[['TotalIncome_log','LoanAmount_log','Loan_Amount_Term', 'Credit_History','Property_Area','Gender','Dependents','Education','Self_Employed','Intercept']])

# fit the model
result = logit.fit()
result.summary()

#from sklearn.preprocessing import scale
#X_train_scale=scale(X_train[['LoanAmount','Loan_Amount_Term', 'Credit_History','Property_Area','Gender','Dependents','Education','Self_Employed']])
#X_test_scale=scale(X_test[['ApplicantIncome', 'CoapplicantIncome','LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
#logit = sm.Logit(X_train['Loan_Status'],X_train_scale)
#result = logit.fit()
#result.summary()
