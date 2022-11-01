#to avoid forced warning in output
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

#importing the dataset
from sklearn.datasets import load_boston
boston_dataset = load_boston()

dataset=pd.DataFrame(boston_dataset.data,columns=boston_dataset.feature_names)
#to see the dataframe
#print(dataset.head())

#assigning the target column as the price of housings
dataset['Price']=boston_dataset.target

#to see the full info about the dataset and to see if any null values
#print(dataset.info())

#to see the whole info about the dataset
#print(dataset.describe())

#Check the missing values
#print(dataset.isnull().sum())

#doing the EDA , seeing the correlation between all the features
#print(dataset.corr())

#plotting the pairplot
# sns.pairplot(dataset)
#plt.show()

# #to see the heatmap
# sns.heatmap(dataset.corr(),annot=True)
#plt.show()

#to plot the scatter plot between crime rate and price
# plt.scatter(dataset['CRIM'],dataset['Price'])
# plt.xlabel("Crime Rate")
# plt.ylabel("Price")

#scatterplot between RM and price
# plt.scatter(dataset['RM'],dataset['Price'])

#plotting boxplot for crime rate
# sns.boxplot(dataset['CRIM'])

#assigning dependent and independent features . price is dependent
X=dataset.iloc[:,:-1]
#all the features except the last one
Y=dataset.iloc[:,-1]
#only the last column, ie price

#to see the new dataset which is seperated as dependent and independent
# print(X.head())
# print(Y.head())


#using train_test_split from Scikitlearn to split the datas to training data and testing data
# we use 33% datas for training
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=10)

#to see the shapes of train and test data
# print(X_train.shape)
# print(Y_train.shape)
# print(X_test.shape)
# print(Y_test.shape)

#standardization of data using  StandardScaler()
scaler=StandardScaler()

#we are doing fit_transform() for train data and transform() for test data , this is to avoid data leakage
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#importing Ridge from Scikitlearn to do ridge regression algorithm
ridgeregr = Ridge(alpha=1)
ridgeregr.fit(X_train,Y_train)
Y_pred_train = ridgeregr.predict(X_train)
Y_pred_test = ridgeregr.predict(X_test)
plt.scatter(Y_test,Y_pred_test)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted Price vs Actual Price")
# plt.show()


#calculate MeanSqureError (MSE) and MeanAbsoluteError (MAE)
MSE = (mean_squared_error(Y_test,Y_pred_test))
MAE = (mean_absolute_error(Y_test,Y_pred_test))
sqrtMSE = (np.sqrt(mean_squared_error(Y_test,Y_pred_test)))

#calculating the r squared value
Rsquare =r2_score(Y_test,Y_pred_test)
# print(Rsquare)


#calculating adjusted R squared value
AdjRsquare = 1 - (1-Rsquare)*(len(Y_test)-1)/(len(Y_test)-X_test.shape[1]-1)
# print(AdjRsquare)



#LASSO regression

folds = 5
#lambda is our alpha value here
params = {'alpha': [0.000001,0.00001,0.0001,0.001, 0.01, 1.0,2]}
# Instantiate Lasso regression
lasso = Lasso()
# Cross validation with 5 folds
model_cv = GridSearchCV(estimator=lasso,
                       param_grid=params,
                       scoring='r2',
                       cv=folds,
                       return_train_score=True,
                       verbose=1)
# Fitting the model with train set
model_cv.fit(X_train, Y_train)

# creating dataframe with model_cv results
lasso_results = pd.DataFrame(model_cv.cv_results_)
lasso_results.head()

# Converting the 'param_alpha' datatype from object to int
lasso_results['param_alpha'] = lasso_results['param_alpha'].astype('int32')

# Plotting mean of Train score
plt.plot(lasso_results['param_alpha'], lasso_results['mean_train_score'])
# Plotting mean of the Test score
plt.plot(lasso_results['param_alpha'], lasso_results['mean_test_score'])

plt.legend(['train score', 'test score'])
plt.xlabel('alpha')
plt.ylabel('mean r2 score')
# plt.show()
# Instantiate Lasso regression with alpha=0.002
model_lasso = Lasso(0.002)
# Fitting the model with the train set
model_lasso.fit(X_train, Y_train)

Y_pred_train = model_lasso.predict(X_train)
# print(metrics.r2_score(y_true = Y_train, y_pred = Y_pred_train))

Y_pred_test = model_lasso.predict(X_test)
# print(metrics.r2_score(y_true = Y_test, y_pred = Y_pred_test))


#Lasso regression parameters
# Coefficients list
model_lasso_parameters = list(model_lasso.coef_)
# Inserting Y Intercept to model parameters list
model_lasso_parameters.insert(0, model_lasso.intercept_)
# Rounding off the coefficients
model_lasso_parameters = [round(i,3) for i in model_lasso_parameters]

cols = X_train.shape()
cols = cols.insert(0,'constant')
lasso_param_list = list(zip(cols, model_lasso_parameters))
# print(lasso_param_list)

lasso_params_df = pd.DataFrame({'Params':cols, 'Coef':model_lasso_parameters})
lasso_params_df = lasso_params_df.loc[lasso_params_df['Coef'] != 0]
#print(lasso_params_df)













