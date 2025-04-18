import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
import seaborn as sns

from warnings import filterwarnings
filterwarnings(action='ignore')

# Loading Dataset

wine = pd.read_csv("Wine\winequality-red.csv")
print("Successfully Imported Data!")
wine.head()
print(wine.shape)

# Description

wine.describe(include='all')

# Finding Null Values

print(wine.isna().sum())

wine.corr()
wine.groupby('quality').mean()


#Data Analysis
# Countplot:

sns.countplot(wine['quality'])
plt.show()



sns.countplot(wine['pH'])
plt.show()

sns.countplot(wine['alcohol'])
plt.show()

sns.countplot(wine['fixed acidity'])
plt.show()

sns.countplot(wine['volatile acidity'])
plt.show()

sns.countplot(wine['citric acid'])
plt.show()

sns.countplot(wine['density'])
plt.show()


# KDE plot:

sns.kdeplot(wine.query('quality > 2').quality)


#Distplot:

sns.displot(wine['alcohol'])

wine.plot(kind ='box',subplots = True, layout =(4,4),sharex = False)

wine.plot(kind ='density',subplots = True, layout =(4,4),sharex = False)


# Histogram

wine.hist(figsize=(10,10),bins=50)
plt.show()


#Heatmap for expressing correlation

corr = wine.corr()
sns.heatmap(corr,annot=True)


#Pair Plot:

sns.pairplot(wine)


#Violinplot:

sns.violinplot(x='quality', y='alcohol', data=wine)


# Feature Selection

# Create Classification version of target variable
wine['goodquality'] = [1 if x >= 7 else 0 for x in wine['quality']]# Separate feature variables and target variable
X = wine.drop(['quality','goodquality'], axis = 1)
Y = wine['goodquality']

# See proportion of good vs bad wines
wine['goodquality'].value_counts()
X
print(Y)


# Feature Importance

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

from sklearn.ensemble import ExtraTreesClassifier
classifiern = ExtraTreesClassifier()
classifiern.fit(X,Y)
score = classifiern.feature_importances_
print(score)


# Splitting Dataset

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=7)


# LogisticRegression:

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix
print("Accuracy Score:",accuracy_score(Y_test,Y_pred))

confusion_mat = confusion_matrix(Y_test,Y_pred)
print(confusion_mat)


# Using KNN:

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred))


# Using SVC:

from sklearn.svm import SVC
model = SVC()
model.fit(X_train,Y_train)
pred_y = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,pred_y))


# Using Decision Tree:

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy',random_state=7)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred))


# Using GaussianNB:

from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(X_train,Y_train)
y_pred3 = model3.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred3))


# Using Random Forest:

from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(random_state=1)
model2.fit(X_train, Y_train)
y_pred2 = model2.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred2))


# Using Xgboost:

import xgboost as xgb # type: ignore
model5 = xgb.XGBClassifier(random_state=1)
model5.fit(X_train, Y_train)
y_pred5 = model5.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred5))


results = pd.DataFrame({
    'Model': ['Logistic Regression','KNN', 'SVC','Decision Tree' ,'GaussianNB','Random Forest','Xgboost'],
    'Score': [0.870,0.872,0.868,0.864,0.833,0.893,0.879]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df
 
 

#Hence I will use Random Forest algorithms for training my model.







































