import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#load dataset to pandas dataframe
credit_card_data= pd.read_csv('creditcard.csv')

#check missing alues in each column
credit_card_data.isnull().sum()
#changing into non-null 
credit_data=credit_card_data.fillna(0)
credit_data.isnull().sum()

#distribution of legit transaction and fraudlent transaction
credit_data['Class'].value_counts()
legit = credit_data[credit_data.Class == 0]
fraud = credit_data[credit_data.Class == 1]
print(legit.shape)
print(fraud.shape)

### statistical measures of the data
legit.Amount.describe()
fraud.Amount.describe()

#compare  the values for both transaction
credit_data.groupby('Class').mean()

legit_sample= legit.sample(n=81)
new_dataset = pd.concat([legit_sample,fraud],axis=0)  ## if axis=1 then the values added column wise

new_dataset.shape
new_dataset['Class'].value_counts()
new_dataset.groupby('Class').mean()

X=new_dataset.drop(columns='Class',axis=1)
Y=new_dataset['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

model= LogisticRegression()

#### training the logistic regression model with training data
model.fit(X_train, Y_train)

## accuracy on training data
X_train_prediction= model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Acuuracy on training data:',training_data_accuracy)

### accuracy on test data
X_test_prediction= model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Acuuracy on test data:',test_data_accuracy)
