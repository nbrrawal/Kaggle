#   Sample Python script 
#   Testing other peoples codes approaches : Sina Khorami(Python) and orginally from Meghan Risdal(R). 
#   Declare libraries 
%matplotlib inline 

import pandas as pd
import numpy as np 
import re as re 

#   import the data and combine them train and test 
train = pd.read_csv("..//input/train.csv")
test  = pd.read_csv("..//input/test.csv")
# Here we combine both datasets to clean them. this will help avoiding cleaning them, droping vars, missing values imputatiions seprately. 

full_data = [train, test]
#   list var names and types of them 
train.info()

##  Check Gender variable Gender 
#   Since the women and kids were given priorities, we will check if its true, 
#   if its true then women should have higher probability 
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()

# with 0.742038 probability, women have higher chance of survival

##  Check Class variable Pclass 
# Next we will check if class had an effect on survival. 
# It could be possible the people in different classes were given priority during rescue 
train[["Pclass", "Survived"]].groupby(["Pclass"], as_index=False).mean()
#this variable looks to be predicitng that class ertaily had an effect on survival 
# class 1   -- 0.629630 , class 2 --  0.472826 , Class 3  -- 0.242363

# Family seem to have an effect too. We will derive this variable from Sibsp and Parch variable 

for dt in full_data: 
    dt['TotalMembers'] =    dt['SibSp'] + dt['Parch'] + 1 

print(train[['TotalMembers','Survived']].groupby(['TotalMembers'], as_index=False).mean())
#   Derived variable Total family member says that all the families that have family size = 4 had higher chance of survival 

#trying to see if anyone was alone on the ship 
for dt in full_data:
    dt["isAlone"] = 0; 
    dt.loc[dt["TotalMembers"]==1,"isAlone"]=1 
print(train[["isAlone","Survived"]].groupby(["isAlone"], as_index=False).mean())    
# it goes to show that people with no family on the ship had less chances of survival 
# Next we check whether embarked variable is impactful 
# there are few missings present in this variables, we will have to treat them first. Filled it with most frequently present values  
for dt in full_data : 
    dt["Embarked"] = dt["Embarked"].fillna('S')

print(train[["Embarked", "Survived"]].groupby("Embarked",as_index=False).mean())

# Then we will pick fare variable, 
# fill same missing values with mean unlike in tutorial 
#need to understand this one below ????
for dt in full_data :
    dt["Fare"] = dt["Fare"].fillna(train["Fare"].median())
    
train['CatFare'] = pd.qcut(train['Fare'],4)
print(train[['CatFare', 'Survived']].groupby(['CatFare'], as_index=False).mean())

for dt in full_data: 
    age_avg= dt["Age"].mean()
    age_std= dt["Age"].std()
    age_null_count = dt["Age"].isnull().sum()
    age_null_random_list = np.random.randint( age_avg - age_std, age_avg + age_std, size = age_null_count)
    dt['Age'][np.isnan(dt['Age'])] = age_null_random_list
    dt['Age']=dt['Age'].astype(int)
    
train['CatAge'] = pd.qcut(train['Age'], 5)

print(train[['CatAge','Survived']].groupby(['CatAge'] , as_index=False).mean())


def take_title(nme): 
    title_srch = re.search(' ([A-Za-z]+)\.', nme)
    #find title if it exists 
    if title_srch: 
        return title_srch.group(1)
    else:
        return "" 
        
for dt in full_data: 
    dt['Title'] = dt['Name'].apply(take_title)
# find male/female comparison with title 
print(pd.crosstab(train['Title'], train['Sex']))
#squeeze titles to minimum number of categories 

for dt in full_data: 
    dt["Title"] = dt["Title"].replace(["Lady", "Countess", "Capt", "Col","Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"], "Rare")
    dt["Title"] = dt["Title"].replace("Mlle", "Miss")
    dt["Title"] = dt["Title"].replace("Ms", "Miss")
    dt["Title"] = dt["Title"].replace("Mme", "Mrs")

print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())


for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

############################################################################################
#Data Cleaning 
############################################################################################
for dt in full_data:
    dt['Sex'] = dt['Sex'].map({'female' : 0, 'male' : 1}).astype(int)

    dt['Title'] = dt['Title'].map({"Mr." : 1,"Miss" : 2, "Mrs.":3, "Master":4, "Rare":5})
    dt['Title'] = dt['Title'].fillna(0)

    dt['Embarked'] = dt['Embarked'].map({'S':0, 'C':1, 'Q':2 }).astype(int)

    dt.loc[dt['Fare'] <= 7.91, 'Fare']=0 
    dt.loc[(dt['Fare'] > 7.91) & (dt['Fare'] <=14.454) , 'Fare'] = 1
    dt.loc[(dt['Fare'] > 14.454) & (dt['Fare'] <= 31 ), 'Fare'] = 2    
    dt.loc[dt['Fare'] > 31 , 'Fare'] = 3    
    dt['Fare'] = dt['Fare'].astype(int)

    dt.loc[(dt['Age'] <=16 ), 'Age'] = 0
    dt.loc[(dt['Age'] >16 ) & (dt['Age'] <= 32 ) , 'Age'] = 1
    dt.loc[(dt['Age'] >32 ) & (dt['Age'] <= 48 ) , 'Age'] = 2
    dt.loc[(dt['Age'] >48 ) & (dt['Age'] <= 64 ) , 'Age'] = 3
    dt.loc[(dt['Age'] >64 ) , 'Age'] = 4
############################################################################
# Feature selection #
#drop certain vars 
drop_elemnts  = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'TotalMembers']
train = train.drop(drop_elemnts, axis=1)
#drop two more cat vars 
train = train.drop(['CatAge', 'CatFare'], axis=1)
test = test.drop(drop_elemnts, axis=1)

train.head(15)

train = train.values
test  = test.values

##########################################################################
# Modelling #
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

classifier = [  KNeighborsClassifier(3), SVC(probability= True), DecisionTreeClassifier(), RandomForestClassifier(),  
                AdaBoostClassifier(), GradientBoostingClassifier(), GaussianNB(), LinearDiscriminantAnalysis(),  
                QuadraticDiscriminantAnalysis(), LogisticRegression()]

log = pd.DataFrame(columns=["Classifier", "Accuracy"])

stshsp =    StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
 
a = train[0::, 1::]
b = train[0::, 0]

acct_dict = {}

for train_index, test_index in stshsp.split(a,b): 
    a_train, a_test = a[train_index], a[test_index]
    b_train, b_test = b[train_index], b[test_index]
    

for cl in classifier: 
    name = cl.__class__.__name__
    cl.fit(a_train, b_train)
    train_predict = cl.predict(a_test)
    acct = accuracy_score(b_test, train_predict)
    if name in acct_dict: 
        acct_dict[name] += acct 
    else : 
        acct_dict[name] = acct 
    
for cl in acct_dict: 
    acct_dict[cl] = acct_dict[cl]/10.0
    log_entry = pd.DataFrame([[cl, acct_dict[cl]]], columns=["Classifier", "Accuracy"])
    log= log.append(log_entry)
    
plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')
sns.set_color_codes('muted')
sns.barplot(x="Accuracy", y ="Classifier", data=log, color="b")

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
