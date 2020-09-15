import pandas as pd
import numpy as np
from sklearn.linear_model import  LogisticRegression
#reading data
df = pd.read_csv('data.csv')
#print(df.head())
#print(df.tail())
#print(df.info()) # give infp of dataset

print(len(df['fever']))
#
#print(df['fever'].value_counts())
# #
# # print(df['diffBreath'].value_counts())
print(df.describe()) # it will describe the whole data
#
# train , test, split function
def data_split(data, ratio) :
    np.random.seed(42) #seed hm tb use krty h jb hr bar ak e random value chayee ho
    #np.random.permutation(7) => output : [1,4,5,6,2,0,3]
    shuffled = np.random.permutation(len(data)) #permiutation esly use kia h k sary  number use hon lkn random way m
    test_set_size = int(len(data)*ratio)  #total lenght ko ratio k 7 multiply kr k test size define kry gy
    test_indices = shuffled[:test_set_size] #test wali rows ko define kr raha h
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

train, test = data_split(df,0.2) # 0.2 ratio choose kry gy tu test datra 20% hojyga

X_train = train[['fever','bodyPain','age','runnyNose','diffBreath']].to_numpy()
X_test = test [['fever','bodyPain','age','runnyNose','diffBreath']].to_numpy()
# print(X_train)
# print(X_test)
Y_train = train[['infectionProb']].to_numpy().reshape(2060,)
Y_test = test[['infectionProb']].to_numpy().reshape(514,)

clf = LogisticRegression(solver='lbfgs', multi_class='auto')
clf.fit(X_train,Y_train)
inputFeature = [101,1,22,1,1]

infProb = clf.predict_proba([inputFeature])[0][1] #  giving  input value to modle
print(infProb)



#know making User interface