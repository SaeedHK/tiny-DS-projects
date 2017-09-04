import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

df = pd.read_csv('Train.csv' , index_col = 0)


df['Ave Dist Two Don'] = (df['Months since First Donation'] - df['Months since Last Donation'])/df['Number of Donations']

#df['Last More Than Ave'] = df['Months since Last Donation'] > df['Ave Dist Two Don']
#df['Last More Than Ave'] = df['Last More Than Ave'].astype(int)

df = df.loc[:,['Months since Last Donation', 'Number of Donations', 'Ave Dist Two Don' , 'Months since First Donation'
               ,'Made Donation in March 2007']]

X = df.loc[:,['Months since Last Donation', 'Number of Donations', 'Ave Dist Two Don','Months since First Donation']]
y = df.loc[:,'Made Donation in March 2007']

X['prod01'] = X.iloc[:,0] * X.iloc[:,1]
X['prod02'] = X.iloc[:,0] * X.iloc[:,2]
X['prod12'] = X.iloc[:,1] * X.iloc[:,2]

#U = X.loc[:,'Number of Donations']/10
#X.loc[:,'Number of Donations'] = U

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1232 , stratify = y)

my_ind = (X_train.iloc[:,0] <30) & (X_train.iloc[:,1] <30)
X_train = X_train[my_ind]
y_train= y_train[my_ind]


plt.scatter(X_train.loc[:,'Months since Last Donation'][y_train==0] , X_train.loc[:,'Ave Dist Two Don'][y_train==0]
            ,c='b'
            , label= 'Not Donated' , alpha=0.5)
plt.scatter(X_train.loc[:,'Months since Last Donation'][y_train==1] , X_train.loc[:,'Ave Dist Two Don'][y_train==1]
            ,c='r'
            , label= 'Donated' , alpha=0.5)


plt.xlabel('Months since Last Donation')
plt.ylabel('Ave Dist Two Don')

plt.axis([0,30 , 0 , 30])


plt.legend()
plt.show()





IND = (X.loc[:,'Months since Last Donation'] < 5) & (X.loc[:,'Ave Dist Two Don'] < 5)
IND_train= (X_train.loc[:,'Months since Last Donation'] < 5) & (X_train.loc[:,'Ave Dist Two Don'] < 5)
IND_test= (X_test.loc[:,'Months since Last Donation'] < 5) & (X_test.loc[:,'Ave Dist Two Don'] < 5)

X_train_new = X_train.loc[~IND_train,:]
y_train_new = y_train.loc[~IND_train]

X_test_new = X_test.loc[~IND_test,:]
y_test_new = y_test.loc[~IND_test]
y_test_pred_don = y_test.loc[IND_test]

u = abs(y_test_pred_don - 1).sum()








from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train_new , y_train_new)
y_pred_proba = logreg.predict_proba(X_test_new)[:,1]
SC1 = []
for k in np.linspace(0,1,100):
    y_pred_new= np.empty(len(y_pred_proba))
    for i in range(len(y_pred_proba)):
        if y_pred_proba[i] > k:
            y_pred_new[i] = 1
        else:
            y_pred_new[i] = 0

    SC1.append(1 - (abs(y_pred_new - y_test_new).sum() + u )/len(y_test))

plt.plot(np.linspace(0,1,100) , SC1)
plt.show()

SC = logreg.score(X_test , y_test)
