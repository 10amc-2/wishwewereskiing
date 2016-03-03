%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

c1 = pd.read_csv('raw_ski_data/Carolyn Run1.txt',header=None)
c1.columns = ['Time m','Ball of foot m','Bridge m','Heel m','Calf m','Inner shin m','Outer shin m','acc x m','acc y m','acc z m','gyro x m','gyro y m','gyro z m','Time s','Ball of foot s','Bridge s','Heel s','Calf s','Inner shin s','Outer shin s','acc x s','acc y s','acc z s','gyro x s','gyro y s','gyro z s']

c1 = c1[1708:2029]
plt.plot(c1['Time m'].values.tolist(),c1['Heel s'].values.tolist())

c1_turns = [1708,1777,1820,1854,1898,1942,1986,2029]

def mark_turns(l):
    n = 0
    newl = []
    for i in range(len(l)-1):
        if n == 0:
            for j in range(l[i+1]-l[i]):
                newl.append(0)
            n=1
        else:
            for j in range(l[i+1]-l[i]):
                newl.append(1)
            n=0
    return newl

c1_turnlabels = mark_turns(c1_turns)
c1['turn label'] = c1_turnlabels



c1.head()

model = Sequential()
model.add(Dense(64, input_dim=20, init='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# "class_mode" defaults to "categorical". For correctly displaying accuracy
# in a binary classification problem, it should be set to "binary".
model.compile(loss='binary_crossentropy',optimizer='rmsprop',class_mode='binary')

model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)

objective_score = model.evaluate(X_test, Y_test, batch_size=32)
