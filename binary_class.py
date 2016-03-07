%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils.visualize_util import plot


t1 = pd.read_csv('Filtered_data/Tom Run1_FILTERED.txt',header=None)
t1.columns = ['Time m','Ball of foot m','Bridge m','Heel m','Calf m','Inner shin m','Outer shin m','acc x m','acc y m','acc z m','gyro x m','gyro y m','gyro z m','Time s','Ball of foot s','Bridge s','Heel s','Calf s','Inner shin s','Outer shin s','acc x s','acc y s','acc z s','gyro x s','gyro y s','gyro z s']
t1 = t1[190:1241]

c1 = pd.read_csv('raw_ski_data/Carolyn Run1.txt',header=None)
c1.columns = ['Time m','Ball of foot m','Bridge m','Heel m','Calf m','Inner shin m','Outer shin m','acc x m','acc y m','acc z m','gyro x m','gyro y m','gyro z m','Time s','Ball of foot s','Bridge s','Heel s','Calf s','Inner shin s','Outer shin s','acc x s','acc y s','acc z s','gyro x s','gyro y s','gyro z s']

c1 = c1[1708:2029]
plt.plot(c1['Time m'].values.tolist(),c1['Heel s'].values.tolist())
plt.plot(t1['Time m'].values.tolist(),t1['gyro x s'].values.tolist(), 'ro', t1['Time m'].values.tolist(),[6000*i for i in t1_turnlabels], 'bo', t1['Time m'].values.tolist(), t1['gyro x m'].values.tolist(), 'g^')

plt.plot(c1['Time m'].values.tolist(),c1['gyro x s'].values.tolist(), 'ro', c1['Time m'].values.tolist(),[6000*i for i in c1_turnlabels], 'bs', c1['Time m'].values.tolist(), movingaverage(c1['gyro x s'].values.tolist(),5), 'g^')

smoothed_gyro = movingaverage(t1['gyro x s'].values.tolist(),20)

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

real_turns = [1 if i > 0 else 0 for i in t1['gyro x s']]
real_turns = [1 if i > 0 else 0 for i in smoothed_gyro]

plt.plot(c1['Time m'].values.tolist(),c1['Ball of foot m'].values.tolist(), 'ro', c1['Time m'].values.tolist(),[500*i for i in real_turns], 'bs', c1['Time m'].values.tolist(), c1['Heel s'].values.tolist(), 'g^')


c1_turns = [1708,1777,1820,1854,1898,1942,1986,2029]
t1_turns = [190,234,277,380,438,496,569,628,715,745,788,847,905,964,1022,1080,1139,1183,1241]


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
c1['smoothed gyro'] = smoothed_gyro

len(t1_turnlabels)
len(t1)


t1_turnlabels = mark_turns(t1_turns)
t1['turn label'] = real_turns
c1['smoothed gyro'] = smoothed_gyro

np.asarray(c1_turnlabels)

len(c1[['Inner shin m','Outer shin m']].values)

len(c1['Ball of foot m'].values.tolist())
len(c1_turnlabels)
c1.head()

model = Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# "class_mode" defaults to "categorical". For correctly displaying accuracy
# in a binary classification problem, it should be set to "binary".
model.compile(loss='binary_crossentropy',optimizer='rmsprop',class_mode='binary')

model.fit(c1[['Inner shin m','Outer shin m','Inner shin s','Outer shin s']].values, np.asarray(c1_turnlabels), nb_epoch=5, batch_size=32)

objective_score = model.evaluate(c1[['Inner shin m','Outer shin m','Inner shin s','Outer shin s']].values, np.asarray(c1_turnlabels), batch_size=32)
