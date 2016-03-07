from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM

test_data = t1[['Heel s','Ball of foot m','Ball of foot s']].values

tdata_3d = np.swapaxes(np.dstack(np.split(test_data[:-1], 10)),1,2)
np.shape(tdata_3d)
Ys = np.rint([np.mean(i) for i in np.split(np.asarray(t1_turnlabels[:-1]),105)])
len(Ys)



tdata_3d = np.swapaxes(np.swapaxes(np.dstack(test_data),1,2),0,1)
np.shape(tdata_3d)
Ys = np.rint([np.mean(i) for i in np.split(np.asarray(real_turns),1051)])
np.shape(Ys)


model = Sequential()
model.add(LSTM(output_dim=30, activation='sigmoid', inner_activation='hard_sigmoid', input_shape=(105, 3)))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop',class_mode='binary')

model.fit(tdata_3d, Ys, batch_size=32, nb_epoch=10, show_accuracy=True)
score = model.evaluate(tdata_3d, Ys, batch_size=32)
score

#divide y vec into 10 sub lists for each list in lol get mean value

np.size(c1[['Inner shin m','Outer shin m','Inner shin s','Outer shin s']].values)
