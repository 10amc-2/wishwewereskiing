from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM


test_data = c1[['Inner shin m','Outer shin m','Inner shin s','Outer shin s']].values

tdata_3d = np.dstack(np.split(test_data[:-1], 10))

in_out_neurons = 2
hidden_neurons = 300

model = Sequential()
model.add(LSTM(in_out_neurons, hidden_neurons, return_sequences=False))
model.add(Dense(hidden_neurons, in_out_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")
model.fit(X_train, y_train, nb_epoch=10, validation_split=0.05)




model = Sequential()
model.add(LSTM(output_dim=2, activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(Activation('time_distributed_softmax'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop',class_mode='binary')

model.fit(c1[['Inner shin m','Outer shin m','Inner shin s','Outer shin s']].values, np.asarray(c1_turnlabels), batch_size=16, nb_epoch=10)
score = model.evaluate(X_test, Y_test, batch_size=16)


np.size(c1[['Inner shin m','Outer shin m','Inner shin s','Outer shin s']].values)
