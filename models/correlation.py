from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
np.random.seed(7)

dataset = np.load("../preprocessing/datasetMod4.npy").transpose()
values = [0.0, 0.5, 1.0]
#dataset[:,:34] = np.absolute(dataset[:,:34])
row_count = 34
row_list = [8,4,4,27,33,12,16,13,3,31]
X = dataset[0:int(8 * dataset.shape[0]/10),row_list]
# Y = dataset[0:int(8 * dataset.shape[0]/10),row_count]
Y = np.zeros([int(8 * dataset.shape[0]/10), 3])
for i in range(0, int(8 * dataset.shape[0]/10)):
    Y[i, int(dataset[i,row_count])] = 1.0

Xtest = dataset[int(8 * dataset.shape[0]/10):,row_list]
#Ytest = dataset[int(8 * dataset.shape[0]/10):,row_count]
Ytest = np.zeros([dataset.shape[0] - int(8 * dataset.shape[0]/10), 3])
for i in range(int(8 * dataset.shape[0]/10), dataset.shape[0]-1):
    Ytest[i - int(8*dataset.shape[0]/10), int(dataset[i,row_count])] = 1.0

model = Sequential()
model.add(Dense(row_count, input_dim=10, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(20, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00019,
           amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.fit(X,Y, epochs=10, batch_size=200)
scores = model.evaluate(Xtest, Ytest)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

with open('correlationModel.json','w') as json_file:
    json_file.write(model.to_json())

model.save_weights("correlationModel.h5")

json_file = open('correlationModel.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("correlationModel.h5")
loaded_model.compile(loss='binary_crossentropy', optimizer='adam',
             metrics=['accuracy'])
score = loaded_model.evaluate(Xtest,Ytest)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

