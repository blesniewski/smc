from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
import numpy as np
np.random.seed(7)

dataset = np.load("../preprocessing/datasetFull.npy").transpose()
values = [0.0, 0.5, 1.0]

# for x in range(0,dataset.shape[0]):
#     dataset[x,34] =  values[int(dataset[x,34])]

X = dataset[0:int(8 * dataset.shape[0]/10),0:34]
# Y = dataset[0:int(8 * dataset.shape[0]/10),34]
Y = np.zeros([int(8 * dataset.shape[0]/10), 3])
for i in range(0, int(8 * dataset.shape[0]/10)):
    Y[i, int(dataset[i,34])] = 1.0

Xtest = dataset[int(8 * dataset.shape[0]/10):,0:34]
#Ytest = dataset[int(8 * dataset.shape[0]/10):,34]
Ytest = np.zeros([dataset.shape[0] - int(8 * dataset.shape[0]/10), 3])
for i in range(int(8 * dataset.shape[0]/10), dataset.shape[0]-1):
    Ytest[i - int(8*dataset.shape[0]/10), int(dataset[i,34])] = 1.0

model = Sequential()
model.add(Dense(34, input_dim=34, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(25, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.fit(X,Y, epochs=10, batch_size=30)
scores = model.evaluate(Xtest, Ytest)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

with open('oneLayerModel.json','w') as json_file:
    json_file.write(model.to_json())

model.save_weights("oneLayerModel.h5")

json_file = open('oneLayerModel.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("oneLayerModel.h5")
loaded_model.compile(loss='binary_crossentropy', optimizer='adam',
             metrics=['accuracy'])
score = loaded_model.evaluate(Xtest,Ytest)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

