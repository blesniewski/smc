from keras.models import model_from_json
import sys

if __name__ == '__main__':
    name = sys.argv[1]
    json_file = open(name+'.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(name+".h5" )
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['categorical_accuracy'])
    loaded_model.save(name+"caly.h5")
