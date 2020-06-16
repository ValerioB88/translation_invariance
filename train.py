from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout, MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from os import listdir
from os.path import isfile, isdir, join
import numpy as np
from sklearn.utils import shuffle
import os

from argparser import args

# batch_size = 1 if args.noise > 0 else 1
batch_size = 32

class TranslInvStimuli(Sequence):

    def __init__(self, images, batch_size = 1):
        self.__n_cats, self.__n_exemplars, img_w, img_h, img_z = images.shape
        self.__batch_size = batch_size
        
        self.__inputs = np.zeros(shape=(self.__n_cats * self.__n_exemplars, img_w, img_h, img_z))
        self.__targets = np.zeros(shape=(self.__n_cats * self.__n_exemplars, self.__n_cats))
        
        i = 0
        
        for cat in range(self.__n_cats):
            for exemplar in range(self.__n_exemplars):
                self.__inputs[i] = images[cat, exemplar]
                self.__targets[i][cat] = 1
                
                i += 1
                
        self.__inputs, self.__targets = shuffle(self.__inputs, self.__targets)
        
    def __len__(self):
        return int(np.ceil(self.__targets.shape[0]  /  self.__batch_size))

    def __getitem__(self, idx):
        batch_x = self.__inputs[idx * self.__batch_size:(idx + 1) * self.__batch_size]
        batch_y = self.__targets[idx * self.__batch_size:(idx + 1) * self.__batch_size]

        return batch_x, batch_y
    
    @staticmethod
    def process_stimuli_dir(stimuli_dir):
        image_files = []
        for f in listdir(stimuli_dir):
            if isdir("{}/{}".format(stimuli_dir, f)):
                cat_images = []
                for f2 in listdir("{}/{}".format(stimuli_dir, f)):
                    if isfile("{}/{}/{}".format(stimuli_dir, f, f2)):
                        img = image.load_img("{}/{}/{}".format(stimuli_dir, f, f2), target_size=(224, 224))
                        x = image.img_to_array(img)
                        x = np.expand_dims(x, axis=0)
                        x = preprocess_input(x)
                        cat_images.append(x[0])
                image_files.append(cat_images)
                
        return np.array(image_files)
    
def test_model(model, data_gen, batch_n = 1):
    acc = 0
    n = 0
    act = 0.0
    while n < batch_n:
        inputs, targets = next(data_gen)
        r = model.predict(inputs)
        acc += float(np.mean(np.equal(np.argmax(targets, axis=1), np.argmax(r, axis=1)) * 1.0))
        act += np.mean(np.max(r, axis=1))
        n += 1
            
    return float(acc) / n, float(act) / n
    
#train_set = TranslInvStimuli.process_stimuli_dir("/home/cogs/projects/translinv/train/")
    
def vgg16_preprocess(image):
    x = np.expand_dims(image, axis=0)
    x = preprocess_input(x)
    return x[0]

img_gen = ImageDataGenerator(
        preprocessing_function=vgg16_preprocess,            
    )

target_size = (args.s,) * 2
g = img_gen.flow_from_directory(
        'data/train/{}.{}.{}.{}'.format(args.s, args.e, args.m, args.noise),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical')

g_test2 = img_gen.flow_from_directory(
            'data/test/{}.{}.{}.{}'.format(args.s, args.e, args.m, args.noise),
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical')
    
#TranslInvStimuli(images=test_set, batch_size=50)#test_set.shape[0] * test_set.shape[1])

if args.t == 0 or args.t == 2:
    include_top = False
else:
    include_top = True
base_model = VGG16(
        weights=('imagenet' if args.pretrained else None), 
        include_top=include_top,
        input_shape=(*target_size, 3)
    )
#print(base_model.summary())
x = base_model.output

if args.gap > 0:
    x = GlobalAveragePooling2D()(x)
else:
    if args.t == 0 or args.t == 2:
        x = Flatten()(x)
    pass


if args.t == 0:
    predictions = Dense(2, activation='softmax', name='predictions')(x)
if args.t == 1:
    predictions = Dense(2, activation='softmax', name='predictions')(base_model.layers[-2].output)
if args.t == 2:
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    predictions = Dense(2, activation='softmax', name='predictions')(x)

model = Model(inputs=base_model.input, outputs=predictions)

if args.pretrained:
    for layer in base_model.layers:
        layer.trainable = False
    lr = 0.0001
else:
    lr = 0.00001
#print(model.summary())
for layer in model.layers:
    print(layer, layer.trainable)

print(model.summary())
model.compile(
        optimizer=Adam(lr=lr),
        loss='categorical_crossentropy', 
        metrics=["acc"])

#train_acc, train_act = test_model(model, g, batch_n=5)        
#test2_acc, test2_act = test_model(model, g_test2)    
#print("Init\ttrain acc:{:.3f}({:.3f})\ttest 2 acc:{:.3f}({:.3f})".format(train_acc, train_act, test2_acc, test2_act))                                
      
#for i in range(100):
steps_per_epoch = g.n // batch_size
steps_per_epoch_valid = g_test2.n // batch_size
model.fit(
        g, 
        steps_per_epoch=24 * 100 // batch_size,  # if args.noise > 0 else 24 * 10,
        epochs=20,
        verbose=1,
        validation_data=g_test2,
        validation_steps=100,
        callbacks=[EarlyStopping(monitor="acc", min_delta = 0.05, patience = 1, verbose=True)]
        )
print('TESTING!')
train_acc, train_act = test_model(model, g, batch_n=240)        
test2_acc, test2_act = test_model(model, g_test2, batch_n=240)    
print("{}\t{}\t{}\t{}".format(train_acc, train_act, test2_acc, test2_act))                                

    