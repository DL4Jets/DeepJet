
#import sys
#import tensorflow as tf
#sys.modules["keras"] = tf.keras

from DeepJetCore.training.training_base import training_base
from DeepJetCore.modeltools import fixLayersContaining,printLayerInfosAndWeights


#also does all the parsing
train=training_base(testrun=False)

newtraining= not train.modelSet()
#for recovering a training
if newtraining:
    from models import model_deepCSV
    
    train.setModel(model_deepCSV,dropoutRate=0.1)
    
    #train.keras_model=fixLayersContaining(train.keras_model, 'regression', invert=False)
    
    train.train_data.maxFilesOpen=1
 
train.compileModel(learningrate=0.003,
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
    
print(train.keras_model.summary())
#printLayerInfosAndWeights(train.keras_model)

model,history = train.trainModel(nepochs=50,
                                 batchsize=10000, 
                                 stop_patience=300, 
                                 lr_factor=0.5, 
                                 lr_patience=-1, 
                                 lr_epsilon=0.0001, 
                                 lr_cooldown=10, 
                                 lr_minimum=0.00001,
                                 verbose=1,checkperiod=1)
