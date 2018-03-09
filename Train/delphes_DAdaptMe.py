

from DeepJetCore.training.training_base import training_base
from keras.layers import Dense, Dropout, Concatenate, LocallyConnected1D, Reshape, Flatten
from keras.models import Model

from Layers import GradientReversal
from Losses import binary_crossentropy_labelweights_Delphes, binary_crossentropy_MConly_Delphes
import keras.backend as K
from keras.layers.core import Reshape
from pdb import set_trace

import keras.backend as K
from DeepJetCore.modeltools import set_trainable, fixLayersContaining

def myDomAdaptModel(Inputs,nclasses,nregclasses,dropoutRate=0.05):
    
    X = Dense(60, activation='relu',name='classifier_dense0') (Inputs[0])#reco inputs    
    X = Dropout(dropoutRate)(X)
    X = Dense(60, activation='relu',name='classifier_dense1')(X)
    X = Dropout(dropoutRate)(X)
    X = Dense(60, activation='relu',name='classifier_dense2')(X)

    Xa = Dropout(dropoutRate)(X)
    X = Dense(40, activation='relu',name='classifier_dense3')(Xa)
    X = Dropout(dropoutRate)(X)
    X = Dense(20, activation='relu',name='classifier_dense4')(X)
    X = Dropout(dropoutRate)(X)
    #three labels
    labelpred = Dense(3, activation='softmax',name='classifier_pred')(X)
    
    Ad = GradientReversal(name='domada_gradrev0')(Xa)
    Ad = Dense(30, activation='relu',name='domada_dense0')(Ad)
    Ad = Dropout(dropoutRate)(Ad)
    Ad = Dense(30, activation='relu',name='domada_dense1')(Ad)
    Ad = Dropout(dropoutRate)(Ad)
    Ad = Dense(30, activation='relu',name='domada_dense2')(Ad)
    Ad = Dense(1,  activation='sigmoid')(Ad)
    
    #make list out of it, three labels from truth - make weights
    Weight = Reshape((3,1),name='domada_reshape')(Inputs[1])
    
    # one-by-one apply weight to label
    Weight = LocallyConnected1D(
			1,1, 
			activation='linear',use_bias=False, 
			name="domada_weight0") (Weight)    
    Weight= Flatten()(Weight)
    Weight = GradientReversal(name='domada_gradrev1')(Weight)
    
    Ad = Concatenate(name='domada_pred')([Ad,Weight]) 
    
    predictions = [labelpred,Ad]
    model = Model(inputs=Inputs, outputs=predictions)
    return model
    

from argparse import ArgumentParser    
parser = ArgumentParser('Run the training')
parser.add_argument('--classifier', action='store_true')
parser.add_argument('--lockdisc', action='store_true')
parser.add_argument('--discriminator', action='store_true')
parser.add_argument('--domada', action='store_true')
parser.add_argument('--lambda', type=float, dest='hp_lambda', default=1, help='Gradient reversal scaling')
parser.add_argument('--classweight', type=float, default=1, help='classification loss weight')
parser.add_argument('--adaweight', type=float, default=1, help='domain adaptation weight')

#also does all the parsing
train=training_base(
	testrun=False,
	parser=parser,
	resumeSilently=True
)
args = train.args
## set_trace()

print 'Setting model'
if not train.modelSet():
	train.setModel(myDomAdaptModel,dropoutRate=0.15)

train.defineCustomPredictionLabels(['prob_isB','prob_isC','prob_isUDSG',
                                    'prob_isMC',
                                    'labweight_0',
                                    'labweight_1',
                                    'labweight_2'])

if args.classifier:
	if args.lockdisc:
		train.keras_model = set_trainable(
			train.keras_model,
			'domada_', False
			)
	for layidx in range(len(train.keras_model.layers)):
		layer = train.keras_model.get_layer(index=layidx)
		print layer.name, layer.trainable
	train.compileModel(
		learningrate=0.0001,
		loss=[binary_crossentropy_MConly_Delphes,
					binary_crossentropy_labelweights_Delphes],
		metrics=['accuracy'],
		loss_weights=[1.,0.])

	print(train.keras_model.summary())
	model,history = train.trainModel(
		nepochs=100, 
		batchsize=2000, 
		maxqsize=10,verbose=1
		)

if args.discriminator:
	train.keras_model = set_trainable(
		train.keras_model,
		'classifier_', False
		)
	train.keras_model = set_trainable(
		train.keras_model,
		'domada_', True
		)
	for layidx in range(len(train.keras_model.layers)):
		layer = train.keras_model.get_layer(index=layidx)
		print layer.name, layer.trainable
	
	train.compileModel(
		learningrate=0.0001,
		loss=[binary_crossentropy_MConly_Delphes,
					binary_crossentropy_labelweights_Delphes],
		metrics=['accuracy'],
		loss_weights=[1.,1.])

	print(train.keras_model.summary())
	train.trainedepoches = 0
	model,history = train.trainModel(
		nepochs=100, 
		batchsize=2000, 
		maxqsize=10
		)

if args.domada:
	train.keras_model = set_trainable(
    train.keras_model,
    'classifier_', True
    )
	train.keras_model = set_trainable(
		train.keras_model,
		'domada_', True
		)
	for layidx in range(len(train.keras_model.layers)):
		layer = train.keras_model.get_layer(index=layidx)
		if 'gradrev' in layer.name:
			train.keras_model.get_layer(index=layidx).hp_lambda = args.hp_lambda
		print layidx, layer.name, layer.trainable
	
	train.compileModel(
		learningrate=0.0001,
		loss=[binary_crossentropy_MConly_Delphes,
					binary_crossentropy_labelweights_Delphes],
		metrics=['accuracy'],
		loss_weights=[args.classweight, args.adaweight]
		)
	print(train.keras_model.summary())
	train.trainedepoches = 0
	model,history = train.trainModel(
		nepochs=100, 
		batchsize=2000, 
		maxqsize=10)


