#python plotTools/plotModelADA.py -i  trainOutput/full_info.log

from argparse import ArgumentParser

parser = ArgumentParser('program to convert root tuples to traindata format')
parser.add_argument("-i", help="set input sample description (output from the check.py script)", metavar="FILE")


# process options                                                                                                                                                              
                                 
args=parser.parse_args()
intextfile=args.i

#Load partly trained model
import numpy as np
import json
from pprint import pprint
#data = json.load(open('trainOutput/full_info.log'))
#data = json.load(open('trainOutput_vd0x5_onDA/full_info.log'))
#data = json.load(open('trainOutput_vd0x5_onDA_postFix/full_info.log'))
data = json.load(open(intextfile))


#xVal = np.zeros( len(data) )
val_domada0_acc = np.zeros( len(data) )
domada0_loss = np.zeros( len(data) )
classifier_pred_loss = np.zeros( len(data) )
loss = np.zeros( len(data) )
val_classifier_pred_acc = np.zeros( len(data) )
val_domada0_loss  = np.zeros( len(data) )
domada0_acc = np.zeros( len(data) )
val_loss = np.zeros( len(data) )
classifier_pred_acc = np.zeros( len(data) )
val_classifier_pred_loss = np.zeros( len(data) )

#print (xVal)

index = np.int(0)
for element in data:
    #   xVal[index] = xVal+1
    #print element
    indexD = np.int(0)
    for val in element:
        #print val
        if (indexD == 0):
            val_domada0_acc[index] = element[val]
        if (indexD == 1):
            domada0_loss[index] = element[val]
        if (indexD == 2):
            classifier_pred_loss[index] = element[val]
        if (indexD == 3):
            loss[index] = element[val]
        if (indexD == 4):
            val_classifier_pred_acc[index] = element[val]
        if (indexD == 5):
            val_domada0_loss[index] = element[val]
        if (indexD == 6):
            domada0_acc[index] = element[val]
        if (indexD == 7):
            val_loss[index] = element[val]
        if (indexD == 8):
            classifier_pred_acc[index] = element[val]
        if (indexD == 9):
            val_classifier_pred_loss[index] = element[val]
        indexD = indexD +1
    index = index +1


#val_domada0_acc
#domada0_loss
#classifier_pred_loss
#loss
#val_classifier_pred_acc
#val_domada0_loss
#domada0_acc
#val_loss
#classifier_pred_acc
#val_classifier_pred_loss


import matplotlib.pyplot as plt
# summarize history for accuracy
plt.figure(1)
plt.grid()

plt.subplot(321)

#from pdb import set_trace
#set_trace()

yAx1 = np.concatenate([domada0_loss, val_domada0_loss])

plt.plot(domada0_loss)
plt.plot(val_domada0_loss)
#plt.grid()
plt.xticks(np.arange(0, index+1, 5.0))
plt.ylim(min(yAx1)*0.95, max(yAx1)*1.05)    # set the xlim to xmin, xmax
plt.ylabel('loss')
#plt.xlabel('epoch')
plt.legend(['train da', 'test da'], loc='upper right')


yAx2 = np.concatenate([domada0_acc, val_domada0_acc])

plt.subplot(322)
plt.ylim( 0.4, 1.)    # set the xlim to xmin, xmax
plt.plot(domada0_acc)
plt.plot(val_domada0_acc)
plt.xticks(np.arange(0, index+1, 5.0))
plt.ylim(min(yAx2)*0.95, max(yAx2)*1.05)
plt.ylabel('acc')
#plt.xlabel('epoch')
plt.legend(['train da', 'test da'], loc='upper right')
#plt.show()

#plt.figure(2)

yAx3 = np.concatenate([classifier_pred_loss, val_classifier_pred_loss])

plt.subplot(323)
plt.ylim( 0.1, 1. )    # set the xlim to xmin, xmax
plt.plot(classifier_pred_loss)
plt.plot(val_classifier_pred_loss)
plt.xticks(np.arange(0, index+1, 5.0))
plt.ylim(min(yAx3)*0.95, max(yAx3)*1.05)
plt.ylabel('loss')
#plt.xlabel('epoch')
plt.legend(['train cl', 'test cl'], loc='upper right')


yAx4 = np.concatenate([classifier_pred_acc, val_classifier_pred_acc])

plt.subplot(324)
plt.ylim( 0.1, 1. )    # set the xlim to xmin, xmax
plt.plot(classifier_pred_acc)
plt.plot(val_classifier_pred_acc)
plt.xticks(np.arange(0, index+1, 5.0))
plt.ylim(min(yAx4)*0.95, max(yAx4)*1.05)
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train cl', 'test cl'], loc='upper right')
#plt.show()


yAx5 = np.concatenate([loss, val_loss])

plt.subplot(325)
plt.ylim( 13., 17. )    # set the xlim to xmin, xmax
plt.plot(loss)
plt.plot(val_loss)
plt.xticks(np.arange(0, index+1, 5.0))
plt.ylim(min(yAx5)*0.95, max(yAx5)*1.05)
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train tot', 'test tot'], loc='upper right')
plt.show()
