

DeepJet: Repository for training and evaluation of deep neural networks for Jet identification
===============================================================================

This package depends on DeepJetCore (https://github.com/DL4Jets/DeepJetCore)

Setup (CERN)
==========

The DeepJet package and DeepJetCore have to share the same parent directory.

Usage
==============

After logging in, please source the right environment (please cd to the directory first!):
```
cd <your working dir>/DeepJet
source lxplus_env.sh / gpu_env.sh
```

There are several steps needed to use DeepJet.

Define data structure
====

First you should define a datastructure. The datastructures can be found in modules/datastructures. In the TrainData_deepCSV.py file, the class TrainData_deepCSV is the datastructure used for the DeepCSV tagger. Another example is the class TrainData_deepFlavour_FT_reg_noScale in TrainData_deepFlavour.py which is the datastructure used for the DeepFlavour tagger. The input variables are defined using the addBranches function and the truth labels are defined in TrainDataDeepJet.py as classes that are then called in datastructure files.

====
Convert to root to data structure
====
After having defined the datastructure you can now convert a root file into the data structure format, which is what we will train on. For instance if you want to produce files to train DeepCSV you would do
```
convertFromRoot.py -i /path/to/merged/root/ntuple/samples.txt -o /output/path/that/needs/some/disk/space -c TrainData_deepCSV -n 4
```
Here -n refers to the number of cores used. While running a snapshot.dc is produced after every file. If the process crashes you can resume the production using
```
convertFromRoot.py -r /output/path/that/needs/some/disk/space/snapshot.dc -n 4
```

====
Training
====
After having produced the training files, you can proceed with the actual training. In the DeepJet/Train directory are the training files. The one used to train DeepCSV is train_DeepCSV.py and for DeepFlavour it is deepFlavour_reference.py. In the train.trainModel() function, you can adjust the training hyperparameters. The default model checkperiod is 10, but it can be adjusted with the argument checkperiod=5 in the trainModel() function.

Since the training can take a while, it is advised to open a screen session, such that it does not die at logout.
```
ssh lxplus.cern.ch
k5reauth -x -f pagsh
aklog
bash
<note the machine you are on, e.g. lxplus058>
screen
ssh lxplus7
```
Then source the environment, and proceed with the training. 

The training is launched in the following way:
```
cd DeepJet/Train/
python train_template.py /path/to/the/output/of/convert/dataCollection.dc <output dir of your choice>
```
In case the convertFromRoot didn't finish you can also run on snapshot.dc. In case you are running on gpu, you can specify which gpu to run on with --gpu and the fraction of the gpu to use with --gpufraction.

In the output directory a loss curve is produced after every epoch together with a info file containing validation and training loss and accuracy.

Detach the screen session with ctr+a d.
You can go back to the session by logging in to the machine the session is running on (e.g. lxplus58):

```
ssh lxplus.cern.ch
ssh lxplus058
screen -r
``` 
Please close the session when the training is finished. If the training stopped, or your model is undertrained, you can continue a training by simply doing the same training command again. If the output directory already exists, you will be asked if you want to continue the training.

====
Evaluation
====

After the training has finished, the performance can be evaluated.
The evaluation consists of a few steps:

1) You first need to convert the test data to the datastructure format. This is done with
```
convertFromRoot.py --testdatafor <output dir of training>/trainsamples.dc -i /path/to/the/root/ntuple/list_of_test_root_files.txt -o /output/path/for/test/data -n 4
```

2) After having produced the test data structure, you can evaluate the trained model on the test data
```
predict.py <output dir of training>/KERAS_model.h5  /output/path/for/test/data/dataCollection.dc <output directory>
```
This creates output trees which only have the discriminators, but does not store the truth data or pt. There is also a tree_association.txt file, which contains the path to the full root file with the truth data, and the path to the corresponding prediction file.

There is a set of plotting tools with examples in 
DeepJet/Train/Plotting

For instance for making ROC curves, you can use ROC_example.py (after updating the input).

