DeepJet: Repository for training and evaluation of deep neural networks for Jet identification
===============================================================================

This package depends on DeepJetCore 3.X (https://github.com/DL4Jets/DeepJetCore). (but it should still work for DJC2)

Usage
==============

After logging in and setting up the DeepJetCore singularity environment, please source the DeepJet environment (please cd to the directory first!). 
```
cd <your working dir>/DeepJet
source env.sh
```


The preparation for the training consists of the following steps
====

- define the data structure for the training. The DeepJet datastructure is found in the modules directory as the class TrainData_DF. DeepCSV is found as the class TrainData_DeepCSV

- convert the root file to the data strucure for training using DeepJetCore tools. The class argument should be TrainData_DF for DeepJet and TrainData_DeepCSV for DeepCSV:
  ```
    convertFromSource.py -i /path/to/the/root/ntuple/list_of_root_files.txt -o /output/path/that/needs/some/disk/space -c TrainData_DF
  ```

  This step can take a while.


- prepare the training file and the model. Please refer to DeepJet/Train/train_DeepFlavour.py for training of DeepJet and DeepJet/Train/train_DeepCSV.py for DeepCSV.

Training
====

Since the training can take a while, it is advised to open a screen session, such that it does not die at logout.
```
ssh lxplus.cern.ch
<note the machine you are on, e.g. lxplus058>
screen
ssh lxplus7
```
Then source the environment, and proceed with the training. Detach the screen session with ctr+a d.
You can go back to the session by logging in to the machine the session is running on (e.g. lxplus58):

```
ssh lxplus.cern.ch
ssh lxplus058
screen -r
```

Please close the session when the training is finished

the training is launched in the following way:
```
python3 train_DeepFlavour.py /path/to/the/output/of/convert/dataCollection.djcdc <output dir of your choice>
```


Evaluation
====

After the training has finished, the performance can be evaluated.

```
predict.py <output dir of training>/KERAS_model.h5  <output dir of training>/trainsamples.dc <dir with test sample stored as rootfiles>/filelist.txt <output directory>
```

This creates output trees with the prediction scores as well as truth information and some kinematic variables.
