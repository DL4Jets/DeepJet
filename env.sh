#! /bin/bash

THISDIR=`pwd`
export DEEPJET=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P)
export DEEPJETCORE=$DEEPJET/../DeepJetCore
cd $DEEPJETCORE
if command -v nvidia-smi > /dev/null
then
		source gpu_env.sh
else
		source lxplus_env.sh
fi
cd $DEEPJET
export PYTHONPATH=`pwd`/modules:$PYTHONPATH
cd $THISDIR
