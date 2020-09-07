


#
# This file might need some adjustments but should serve as a good basis
#

PYTHON_INCLUDE = `python-config --includes`
PYTHON_LIB=`python-config --libs`

ROOTSTUFF=`root-config --libs --glibs --ldflags`
ROOTCFLAGS=`root-config  --cflags`

CPP_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))
LD_FLAGS := `root-config --cflags --glibs`  -lMathMore -L${DEEPJETCORE}/compiled -ldeepjetcorehelpers -lquicklz
CC_FLAGS := -fPIC -g -Wall `root-config --cflags`
CC_FLAGS += -I./interface -I${DEEPJETCORE}/compiled/interface
DJC_LIB = -L${DEEPJETCORE}/compiled -ldeepjetcorehelpers 


MODULES := $(wildcard src/*.C)
MODULES_OBJ_FILES := $(addprefix ./,$(notdir $(MODULES:.C=.o)))
MODULES_SHARED_LIBS := $(addprefix ./,$(notdir $(MODULES:.C=.so)))


all: $(MODULES_SHARED_LIBS) $(patsubst bin/%.cpp, %, $(wildcard bin/*.cpp))

#compile the module helpers if necessary
#../modules/libsubpackagehelpers.so:
#        cd ../modules; make; cd -

%: bin/%.cpp  $(OBJ_FILES) 
	g++ $(CC_FLAGS) $(LD_FLAGS) $(OBJ_FILES) $< -o $@ 


obj/%.o: src/%.cpp
	g++ $(CC_FLAGS) -c -o $@ $<


#python modules

%.so: %.o 
	g++  -o $(@) -shared -fPIC  $(LINUXADD) $<   $(ROOTSTUFF)  $(PYTHON_LIB) -lboost_python -lboost_numpy $(DJC_LIB)

%.o: src/%.C 
	g++   $(ROOTCFLAGS) -O2 $(CC_FLAGS) -I./interface $(PYTHON_INCLUDE) -fPIC -c -o $(@) $<


clean: 
	rm -f obj/*.o obj/*.d *.so
	rm -f %

