

#include <boost/python.hpp>
#include "boost/python/numpy.hpp"
#include "boost/python/list.hpp"
#include "boost/python/str.hpp"
#include <boost/python/exception_translator.hpp>
#include <exception>

//includes from deepjetcore
#include "helper.h"
#include "simpleArray.h"

namespace p = boost::python;
namespace np = boost::python::numpy;

/*
 * Example of a python module that will be compiled.
 * It can be used, e.g. to convert from fully custom input data
 */

np::ndarray readFirstFeatures(std::string infile){

    auto arr = djc::simpleArray<float>({10,3,4});
    arr.at(0,2,1) = 5. ;//filling some data

    return simpleArrayToNumpy(arr);
}

BOOST_PYTHON_MODULE(c_convert) {
    Py_Initialize();
    np::initialize();
    def("readFirstFeatures", &readFirstFeatures);
}

