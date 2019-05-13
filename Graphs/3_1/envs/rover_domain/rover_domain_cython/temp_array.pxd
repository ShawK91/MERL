# distutils: language = c++

from libcpp.vector cimport vector

cdef extern from "temp_array.hpp":
    cdef cppclass TempArray[T]:
        TempArray() except +
        void alloc(vector[T]*, size_t) except +
        T& operator[](size_t) except +
        vector[T].iterator begin() except +
        vector[T].iterator end() except +
        