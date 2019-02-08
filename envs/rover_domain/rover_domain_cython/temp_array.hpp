#ifndef TEMP_ARRAY_H
#define TEMP_ARRAY_H
/*
For more efficient programming: Avoid extra malloc/new calls in 
performance critic locations
by saving the memory used by temporary arrays 

temporary arrays should only be defined on the stack
Dynamic Allocation of TempArrays not recommened (behavior is undefined).

There is no bounds checking

Usage:
    TempArrayManager<int> tam;
    
    // 
    // create an array with 10 int
    TempArray<int> ta = tam.alloc(10)
    
    
    // Use array
    ta[4] = 20;
    std::cout << ta[4] << std::endl;
    // memory "freed" automatically when ta goes out of scope
*/

//#include <cstdlib>
#include <iostream>

//using namespace std;
#include <vector>



template <typename T>
class TempArray {
    std::vector<T>* m_buffer;
    std::size_t m_loc;
    std::size_t m_size;
    
public:
    TempArray() {
        m_buffer = nullptr;
        m_loc = 0;
        m_size = 0;
    }
    void alloc(std::vector<T>* buffer, std::size_t size) {
        m_buffer = buffer;
        m_loc = buffer->size();
        buffer->resize(m_loc + size);
        m_size = size;
    }
    ~TempArray() {
        m_buffer->resize(m_loc);
    }
    T& operator[](std::size_t pos) {
        return m_buffer->operator[](m_loc + pos);
    }
    
    typename std::vector<T>::iterator begin() {
        return m_buffer->begin() + m_loc;
    }
    typename std::vector<T>::iterator end() {
        return m_buffer->begin() + m_loc + m_size;
    }
};



#endif
