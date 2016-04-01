#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL _scipy_sparse_sparsetools_ARRAY_API

// #include "sparsetools.h"
#include "vov.h"
#include <iostream> 


int main() {
    VOVMatrix<int, int> bla = VOVMatrix<int, int>(1, 1);
    VOVMatrix<int, float> blaf = VOVMatrix<int, float>(1, 1.0);

    bla.set(0, 0, 5);
    std::cout << 10000000 << "\n";
    std::cout << bla.get(0, 0);
    // std::cout << bla.safe_get(0, 10);

    int idx [2] = {0, 0};

    VOVMatrix<int, int> slc = bla.fancy_get_elems(&idx[0], &idx[0], 2);
    std::cout << "\n" << "should be 55: " << slc.get(0, 0) << slc.get(0, 1);

}
