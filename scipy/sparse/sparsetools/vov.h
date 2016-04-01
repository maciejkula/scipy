#include <vector>
#include <exception>
// #ifndef __CSR_H__
// #define __CSR_H__

// #include <set>

// #include <algorithm>
// #include <functional>

// #include "util.h"
// #include "dense.h"


class IndexException: public std::exception {
    virtual const char* what() const throw() {
	return "IndexError";
    }
};


template <class I> I binary_search(std::vector<I>& array, I first, I last, I key) {
   // function:
   //   Searches sortedArray[first]..sortedArray[last] for key.  
   // returns: index of the matching element if it finds key, 
   //         otherwise  -(index where it could be inserted)-1.
   // parameters:
   //   sortedArray in  array of sorted (ascending) values.
   //   first, last in  lower and upper subscript bounds
   //   key         in  value to search for.
   // returns:
   //   index of key, or -insertion_position -1 if key is not 
   //                 in the array. This value can easily be
   //                 transformed into the position to insert it.

   I mid;
   
   while (first <= last) {
       mid = (first + last) / 2;  // compute mid point.
       if (key > array[mid]) 
           first = mid + 1;  // repeat search in top half.
       else if (key < array[mid]) 
           last = mid - 1; // repeat search in bottom half.
       else
           return mid;     // found it. return position /////
   }
   return - (first + 1);    // failed to find key
}


template <class I, class T> class VOVMatrix {
    I rows;
    I cols;

    std::vector<std::vector<I> > indices;
    std::vector<std::vector<T> > data;

public:

    VOVMatrix (I num_rows, I num_cols) {

	rows = num_rows;
	cols = num_cols;

	for (int i = 0; i < rows; i++) {
	    indices.push_back(std::vector<I>());
	    data.push_back(std::vector<T>());
	}
    }

    I check_idx(I idx, I max_idx) {
	if (idx < 0) {
	    idx = max_idx + idx;
	};

	if (idx >= max_idx || idx < 0) {
	    throw IndexException();
	} else {
	    return idx;
	}
    }

    void set_unchecked(I row, I col, T value) {

	I idx;
	std::vector<I>& row_indices = indices[row];
	std::vector<T>& row_data = data[row];

	if (value == T()) {
	    return;
	}

	if (row_indices.size() == 0) {
	    row_indices.push_back(col);
	    row_data.push_back(value);
	} else {
	    idx = binary_search(row_indices,
				I(),
				(I)row_indices.size(),
				col);

	    if (idx >= 0) {
		row_indices[idx] = value;
	    } else {
		idx = -(idx + 1);
		row_indices.insert(row_indices.begin() + idx, value);
	    }
	}
    }

    T get_unchecked(I row, I col) {

	I idx;
	std::vector<I>& row_indices = indices[row];
	std::vector<T>& row_data = data[row];

	if (row_indices.size() == 0) {
	    return T();
	} else {
	    idx = binary_search(row_indices,
				I(),
				(I)row_indices.size(),
				col);

	    if (idx >= 0) {
		return row_data[idx];
	    } else {
		return T();
	    }
	}
    }

    void set(I row, I col, T value) {

	row = check_idx(row, rows);
	col = check_idx(col, cols);

	set_unchecked(row, col, value);
    }

    T get(I row, I col) {

	row = check_idx(row, rows);
	col = check_idx(col, cols);

	return get_unchecked(row, col);
    }

    VOVMatrix fancy_get_elems(I *row_idx_array, I *col_idx_array, I num) {

	I row_idx, col_idx;
	T value;

	VOVMatrix<I, T> mat = VOVMatrix<I, T>(1, num);

	std::vector<I>& row_indices = mat.indices[0];
	std::vector<T>& row_data = mat.data[0];

	for (I i = I(); i < num; i++) {
	    row_idx = check_idx(row_idx_array[i], rows);
	    col_idx = check_idx(col_idx_array[i], cols);

	    value = get_unchecked(row_idx, col_idx);

	    if (value != T()) {
		row_indices.push_back(i);
		row_data.push_back(value);
	    }
	}

	return mat;
    }

    VOVMatrix fancy_get_rows(I *row_idx_array, I num) {

	I row_idx;

	VOVMatrix<I, T> mat = VOVMatrix<I, T>(num, cols);

	for (I i = I(); i < num; i++) {
	    row_idx = check_idx(row_idx_array[i], rows);

	    mat.indices[i].assign(indices[row_idx].begin(),
				  indices[row_idx].end());
	    mat.data[i].assign(data[row_idx].begin(),
			       data[row_idx].end());
	}

	return mat;
    }

    VOVMatrix fancy_get_cols(I *col_idx_array, I num) {

	I row_idx, col_idx;
	T value;

	VOVMatrix<I, T> mat = VOVMatrix<I, T>(rows, num);

	for (I i = I(); i < rows; i++) {

	    std::vector<I>& self_row_indices = indices[i];
	    std::vector<T>& self_row_data = data[i];

	    if (self_row_indices.size() == I()) {
		continue;
	    }

	    std::vector<I>& other_row_indices = mat.indices[i];
	    std::vector<T>& other_row_data = mat.data[i];

	    for (I j = I(); j < num; j++) {
		col_idx = check_idx(col_idx_array(j), cols);

		value = get_unchecked(i, col_idx);

		if (value != T()) {
		    other_row_indices.push_back(j);
		    other_row_data.push_back(value);
		}
	    }
	}

	return mat;
    }
};




