"""LInked List sparse matrix class
"""

from __future__ import division, print_function, absolute_import

__docformat__ = "restructuredtext en"

__all__ = ['fast_lil_matrix', 'isspmatrix_fast_lil']

import numpy as np

from scipy._lib.six import xrange
from .base import spmatrix, isspmatrix
from .sputils import (getdtype, isshape, isscalarlike, IndexMixin,
                      upcast_scalar, get_index_dtype, isintlike)
from . import _fastlil
from .lil import isspmatrix_lil


class fast_lil_matrix(spmatrix, IndexMixin):
    """Row-based linked list sparse matrix

    This is a structure for constructing sparse matrices incrementally.
    Note that inserting a single item can take linear time in the worst case;
    to construct a matrix efficiently, make sure the items are pre-sorted by
    index, per row.

    This can be instantiated in several ways:
        lil_matrix(D)
            with a dense matrix or rank-2 ndarray D

        lil_matrix(S)
            with another sparse matrix S (equivalent to S.tolil())

        lil_matrix((M, N), [dtype])
            to construct an empty matrix with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
        Number of nonzero elements
    data
        LIL format data array of the matrix
    rows
        LIL format row index array of the matrix

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the LIL format
        - supports flexible slicing
        - changes to the matrix sparsity structure are efficient

    Disadvantages of the LIL format
        - arithmetic operations LIL + LIL are slow (consider CSR or CSC)
        - slow column slicing (consider CSC)
        - slow matrix vector products (consider CSR or CSC)

    Intended Usage
        - LIL is a convenient format for constructing sparse matrices
        - once a matrix has been constructed, convert to CSR or
          CSC format for fast arithmetic and matrix vector operations
        - consider using the COO format when constructing large matrices

    Data Structure
        - An array (``self.rows``) of rows, each of which is a sorted
          list of column indices of non-zero elements.
        - The corresponding nonzero values are stored in similar
          fashion in ``self.data``.


    """
    format = 'fastlil'

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        spmatrix.__init__(self)
        self.dtype = getdtype(dtype, arg1, default=float)

        # First get the shape
        if isspmatrix(arg1):

            A = arg1

            if dtype is not None:
                A = A.astype(dtype)

            self.shape = A.shape
            self.dtype = A.dtype

            if isspmatrix_fast_lil(A):
                if copy:
                    A = A.copy()
                self._matrix = A._matrix
            else:
                A = A.tocsr()
                self._matrix = self._get_matrix()

                if A.data.dtype == np.bool:
                    self._matrix.fromcsr(A.indices, A.indptr, A.data.astype(np.uint8))
                else:
                    self._matrix.fromcsr(A.indices, A.indptr, A.data)

        elif isinstance(arg1, tuple):
            if isshape(arg1):
                if shape is not None:
                    raise ValueError('invalid use of shape parameter')
                M, N = arg1
                self.shape = (M, N)
                self._matrix = self._get_matrix()
            else:
                raise TypeError('unrecognized lil_matrix constructor usage')
        else:
            # assume A is dense
            try:
                A = np.asmatrix(arg1)
            except TypeError:
                raise TypeError('unsupported matrix type')
            else:
                from .csr import csr_matrix
                A = csr_matrix(A, dtype=dtype)

                self.shape = A.shape
                self.dtype = A.dtype
                self._matrix = self._get_matrix()

                if A.data.dtype == np.bool:
                    self._matrix.fromcsr(A.indices, A.indptr, A.data.astype(np.uint8))
                else:
                    self._matrix.fromcsr(A.indices, A.indptr, A.data)

    def _get_matrix(self):

        rows, cols = self.shape

        return getattr(_fastlil,
                       'fast_lil_matrix_{}_{}'.format(
                           'int32',
                           str(self.dtype)))(np.int32(rows),
                                             np.int32(cols))

    def _set(self, row, col, value):

        self._matrix.set(np.int32(row), np.int32(col), self.dtype.type(value))

    def _get(self, row, col, value):

        return self._matrix.get(np.int32(row),
                                np.int32(col),
                                self.dtype.type(value))

    def _from_lil(self, lil_matrix):

        for row_idx, (row_indices, row_data) in enumerate(zip(lil_matrix.rows,
                                                              lil_matrix.data)):
            for col_idx, value in zip(row_indices, row_data):
                self._set(row_idx, col_idx, value)

    def set_shape(self,shape):
        shape = tuple(shape)

        if len(shape) != 2:
            raise ValueError("Only two-dimensional sparse arrays "
                                     "are supported.")
        try:
            shape = int(shape[0]),int(shape[1])  # floats, other weirdness
        except:
            raise TypeError('invalid shape')

        if not (shape[0] >= 0 and shape[1] >= 0):
            raise ValueError('invalid shape')

        if (self._shape != shape) and (self._shape is not None):
            try:
                self = self.reshape(shape)
            except NotImplementedError:
                raise NotImplementedError("Reshaping not implemented for %s." %
                                          self.__class__.__name__)
        self._shape = shape

    shape = property(fget=spmatrix.get_shape, fset=set_shape)

    def __iadd__(self,other):
        self[:,:] = self + other
        return self

    def __isub__(self,other):
        self[:,:] = self - other
        return self

    def __imul__(self,other):
        if isscalarlike(other):
            self[:,:] = self * other
            return self
        else:
            return NotImplemented

    def __itruediv__(self,other):
        if isscalarlike(other):
            self[:,:] = self / other
            return self
        else:
            return NotImplemented

    # Whenever the dimensions change, empty lists should be created for each
    # row

    def getnnz(self, axis=None):
        """Get the count of explicitly-stored values (nonzeros)

        Parameters
        ----------
        axis : None, 0, or 1
            Select between the number of values across the whole matrix, in
            each column, or in each row.
        """

        if axis is None:
            return self._matrix.getnnz(None)
        if axis < 0:
            axis += 2

        if axis in (0, 1):
            return self._matrix.getnnz(axis)
        else:
            raise ValueError('axis out of bounds')

    nnz = property(fget=getnnz)

    def __str__(self):
        val = ''
        for i, row in enumerate(self.rows):
            for pos, j in enumerate(row):
                val += "  %s\t%s\n" % (str((i, j)), str(self.data[i][pos]))
        return val[:-1]

    def getrowview(self, i):
        """Returns a view of the 'i'th row (without copying).
        """
        new = lil_matrix((1, self.shape[1]), dtype=self.dtype)
        new.rows[0] = self.rows[i]
        new.data[0] = self.data[i]
        return new

    def getrow(self, i):
        """Returns a copy of the 'i'th row.
        """
        i = self._check_row_bounds(i)
        new = lil_matrix((1, self.shape[1]), dtype=self.dtype)
        new.rows[0] = self.rows[i][:]
        new.data[0] = self.data[i][:]
        return new

    def _check_row_bounds(self, i):
        if i < 0:
            i += self.shape[0]
        if i < 0 or i >= self.shape[0]:
            raise IndexError('row index out of bounds')
        return i

    def _check_col_bounds(self, j):
        if j < 0:
            j += self.shape[1]
        if j < 0 or j >= self.shape[1]:
            raise IndexError('column index out of bounds')
        return j

    def __getitem__(self, index):
        """Return the element(s) index=(i, j), where j may be a slice.
        This always returns a copy for consistency, since slices into
        Python lists return copies.
        """

        # Scalar fast path first
        if isinstance(index, tuple) and len(index) == 2:
            i, j = index
            # Use isinstance checks for common index types; this is
            # ~25-50% faster than isscalarlike. Other types are
            # handled below.
            if ((isinstance(i, int) or isinstance(i, np.integer)) and
                    (isinstance(j, int) or isinstance(j, np.integer))):

                i = self._check_row_bounds(i)
                j = self._check_col_bounds(j)

                return self._matrix.get(i, j)

        # Utilities found in IndexMixin
        i, j = self._unpack_index(index)

        i_intlike = False
        i_slice = False
        i_list = False

        j_intlike = False
        j_slice = False
        j_list= False

        # Proper check for other scalar index types
        if isintlike(i):
            i_intlike = True
        elif isinstance(i, slice):
            i_slice = True
        elif isinstance(i, (list, np.ndarray)):
            i_list = True
        else:
            raise ValueError

        if isintlike(j):
            j_intlike = True
        elif isinstance(j, slice):
            j_slice = True
        elif isinstance(j, (list, np.ndarray)):
            j_list = True
        else:
            raise ValueError

        if i_intlike and j_intlike:
            i = self._check_row_bounds(i)
            j = self._check_col_bounds(j)
            return self._matrix.get(i, j)

        if i_intlike:
            i = self._check_row_bounds(i)
            row_indices = np.array([i], dtype=np.int32)
        elif i_slice:
            row_indices = np.arange(*i.indices(self.shape[0]), dtype=np.int32)
        elif i_list:
            row_indices = np.array(i, dtype=np.int32)
        else:
            raise ValueError

        if j_intlike:
            j = self._check_col_bounds(j)
            col_indices = np.array([j], dtype=np.int32)
        elif j_slice:
            col_indices = np.arange(*j.indices(self.shape[1]), dtype=np.int32)
        elif j_list:
            col_indices = np.array(j, dtype=np.int32)
        else:
            raise ValueError

        if (
            row_indices.ndim == 1 and col_indices.ndim == 1
                and i_list and j_list
        ):

            if row_indices.shape != col_indices.shape:
                raise IndexError

            new_shape = (1, len(row_indices))
            new = fast_lil_matrix(new_shape, dtype=self.dtype)
            new._matrix = self._matrix.fancy_get_elems(row_indices,
                                                       col_indices)

        else:
            new_shape = (len(row_indices),
                         len(col_indices))
            new = fast_lil_matrix(new_shape, dtype=self.dtype)
            new._matrix = self._matrix.fancy_get(row_indices.flatten(),
                                                 col_indices.flatten())

        return new

    def _get_row_ranges(self, rows, col_slice):
        """
        Fast path for indexing in the case where column index is slice.

        This gains performance improvement over brute force by more
        efficient skipping of zeros, by accessing the elements
        column-wise in order.

        Parameters
        ----------
        rows : sequence or xrange
            Rows indexed. If xrange, must be within valid bounds.
        col_slice : slice
            Columns indexed

        """
        j_start, j_stop, j_stride = col_slice.indices(self.shape[1])
        col_range = xrange(j_start, j_stop, j_stride)
        nj = len(col_range)
        new = lil_matrix((len(rows), nj), dtype=self.dtype)

        _csparsetools.lil_get_row_ranges(self.shape[0], self.shape[1],
                                         self.rows, self.data,
                                         new.rows, new.data,
                                         rows,
                                         j_start, j_stop, j_stride, nj)

        return new

    def __setitem__(self, index, x):
        # Scalar fast path first
        if isinstance(index, tuple) and len(index) == 2:
            i, j = index
            # Use isinstance checks for common index types; this is
            # ~25-50% faster than isscalarlike. Scalar index
            # assignment for other types is handled below together
            # with fancy indexing.
            if ((isinstance(i, int) or isinstance(i, np.integer)) and
                    (isinstance(j, int) or isinstance(j, np.integer))):
                x = self.dtype.type(x)
                if x.size > 1:
                    # Triggered if input was an ndarray
                    raise ValueError("Trying to assign a sequence to an item")

                i = self._check_row_bounds(i)
                j = self._check_col_bounds(j)

                self._matrix.set(i, j, x)
                return

        # General indexing
        i, j = self._unpack_index(index)

        # shortcut for common case of full matrix assign:
        if (isspmatrix(x) and isinstance(i, slice) and i == slice(None) and
                isinstance(j, slice) and j == slice(None)
                and x.shape == self.shape):
            x = fast_lil_matrix(x, dtype=self.dtype)
            self._matrix = x._matrix
            return

        i, j = self._index_to_arrays(i, j)

        if isspmatrix(x):
            x = x.toarray()

        # Make x and i into the same shape
        x = np.asarray(x, dtype=self.dtype)
        x, _ = np.broadcast_arrays(x, i)

        if x.shape != i.shape:
            raise ValueError("shape mismatch in assignment")

        # Set values
        i, j, x = _prepare_index_for_memoryview(i, j, x)

        for row_idx in range(i.shape[0]):
            for col_idx in range(i.shape[1]):
                self[i[row_idx, col_idx], j[row_idx, col_idx]] = x[row_idx, col_idx]

    def _mul_scalar(self, other):
        if other == 0:
            # Multiply by zero: return the zero matrix
            new = fast_lil_matrix(self.shape, dtype=self.dtype)
        else:
            res_dtype = upcast_scalar(self.dtype, other)

            new = self.copy()
            new = new.astype(res_dtype)
            new._matrix.mul(new.dtype.type(other))

        return new

    def __truediv__(self, other):           # self / other
        if isscalarlike(other):
            new = self.copy()
            new._matrix.mul(1 / new.dtype.type(other))
            return new
        else:
            return self.tocsr() / other

    def copy(self):

        new = fast_lil_matrix(self.shape, dtype=self.dtype)
        new._matrix = self._matrix.copy()

        return new

    def reshape(self,shape):
        new = lil_matrix(shape, dtype=self.dtype)
        j_max = self.shape[1]
        for i,row in enumerate(self.rows):
            for col,j in enumerate(row):
                new_r,new_c = np.unravel_index(i*j_max + j,shape)
                new[new_r,new_c] = self[i,j]
        return new

    def toarray(self, order=None, out=None):
        """See the docstring for `spmatrix.toarray`."""
        d = self._process_toarray_args(order, out)

        for row_idx in range(self.shape[0]):
            row_indices, row_data = self._matrix.get_row(row_idx)

            for i in range(len(row_indices)):
                d[row_idx, row_indices[i]] = row_data[i]

        return d

    def transpose(self):
        return self.tocsr().transpose().tolil()

    def tofastlil(self, copy=False):
        if copy:
            return self.copy()
        else:
            return self

    def tocsr(self):
        """ Return Compressed Sparse Row format arrays for this matrix.
        """

        indices, indptr, data = self._matrix.tocsr()

        if self.dtype == np.bool:
            data = data.astype(np.bool)

        from .csr import csr_matrix
        return csr_matrix((data, indices, indptr), shape=self.shape)

    def tocsc(self):
        """ Return Compressed Sparse Column format arrays for this matrix.
        """
        return self.tocsr().tocsc()


def _prepare_index_for_memoryview(i, j, x=None):
    """
    Convert index and data arrays to form suitable for passing to the
    Cython fancy getset routines.

    The conversions are necessary since to (i) ensure the integer
    index arrays are in one of the accepted types, and (ii) to ensure
    the arrays are writable so that Cython memoryview support doesn't
    choke on them.

    Parameters
    ----------
    i, j
        Index arrays
    x : optional
        Data arrays

    Returns
    -------
    i, j, x
        Re-formatted arrays (x is omitted, if input was None)

    """
    if i.dtype > j.dtype:
        j = j.astype(i.dtype)
    elif i.dtype < j.dtype:
        i = i.astype(j.dtype)

    if not i.flags.writeable or i.dtype not in (np.int32, np.int64):
        i = i.astype(np.intp)
    if not j.flags.writeable or j.dtype not in (np.int32, np.int64):
        j = j.astype(np.intp)

    if x is not None:
        if not x.flags.writeable:
            x = x.copy()
        return i, j, x
    else:
        return i, j


def isspmatrix_fast_lil(x):
    return isinstance(x, fast_lil_matrix)
