import numpy as np
import ctypes
def convert_tensor_to_numpy(dt_tensor, dtype='float32'):
    data_shape = dt_tensor.shape
    mat_size = []
    all_dim = 1
    for dim in data_shape:
        if dim != -1:
            mat_size.append(dim)
            all_dim *= dim
    data_pointer = dt_tensor.get_type_ptr(0 * all_dim, dtype)#._Tensor__tensor_ptr#
    dtypes = {
        'int32': ctypes.c_int32,
        'float32': ctypes.c_float,
        'uchar': ctypes.c_uint8
    }
    data_type = dtypes[dtype]
    data_count = all_dim

    array_c = ctypes.cast(data_pointer, ctypes.POINTER(data_type*data_count)).contents
    ndarray = np.ctypeslib.as_array(array_c).reshape(*data_shape).astype(np.float32) # Channel : 1

    return ndarray