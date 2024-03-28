import ctypes as ct
import os
import numpy as np

#========================================
# DLL Load
cwd = os.path.dirname(__file__)
if os.name == "posix":
    lib = ct.CDLL(cwd + "/darknet.so", ct.RTLD_GLOBAL)
elif os.name == "nt":
    cwd = cwd.replace("\\", "/")
    src = "/darknet/build/darknet/x64/"
    cpu = "yolo_cpp_dll_no_gpu.dll"
    gpu = "yolo_cpp.dll"
    lib = ct.CDLL(cwd + src + cpu, winmode=0, mode=ct.RTLD_GLOBAL)
else:
    lib = None
    print("Unsupported OS")
    exit()

#========================================
# Structure

FloatPtr = ct.POINTER(ct.c_float)
IntPtr = ct.POINTER(ct.c_int)

class BOX(ct.Structure):
    _fields_ = (
        ("x", ct.c_float),
        ("y", ct.c_float),
        ("w", ct.c_float),
        ("h", ct.c_float),
    )
    
class DETECTION(ct.Structure):
    _fields_ = (
        ("bbox", BOX),
        ("classes", ct.c_int),
        ("best_class_idx", ct.c_int),
        ("prob", FloatPtr),
        ("mask", FloatPtr),
        ("objectness", ct.c_float),
        ("sort_class", ct.c_int),
        ("uc", FloatPtr),
        ("points", ct.c_int),
        ("embeddings", FloatPtr),
        ("embedding_size", ct.c_int),
        ("sim", ct.c_float),
        ("track_id", ct.c_int),
    )
    
DETECTIONPtr = ct.POINTER(DETECTION)

class IMAGE(ct.Structure):
    _fields_ = (
        ("w", ct.c_int),
        ("h", ct.c_int),
        ("c", ct.c_int),
        ("data", FloatPtr),
    )

class DTYPES:
    Char = 0
    Int = 1
    Float = 2
    Dict = 3
    CharPtr = 4
    IntPtr = 5
    FloatPtr = 6
    DictPtr = 7
    IntArray = 8
    FloatArray = 9
    IntDPtr = 10
    FloatDPtr = 11
    
class LAYER_TYPE:
    CONVOLUTIONAL = 0
    DECONVOLUTIONAL = 1
    CONNECTED = 2
    MAXPOOL = 3
    LOCAL_AVGPOOL = 4
    SOFTMAX = 5
    DETECTION = 6
    DROPOUT = 7
    CROP = 8
    ROUTE = 9
    COST = 10
    NORMALIZATION = 11
    AVGPOOL = 12
    LOCAL = 13
    SHORTCUT = 14
    SCALE_CHANNELS = 15
    SAM = 16
    ACTIVE = 17
    RNN = 18
    GRU = 19
    LSTM = 20
    CONV_LSTM = 21
    HISTORY = 22
    CRNN = 23
    BATCHNORM = 24
    NETWORK = 25
    XNOR = 26
    REGION = 27
    YOLO = 28
    GAUSSIAN_YOLO = 29
    ISEG = 30
    REORG = 31
    REORG_OLD = 32
    UPSAMPLE = 33
    LOGXENT = 34
    L2NORM = 35
    EMPTY = 36
    BLANK = 37
    CONTRASTIVE = 38
    IMPLICIT = 39

class LINKED_KEY_LIST_ITEM(ct.Structure):
    pass

LINKED_KEY_LIST_ITEM._fields_ = (
    ("key", ct.c_char_p),
    ("value", ct.c_void_p),
    ("dtype", ct.c_uint),
    ("n", ct.c_int),
    ("link", ct.POINTER(LINKED_KEY_LIST_ITEM))
)

class LINKED_KEY_LIST(ct.Structure):
    _fields_ = (
        ("count", ct.c_int),
        ("head", ct.POINTER(LINKED_KEY_LIST_ITEM)),
        ("tail", ct.POINTER(LINKED_KEY_LIST_ITEM))
    )

#========================================
# API

# load_network
# (cfg_path, weights_path, clear)
load_net = lib.load_network
load_net.argtypes = (ct.c_char_p, ct.c_char_p, ct.c_int)
load_net.restype = ct.c_void_p

# load_network_custom
# (cfg_path, weights_path, clear, batch)
load_net_custom = lib.load_network_custom
load_net_custom.argtypes = (ct.c_char_p, ct.c_char_p, ct.c_int, ct.c_int)
load_net_custom.restype = ct.c_void_p

# free_network_ptr
free_network_ptr = lib.free_network_ptr
free_network_ptr.argtypes = (ct.c_void_p,)
free_network_ptr.restype = ct.c_void_p

# make_image
# (width, height, channel)
make_image = lib.make_image
make_image.argtypes = (ct.c_int, ct.c_int, ct.c_int)
make_image.restype = IMAGE

# free_image
# (Image)
free_image = lib.free_image
free_image.argtypes = (IMAGE,)

# copy_image_from_bytes
# (IMAGE, bytes)
copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = (IMAGE, ct.c_char_p)

# get_network_boxes
get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = (ct.c_void_p, ct.c_int, ct.c_int, ct.c_float, ct.c_float, IntPtr, ct.c_int, IntPtr,
                              ct.c_int)
get_network_boxes.restype = DETECTIONPtr

#=====================================
# dict

# free_dict_item
# (dict_item)
free_dict_item = lib.free_dict_item
free_dict_item.argtypes = (ct.POINTER(LINKED_KEY_LIST_ITEM), )

# create_dict
create_dict = lib.create_dict
create_dict.restype = ct.POINTER(LINKED_KEY_LIST)

# free_dict
free_dict = lib.free_dict
free_dict.argtypes = (ct.POINTER(LINKED_KEY_LIST), )

# get_item_dict
# (dict_ptr, key)
get_item_dict = lib.get_item_dict
get_item_dict.argtypes = (ct.POINTER(LINKED_KEY_LIST), ct.c_char_p)
get_item_dict.restype = ct.POINTER(LINKED_KEY_LIST_ITEM)

# add_key_dict
add_key_dict = lib.get_item_dict
add_key_dict.argtypes = (ct.POINTER(LINKED_KEY_LIST), ct.c_char_p, ct.c_void_p, ct.c_uint)

# del_key_dict
del_key_dict = lib.del_key_dict
del_key_dict.argtypes = (ct.POINTER(LINKED_KEY_LIST), ct.c_char_p)

#=====================================
# hook -> 지울 예정
network_predict_using_gradient_hook = lib.network_predict_using_gradient_hook
network_predict_using_gradient_hook.argtypes = (ct.c_void_p, IMAGE)
network_predict_using_gradient_hook.restype = ct.POINTER(LINKED_KEY_LIST)

# get_network_info
get_network_info = lib.get_network_info
get_network_info.argtypes = (ct.c_void_p, )
get_network_info.restype = ct.POINTER(LINKED_KEY_LIST)

# get_network_layers
get_network_layers = lib.get_network_layers
get_network_layers.argtypes = (ct.c_void_p, )
get_network_layers.restype = ct.POINTER(LINKED_KEY_LIST)

# network_forward_image
network_forward_image = lib.network_forward_image
network_forward_image.argtypes = (ct.c_void_p, IMAGE)

# network_backward_delta
network_backward = lib.network_backward
network_backward.argtypes = (ct.c_void_p, )

# network_zero_delta
network_zero_delta = lib.network_zero_delta
network_zero_delta.argtypes = (ct.c_void_p, )