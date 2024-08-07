from .api import *

class Dict:
    def __init__(self, dict_ptr):
        self.ptr = dict_ptr
        self._convert()
        
    def __del__(self):
        self.free()

    def free(self):
        pass
    
    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()
    
    def items(self):
        return self.data.items()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        pass
    
    def __delitem__(self, key):
        pass
    
    def _parse_item(self, item):
        key = str(item.contents.key, 'utf-8')
        value = item.contents.value
        dtype = item.contents.dtype
        n = item.contents.n
        data = None
        if dtype == DTYPES.DictPtr:
            dict_p = ct.cast(value, ct.POINTER(LINKED_KEY_LIST))
            data = self._parse_dict(dict_p)
        elif dtype in (DTYPES.Int, DTYPES.Char, DTYPES.Float):
            if value is None:
                data = 0
            else:
                data = value
        elif dtype == DTYPES.CharPtr:
            char_p = ct.cast(value, ct.c_char_p)
            data = char_p.value.decode()
        elif dtype == DTYPES.FloatPtr:
            ptr = ct.cast(value, ct.POINTER(ct.c_float))
            data = ptr
        elif dtype == DTYPES.IntPtr:
            ptr = ct.cast(value, ct.POINTER(ct.c_int))
            data = ptr
        elif dtype == DTYPES.IntDPtr:
            ptr = ct.cast(value, ct.POINTER(ct.POINTER(ct.c_int)))
            data = ptr
        elif dtype == DTYPES.FloatDPtr:
            ptr = ct.cast(value, ct.POINTER(ct.POINTER(ct.c_float)))
            data = ptr
        elif dtype == DTYPES.FloatArray:
            ptr = ct.cast(value, ct.POINTER(ct.c_float))
            data = []
            for i in range(n):
                data.append(float(ptr[i]))
        elif dtype == DTYPES.IntArray:
            ptr = ct.cast(value, ct.POINTER(ct.c_int))
            data = []
            for i in range(n):
                data.append(int(ptr[i]))
        return key, data
    
    def _parse_dict(self, head):
        data = {}
        count = head.contents.count
        item = head.contents.head
        for _ in range(count):
            key, value = self._parse_item(item)
            data[key] = value
            item = item.contents.link
        return data

    def _convert(self):
        self.data = self._parse_dict(self.ptr)