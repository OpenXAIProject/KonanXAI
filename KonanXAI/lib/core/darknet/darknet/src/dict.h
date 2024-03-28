#ifndef DICT_H
#define DICT_H

#include <stdio.h>
#include <stdlib.h>
#include "darknet.h"

#ifdef __cplusplus
exturn "C" {
#endif
typedef enum dtypes {
    Char = 0,
    Int = 1,
    Float = 2,
    Dict = 3,
    CharPtr = 4,
    IntPtr = 5,
    FloatPtr = 6,
    DictPtr = 7,
    IntArray = 8,
    FloatArray = 9,
    IntDPtr = 10,
    FloatDPtr = 11
} dtypes;

typedef struct LINKED_KEY_LIST_ITEM {
    char* key;
    void* value;
    dtypes dtype;
    int n;
    struct LINKED_KEY_LIST_ITEM* link;
} dict_item;

typedef struct LINKED_KEY_LIST_HEAD {
    int count;
    struct LINKED_KEY_LIST_ITEM* head;
    struct LINKED_KEY_LIST_ITEM* tail;
} dict;

// API
// item ����
LIB_API void free_dict_item(dict_item* dict_item);
// dictionary ����
LIB_API dict* create_dict(void);
// dictionary ����
LIB_API void free_dict(dict* dict_ptr);
// dictionary ������ ���
LIB_API dict_item* get_item_dict(dict* dict_ptr, char* key);
// dictionary ������ �߰�
LIB_API void add_key_dict(dict* dict_ptr, char* key, void* value, dtypes dtype, int n);
// dictionary ������ ����
LIB_API void del_key_dict(dict* dict_ptr, char* key);


#ifdef __cplusplus
}
#endif
#endif
