// kskim
// dictionary

#include "dict.h"
#include "utils.h"

// dictionary ����
dict* create_dict(void) {
	dict* dict_ptr = xmalloc(sizeof(dict));
	dict_ptr->count = 0;
	dict_ptr->head = NULL;
	dict_ptr->tail = NULL;
	return dict_ptr;
}

// item ����
void free_dict_item(dict_item* dict_item) {
	free(dict_item->key);
	//if (dict_item->dtype == FloatArray || dict_item->dtype == IntArray)
		//free(dict_item->value);
	if (dict_item->dtype == DictPtr)
		free_dict(dict_item);
	free(dict_item);
}

// dictionary ����
void free_dict(dict* dict_ptr) {
	dict_item* next = NULL;
	dict_item* current = NULL;
	int i = 0;
	if (dict_ptr->count > 0) {
		current = dict_ptr->head;
		for (int i = 0; dict_ptr->count; i++) {
			next = current->link;
			free_dict_item(current);
			current = next;
		}
	}
	free(dict_ptr);
}

// dictionary ������ ���
dict_item* get_item_dict(dict* dict_ptr, char* key) {
	dict_item* current = dict_ptr->head;
	dict_item* next = NULL;
	for (int i = 0; i < dict_ptr->count; i++) {
		next = current->link;
		if (strcmp(current->key, key) == 0) {
			return current;
		}
		current = next;
	}
	return NULL;
}

// dictionary ������ �߰�
void add_key_dict(dict* dict_ptr, char* key, void* value, dtypes dtype, int n) {
	if (get_item_dict(dict_ptr, key) != NULL)
		return;
	dict_item* new_item = xmalloc(sizeof(dict_item));
	new_item->key = key;
	new_item->value = value;
	new_item->dtype = dtype;
	new_item->link = NULL;
	new_item->n = n;
	// Array �� ���, Copy
	//if (dtype == FloatArray) {
	//	float* copy = xcalloc(n, sizeof(float));
	//	for (int i = 0; i < n; i++)
	//		copy[i] = ((float*)value)[i];
	//	new_item->value = copy;
	//}
	//else if (dtype == IntArray) {
	//	int* copy = xcalloc(n, sizeof(int));
	//	for (int i = 0; i < n; i++)
	//		copy[i] = ((int*)value)[i];
	//	new_item->value = copy;
	//}
	// ������ �����Ͱ� ������ ù��° ������
	if (dict_ptr->tail == NULL) {
		dict_ptr->head = new_item;
	}
	// ������ �������� ��ũ�� ����
	else {
		dict_ptr->tail->link = new_item;
	}
	dict_ptr->count += 1;
	dict_ptr->tail = new_item;
}

// dictionary ������ ����
void del_key_dict(dict* dict_ptr, char* key) {
	if (get_item_dict(dict_ptr, key) == NULL)
		return;
	// ��ȸ�ϸ� Ű �����͸� ã��
	dict_item* prev = NULL;
	dict_item* current = dict_ptr->head;
	for (int i = 0; i < dict_ptr->count; i++) {
		if (strcmp(current->key, key) == 0) {
			if (prev != NULL) {
				prev->link = current->link;
				free_dict_item(current);
				break;
			}
		}
		prev = current;
		current = current->link;
	}
	dict_ptr->count -= 1;
}