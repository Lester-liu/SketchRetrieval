//
// Created by lyx on 17/11/15.
//

#include "read_data.h"

void read_int(std::ifstream& stream, int* val) {
	// little endian
	for (int i = sizeof(int) - 1; i >= 0; i--)
		stream.read(((char*)val) + i, 1);
}

void read_bytes(std::ifstream& stream, uint8_t* val, int n) {
	stream.read((char*)val, n);
}




