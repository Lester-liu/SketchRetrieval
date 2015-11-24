//
// Created by lyx on 17/11/15.
//

#include "read_data.h"

void read_int(std::ifstream& stream, int* val, bool little_endian) {
	// little endian
	if (little_endian)
		for (int i = sizeof(int) - 1; i >= 0; i--)
			stream.read(((char*)val) + i, 1);
	else
		stream.read((char*)val, sizeof(int));
}

void read_bytes(std::ifstream& stream, uint8_t* val, int n) {
	stream.read((char*)val, sizeof(uint8_t) * n);
}

void read_floats(std::ifstream& stream, float* val, int n) {
	stream.read((char*)val, sizeof(float) * n);
}




