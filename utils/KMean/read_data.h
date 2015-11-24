//
// Created by lyx on 17/11/15.
//

#ifndef READ_DATA_H_
#define READ_DATA_H_

#include <fstream>

void read_int(std::ifstream& stream, int* val, bool little_endian = false);
void read_bytes(std::ifstream& stream, uint8_t* val, int n);
void read_floats(std::ifstream& stream, float* val, int n);

#endif /* READ_DATA_H_ */
