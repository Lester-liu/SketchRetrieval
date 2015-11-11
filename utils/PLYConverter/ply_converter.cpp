/*
 * OFF to PLY 3D mesh converter
 *
 * The OFF and PLY are both formats for 3D mesh, they are extremely similar,
 * thus the conversion is merely a reformulating. The advantage of PLY is
 * that there is a built-in reader with VTK.
 *
 * Usage: ply_converter Path_to_PLY_file Path_to_OFF_file
 */

#include <iostream>

using namespace std;

int main(int argc, char *argv[]) {
    cout << "Hello, World!" << endl;
    return EXIT_SUCCESS;
}