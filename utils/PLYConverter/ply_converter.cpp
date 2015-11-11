/*
 * OFF to PLY 3D mesh converter
 *
 * The OFF and PLY are both formats for 3D mesh, they are extremely similar,
 * thus the conversion is merely a reformulating. The advantage of PLY is
 * that there is a built-in reader with VTK.
 *
 * Usage: ply_converter Path_to_OFF_file Path_to_PLY_file
 */

#include <iostream>
#include <fstream>

using namespace std;

int main(int argc, char *argv[]) {

    if (argc != 3) {
        cout << "Usage: ply_converter Path_to_PLY_file Path_to_OFF_file" << endl;
        return EXIT_FAILURE;
    }

    string off_path = argv[1];
    string ply_path = argv[2];

    ifstream off_file(off_path);
    ofstream ply_file(ply_path);

    string line;
    getline(off_file, line);

    int n, f;
    off_file >> n >> f;

    ply_file << "ply\n"
             << "format ascii 1.0\n"
             << "element vertex " << n << "\n"
             << "property float x\n"
             << "property float y\n"
             << "property float z\n"
             << "element face " << f << "\n"
             << "property list uchar int vertex_index\n"
             << "end_header\n";

    getline(off_file, line);
    while (getline(off_file, line)) {
        ply_file << line << endl;
    }

    off_file.close();
    ply_file.close();

    return EXIT_SUCCESS;
}