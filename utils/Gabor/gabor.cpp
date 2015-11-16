/*
 * Gabor filter
 *
 * Gabor filter is one way to extract edge or other features from an image. By using multiple filters with
 * different orientation, one can build a bag-of-features for a given image. This program takes as parameters
 * the kernel size, orientation, etc..
 *
 * Parameters:
 *      k: number of orientations
 *      n: size of the kernel
 *      s[sigma]: standard deviation of the gaussian envelope
 *      t[theta]: orientation of the normal to the parallel stripes of a Gabor function
 *      l[lambda]: wavelength of the sinusoidal factor
 *      g[gamma]: spatial aspect ratio
 *      f: input image
 *      t: output image folder (with '/' in the end)
 *
 * Usage:
 *      gabor -k [k] -n [n] -s [sigma] -t [theta] -l [lambda] -g [gamma] -f [Path_to_input] -t [Path_to_output]
 */

#include <iostream>

using namespace std;

int main() {
    cout << "Hello, World!" << endl;
    return 0;
}