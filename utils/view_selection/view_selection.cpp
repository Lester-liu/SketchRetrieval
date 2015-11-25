/*View selector
 *
 * View selector sort the given views according to their area of shape.
 *
 * Parameter:
 *
 *      input folder containing views of format png
 *      output file (txt file)
 *
 * Output format:
 *      [name of image with the largest shape area] [area of image]
 *      [name of image with the second largest shape area] [area of image]
 *      ...
 *
 * Usage:
 *      view_selection [input_path] [output_path]
 */


#include <fstream>
#include <iostream>
#include <map>
#include <dirent.h>
#include <vector>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

string input,output;
vector<string> filenames;

int areaCalculate(string filename){
    Mat img = imread(filename,CV_LOAD_IMAGE_GRAYSCALE);
    //double mini, maxi;
    //minMaxLoc(img, &mini, &maxi);
    //img.convertTo(img,CV_8U, 255.0/(maxi - mini), -mini * 255.0/(maxi - mini));
    //cout << mini <<' ' << maxi <<endl;
    int area = 0;
    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            if ((img.at<uchar>(i,j)) != 0 )
                area++;
        }
    }
    return area;
}

int main(int argc, char** argv) {
    DIR *dir;
    struct  dirent *ent;
    int size = 0;

    input = argv[1];
    output = argv[2];

    //get all the file names in the input folder
    if ((dir = opendir(input.c_str())) != NULL){
        while((ent  = readdir(dir)) != NULL){
            filenames.push_back(ent->d_name);
            size++;
        }
    }

    ofstream out(output);
    vector<pair<int,string> > dic;

    for(int i = 0; i < size; i++){
        if (filenames[i][0] != 'm')
            continue;
        int k = areaCalculate((input + filenames[i]).c_str());
        if (k == 0)
            cout << (input + filenames[i]).c_str() << endl;
        dic.push_back(make_pair(k,filenames[i]));
    }
    //sort according to its area
    sort(dic.begin(),dic.end());

    for(int i =  dic.size() - 1; i > 0; i--)
        out << dic[i].second <<' ' << dic[i].first << endl;

    out.close();
    return 0;
}