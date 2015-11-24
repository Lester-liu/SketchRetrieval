#include <fstream>
#include <iostream>
#include <map>
#include <dirent.h>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int k = 1000;

string input,output;
string filenames[40];

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
    stringstream ss;
    ss << argv[1];
    ss >> input;
    ss.clear();
    ss << argv[2];
    ss >> output;
    if ((dir = opendir(input.c_str())) != NULL){
        while((ent  = readdir(dir)) != NULL){
            filenames[size] = ent->d_name;
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
    sort(dic.begin(),dic.end());

    for(int i =  dic.size() - 1; i > 0; i--)
        out << dic[i].second <<' ' << dic[i].first << endl;

    out.close();
    return 0;
}