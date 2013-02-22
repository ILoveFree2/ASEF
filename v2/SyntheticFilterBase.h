#ifndef SYNTHETICFILTERBASE_H
#define SYNTHETICFILTERBASE_H


#include <opencv2/opencv.hpp>

#include <string>
#include <stdio.h>

using namespace cv;
using namespace std;


template<typename T, size_t N>
T * end(T (&ra)[N]) {
    return ra + N;
}

class SyntheticFilterBase
{
public:
    SyntheticFilterBase();

    Mat computeGaussianDelta(float x, float y, int nRows, int nCols, float sigma);

    Mat CreateGaussianWindow(int _nRows, int _nCols, float Sigma);


    Mat CreateLeftEyeGaussianMask(int nRows, int nCols);

    Mat CreateRightEyeGaussianMask(int nRows,  int nCols);

    inline void SaveFilter(string filename, string label, Mat filter){
        FileStorage ff;
        ff.open(filename, FileStorage::WRITE);
        ff<<label<<filter;
        ff.release();
    }


    inline Mat LoadFilter(string filename, string label){
        Mat g;
        FileStorage ff(filename, FileStorage::READ);
        ff[label.c_str()]>>g;
        ff.release();
        return g;
    }

    // write a set of Mat in a compressed file (i.e. *.tar.gz)
    inline void SaveFilter(string filename, vector<string> labels, vector<Mat> filters){

        FileStorage ff(filename, FileStorage::WRITE);

        for (int i=0; i<labels.size(); i++){
            ff<<labels[i]<<filters[i];
        }
        ff.release();
    }

};

#endif // SYNTHETICFILTERBASE_H
