#ifndef ASEF_H
#define ASEF_H


#include "SyntheticFilterBase.h"

class ASEF : private SyntheticFilterBase
{
public:

    ASEF(string filename);

    ~ASEF();

    void ComputeEyeLocations(IplImage *img, CvRect bb);

    inline CvPoint getLeftEyeLocation(){
        return pEyeL;
    }

    inline CvPoint getRightEyeLocation(){
        return pEyeR;
    }

private:

    static string LeftEyeDetectorLabel;
    static string RightEyeDetectorLabel;
    static string LeftEyeMaskLabel;
    static string RightEyeMaskLabel;

    IplImage *LeftEyeDetector;
    IplImage *RightEyeDetector;
    IplImage *LeftEyeMask;
    IplImage *RightEyeMask;

    IplImage *face_real;
    IplImage *face_im;
    IplImage *complex_data;

    IplImage *F, *Gl, *Gr;
    IplImage *gl, *gr, *g;

    CvPoint pEyeL, pEyeR;

    int nRows, nCols;
};

#endif // ASEF_H
