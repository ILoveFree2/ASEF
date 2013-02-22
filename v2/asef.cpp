#include "asef.h"
#include "FaceNormIllu.h"

string ASEF::LeftEyeDetectorLabel  = "LeftEyeFilter";
string ASEF::RightEyeDetectorLabel = "RightEyeFilter";
string ASEF::LeftEyeMaskLabel = "LeftMask";
string ASEF::RightEyeMaskLabel = "RightMask";

ASEF::ASEF(string filename)
{
    ///////////////////////////////////////////////////
    //  CONVERT *.yml files into *.tar.gz
    ///////////////////////////////////////////////////
//    IplImage *HLeftEye=(IplImage*)cvLoad("./data/eyes_classifier_data/left eye/hLeftEye_espectral.yml");
//    IplImage *HRightEye=(IplImage*)cvLoad("./data/eyes_classifier_data/right eye/hRightEye_espectral.yml");
//    IplImage *LeftGaussianMask=(IplImage*)cvLoad("./data/eyes_classifier_data/left eye/maskEyeL_K20.yml");
//    IplImage *RightGaussianMask=(IplImage*)cvLoad("./data/eyes_classifier_data/right eye/maskEyeR_K20.yml");
//
//    vector<string> labels;
//    labels.push_back(this->LeftEyeDetectorLabel);
//    labels.push_back(this->RightEyeDetectorLabel);
//    labels.push_back(this->LeftEyeMaskLabel);
//    labels.push_back(this->RightEyeMaskLabel);
//
//
//    vector<Mat> filters;
//    filters.push_back(Mat(HLeftEye));
//    filters.push_back(Mat(HRightEye));
//    filters.push_back(Mat(LeftGaussianMask));
//    filters.push_back(Mat(RightGaussianMask));
//
//    this->SaveFilter("EyeDetector.tar.gz", labels, filters);




    ///////////////////////////////////////////////////
    //  Load eye detectors
    ///////////////////////////////////////////////////
    Mat _LeftEyeDetector = this->LoadFilter(filename, this->LeftEyeDetectorLabel);
    this->LeftEyeDetector = cvCloneImage(&(IplImage)_LeftEyeDetector);

    Mat _RightEyeDetector = this->LoadFilter(filename, this->RightEyeDetectorLabel);
    this->RightEyeDetector = cvCloneImage(&(IplImage)_RightEyeDetector);

    Mat _LeftEyeMask = this->LoadFilter(filename, this->LeftEyeMaskLabel);
    this->LeftEyeMask  = cvCloneImage(&(IplImage)_LeftEyeMask);

    Mat _RightEyeMask = this->LoadFilter(filename, this->RightEyeMaskLabel);
    this->RightEyeMask = cvCloneImage(&(IplImage)_RightEyeMask);


    ///////////////////////////////////////////////////
    //  Create variables
    ///////////////////////////////////////////////////

    this->nCols = this->LeftEyeDetector->width;
    this->nRows = this->LeftEyeDetector->height;

    this->face_real = cvCreateImage(cvSize(this->nCols, this->nRows), IPL_DEPTH_64F, 1);
    this->face_im = cvCreateImage(cvSize(this->nCols, this->nRows), IPL_DEPTH_64F, 1);
    cvZero(this->face_im);

    this->complex_data = cvCreateImage(cvSize(this->nCols, this->nRows), IPL_DEPTH_64F, 2);

    this->F = cvCreateImage(cvSize(this->nCols, this->nRows), IPL_DEPTH_64F, 2);

    this->Gl = cvCreateImage(cvSize(this->nCols, this->nRows), IPL_DEPTH_64F, 2);
    this->Gr = cvCreateImage(cvSize(this->nCols, this->nRows), IPL_DEPTH_64F, 2);

    this->gl = cvCreateImage(cvSize(this->nCols, this->nRows), IPL_DEPTH_64F, 1);
    this->gr = cvCreateImage(cvSize(this->nCols, this->nRows), IPL_DEPTH_64F, 1);
    this->g = cvCreateImage(cvSize(this->nCols, this->nRows), IPL_DEPTH_64F, 1);

}


ASEF::~ASEF(){

    if (this->LeftEyeDetector!=NULL)
        cvReleaseImage(&this->LeftEyeDetector);
    this->LeftEyeDetector = NULL;

    if (this->LeftEyeMask !=NULL)
        cvReleaseImage(&this->LeftEyeMask);
    this->LeftEyeMask = NULL;

    if (this->RightEyeDetector != NULL)
        cvReleaseImage(&this->RightEyeDetector);
    this->RightEyeDetector = NULL;

    if (this->RightEyeMask != NULL)
        cvReleaseImage(&this->RightEyeMask);
    this->RightEyeMask = NULL;

    if (this->face_im != NULL)
        cvReleaseImage(&this->face_im);
    this->face_im = NULL;

    if (this->face_real != NULL)
        cvReleaseImage(&this->face_real);
    this->face_real = NULL;

    if (this->complex_data != NULL)
        cvReleaseImage(&this->complex_data);
    this->complex_data = NULL;

    if (this->F != NULL)
        cvReleaseImage(&this->F);
    this->F = NULL;

    if (this->Gl != NULL)
        cvReleaseImage(&this->Gl);
    this->Gl = NULL;

    if (this->Gr != NULL)
        cvReleaseImage(&this->Gr);
    this->Gr = NULL;

    if (this->g != NULL)
        cvReleaseImage(&this->g);
    this->g = NULL;

    if (this->gl != NULL)
        cvReleaseImage(&this->gl);
    this->gl = NULL;

    if (this->gr != NULL)
        cvReleaseImage(&this->gr);
    this->gr = NULL;
}


void ASEF::ComputeEyeLocations(IplImage *img, CvRect bb){


    assert(img->nChannels==1);


    IplImage *roi = cvCreateImage(cvSize(this->nCols, this->nRows), img->depth, 1);

    cvSetImageROI(img, bb);
    cvResize(img, roi, CV_INTER_LINEAR);
    cvResetImageROI(img);


    IplImage *roi_64 = cvCreateImage(cvGetSize(roi), IPL_DEPTH_64F, 1);

    cvConvertScale(roi, roi_64, 1./255., 0.);

    FaceNormIllu::do_NormIlluRETINA(roi_64, this->face_real, 5.0);


    cvMerge(this->face_real, this->face_im, 0, 0, this->complex_data);

    // do DFT
    cvDFT(this->complex_data, this->F, CV_DXT_FORWARD, this->complex_data->height);


    // G left
    cvMulSpectrums(this->F, this->LeftEyeDetector, this->Gl, CV_DXT_ROWS);

    cvDFT(this->Gl, this->Gl, CV_DXT_INV_SCALE, this->Gl->height);
    cvSplit(this->Gl, this->gl, 0, 0, 0);

    // G right
    cvMulSpectrums(this->F, this->RightEyeDetector, this->Gr, CV_DXT_ROWS);

    cvDFT(this->Gr, this->Gr, CV_DXT_INV_SCALE, this->Gl->height);
    cvSplit(this->Gr, this->gr,0,0,0);


    // add both responses
    double minV, maxV;
    cvMinMaxLoc(this->gl, &minV, &maxV);
    cvConvertScale(this->gl, this->gl, 1./(maxV-minV), -minV/(maxV-minV));

    cvMinMaxLoc(this->gr, &minV, &maxV);
    cvConvertScale(this->gr, this->gr, 1./(maxV-minV), -minV/(maxV-minV));

    cvAdd(this->gl, this->gr, this->g);

    cvMul(this->g, this->LeftEyeMask, this->gl);

    cvMul(this->g, this->RightEyeMask, this->gr);


    ///////////////////////////////////////////////////
    //  Compute Eye Locations
    ///////////////////////////////////////////////////
    float scale;

    cvSetImageROI(this->gl, cvRect(0,0, this->nCols>>1, this->nRows>>1));
    cvMinMaxLoc(this->gl, 0,0,0, &this->pEyeL);
    cvResetImageROI(this->gl);

    scale = (float)bb.width/(float)this->nCols;
    this->pEyeL.x=cvRound((float)this->pEyeL.x * scale + bb.x);
    this->pEyeL.y=cvRound((float)this->pEyeL.y * scale + bb.y);


    cvSetImageROI(this->gr, cvRect(this->nCols>>1, 0, this->nCols>>1, this->nRows>>1));
    cvMinMaxLoc(this->gr, 0,0,0, &this->pEyeR);
    cvResetImageROI(this->gr);

    scale = (float)bb.height/(float)this->nRows;
    this->pEyeR.x=cvRound((float)(this->pEyeR.x + this->nCols*0.5)* scale + bb.x);
    this->pEyeR.y=cvRound((float)this->pEyeR.y * scale + bb.y);


    cvReleaseImage(&roi);
    cvReleaseImage(&roi_64);
}

