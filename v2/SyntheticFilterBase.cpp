#include "SyntheticFilterBase.h"

SyntheticFilterBase::SyntheticFilterBase()
{
}

Mat SyntheticFilterBase::computeGaussianDelta(float x, float y, int nRows, int nCols, float sigma){

    Mat g(nRows, nCols, CV_64F);

    float iSigma = 1.f / sigma;

    for (int i=0; i<nRows; i++){
        double *ptr = g.ptr<double>(i);
        float dy = y - i;
        for (int j=0; j<nCols; j++){
            float dx = x - j;
            ptr[j] = exp( -(dx*dx + dy*dy) * iSigma);
        }
    }
    return g;
}


Mat SyntheticFilterBase::CreateGaussianWindow(int _nRows, int _nCols, float Sigma){
    Mat g(_nRows, _nCols, CV_64F);

    const float iSigma = 1.f / Sigma;

    for (int i=0; i<_nRows; i++){
        double *ptr = g.ptr<double>(i);
        float dy = _nRows*0.5f - i;
        for (int j=0; j<_nCols; j++){
            float dx = _nCols*0.5f - j;
            ptr[j] = exp(- (dx*dx+dy*dy) * iSigma);
        }
    }

    double maxV;
    minMaxLoc(g, NULL, &maxV);
    g.convertTo(g, CV_64F, 1./maxV);

    return g;
}

Mat SyntheticFilterBase::CreateLeftEyeGaussianMask(int nRows, int nCols){
    // K=25
    static double mu[2] = {40.3732, 52.2775}; // x,y
    static double iSigma[4] = {0.0043, 0.0005, 0.0005, 0.0030};

    Mat g;
    g.create(nRows, nCols, CV_64F);



    for (int i=0; i<nRows; i++){
        double dy = mu[1] - i;
        double *ptr = g.ptr<double>(i);
        for (int j=0; j<nCols; j++){
            double dx = mu[0] - j;
            ptr[j] = exp(-(iSigma[0]*dx*dx + iSigma[2]*dx*dy + iSigma[1]*dx*dy + iSigma[3]*dy*dy));
        }
    }

    return g;
}


Mat SyntheticFilterBase::CreateRightEyeGaussianMask(int nRows, int nCols){
    // K=25
    static double mu[2] = {87.6268, 52.2775}; // x,y
    static double iSigma[4] = {0.0043, -0.0005, -0.0005, 0.0030};

    Mat g;
    g.create(nRows, nCols, CV_64F);



    for (int i=0; i<nRows; i++){
        double dy = mu[1] - i;
        double *ptr = g.ptr<double>(i);
        for (int j=0; j<nCols; j++){
            double dx = mu[0] - j;
            ptr[j] = exp(-(iSigma[0]*dx*dx + iSigma[2]*dx*dy + iSigma[1]*dx*dy + iSigma[3]*dy*dy));
        }
    }

    return g;
}
