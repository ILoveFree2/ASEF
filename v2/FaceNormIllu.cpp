#include "FaceNormIllu.h"


//static void do_NormIlluRETINA(IplImage *img, IplImage *dst, const double Thr);

FaceNormIllu::FaceNormIllu()
{
}


void FaceNormIllu::do_NormIlluRETINA(IplImage *img_64, IplImage *dst, const double Thr){

    const CvSize fsize=cvGetSize(img_64);

//    IplImage *img_64=cvCreateImage(fsize, IPL_DEPTH_64F, 1);

    IplImage *img_gauss_sigma_1=cvCreateImage(fsize, IPL_DEPTH_64F, 1);
    IplImage *img_gauss_sigma_2=cvCreateImage(fsize, IPL_DEPTH_64F, 1);

    IplImage *F1=cvCreateImage(fsize, IPL_DEPTH_64F, 1);
    IplImage *F2=cvCreateImage(fsize, IPL_DEPTH_64F, 1);

    IplImage *I_la1=cvCreateImage(fsize, IPL_DEPTH_64F, 1);
    IplImage *I_la2=cvCreateImage(fsize, IPL_DEPTH_64F, 1);

    IplImage *Ibip=cvCreateImage(fsize, IPL_DEPTH_64F, 1);
    //IplImage *Inorm=cvCreateImage(fsize, IPL_DEPTH_64F, 1);

//    cvConvert(img, img_64);



    //const double mI1=cvMean(img_64)*0.5;
    const double mI1 = cvAvg(img_64).val[0] * 0.5;

    // NxN, N=6*sigma + 1, where NxN is the gaussian smooth filter
    cvSmooth(img_64, img_gauss_sigma_1, CV_GAUSSIAN, 7, 7, 1, 0);


    cvAddS(img_gauss_sigma_1, cvScalar(mI1), F1, 0);

    double minV, maxV;
    cvMinMaxLoc(img_64, 0, &maxV);

    //	#pragma omp parallel for

    for (int i=0; i<fsize.height; i++){
        double *ptr_img_64=(double*)(img_64->imageData+i*img_64->widthStep);
        double *ptr_F1=(double*)(F1->imageData+i*F1->widthStep);
        double *ptr_I_la1=(double*)(I_la1->imageData+i*I_la1->widthStep);
        for (int j=0; j<fsize.width; j++){
            const double vImg64=ptr_img_64[j];
            const double vF1=ptr_F1[j];
            ptr_I_la1[j]=(maxV+vF1)*vImg64/(vImg64+vF1);
        }
    }

    cvSmooth(I_la1, img_gauss_sigma_2, CV_GAUSSIAN, 19, 19, 3, 0);

//    const double mI2=cvMean(I_la1)*0.5;
    const double mI2 = cvAvg(img_64).val[0] * 0.5;

    cvAddS(img_gauss_sigma_2, cvScalar(mI2), F2, 0);

    cvMinMaxLoc(I_la1, 0, &maxV);

    //#pragma omp parallel for

    for (int i=0; i<fsize.height; i++){
        double *ptr_I_la1=(double*)(I_la1->imageData+i*I_la1->widthStep);
        double *ptr_F2=(double*)(F2->imageData+i*F2->widthStep);
        double *ptr_I_la2=(double*)(I_la2->imageData+i*I_la2->widthStep);
        for (int j=0; j<fsize.width; j++){
            const double vI_la1=ptr_I_la1[j];
            const double vF2=ptr_F2[j];
            ptr_I_la2[j]=(maxV+vF2)*vI_la1/(vI_la1+vF2);
        }
    }



    //// DoG filter, sigma1=0.5, sigma2=4
    cvSmooth(I_la2, img_gauss_sigma_1, CV_GAUSSIAN, 3, 3, 0.5, 0);

    cvSmooth(I_la2, img_gauss_sigma_2, CV_GAUSSIAN, 25, 25, 4, 0);

    cvSub(img_gauss_sigma_1, img_gauss_sigma_2, Ibip,0);

    double meanV, stdV;
//    cvMean_StdDev(Ibip, &meanV, &stdV, 0);
    CvScalar _avgV, _stdV;
    cvAvgSdv(Ibip, &_avgV, &_stdV);
    meanV = _avgV.val[0];
    stdV = _stdV.val[0];

    cvConvertScale(Ibip, dst, 1./stdV, 0);
    //#pragma omp parallel for
    for (int i=0; i<fsize.height; i++){
        double *ptr_Inorm=(double*)(dst->imageData+i*dst->widthStep);
        for (int j=0; j<fsize.width; j++){
            const double vInorm=ptr_Inorm[j];
            //printf("%f	",vInorm);
            if (vInorm>=0){
                ptr_Inorm[j]=MIN(Thr, fabs(vInorm));
            }
            else{
                ptr_Inorm[j]=-MIN(Thr, fabs(vInorm));
            }
        }
    }

//    cvReleaseImage(&img_64);
    cvReleaseImage(&img_gauss_sigma_1);
    cvReleaseImage(&img_gauss_sigma_2);
    cvReleaseImage(&F1);
    cvReleaseImage(&F2);
    cvReleaseImage(&I_la1);
    cvReleaseImage(&I_la2);

    cvReleaseImage(&Ibip);

    // normalizar los valores [0.0, 1.0]
    cvMinMaxLoc(dst, &minV, &maxV);
    cvSubS(dst, cvScalar(minV), dst);
    cvConvertScale(dst, dst, 1.0/(maxV - minV));

}
