#ifndef FACENORMILLU_H
#define FACENORMILLU_H

#include <opencv2/opencv.hpp>

class FaceNormIllu
{
public:
    FaceNormIllu();

    static void do_NormIlluRETINA(IplImage *img_64, IplImage *dst, const double Thr);
};

#endif // FACENORMILLU_H
