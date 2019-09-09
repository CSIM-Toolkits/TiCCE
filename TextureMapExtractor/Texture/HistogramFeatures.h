#ifndef HISTOGRAMFEATURES_H
#define HISTOGRAMFEATURES_H

#include <itkImage.h>

using namespace std;

class HistogramFeatures
{
public:
    typedef itk::Image<double, 3> ImageType;
    typedef itk::Image<unsigned char, 3> UCharImageType;

    HistogramFeatures(ImageType::Pointer img, UCharImageType::Pointer mask, std::vector<int> size);
    ~HistogramFeatures();
    void Run();
    std::vector< ImageType::Pointer > GetOutput();
    void SaveImages(string prefix);
private:
    ImageType::Pointer img;
    UCharImageType::Pointer mask;
    std::vector<int> size;

    std::vector<ImageType::Pointer> outputImages;
    std::vector<string> featureNames;
};

#endif // HISTOGRAMFEATURES_H
