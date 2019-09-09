#ifndef HARALICKFEATURES_H
#define HARALICKFEATURES_H

#include <itkImage.h>

using namespace std;

class HaralickFeatures
{
public:
    typedef itk::Image<double, 3> ImageType;
    typedef itk::Image<unsigned char, 2> SliceUCharImageType;
    typedef itk::Image<unsigned char, 3> UCharImageType;

    HaralickFeatures(ImageType::Pointer img, UCharImageType::Pointer mask, std::vector<int> size);
    ~HaralickFeatures();
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

#endif // HARALICKFEATURES_H
