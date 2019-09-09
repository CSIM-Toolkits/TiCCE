#include "HaralickFeatures.h"

#include <itkImageFileReader.h>
#include <itkImageRegionIterator.h>
#include <itkScalarImageToTextureFeaturesFilter.h>

#include <itkScalarImageToCooccurrenceMatrixFilter.h>
#include <itkHistogramToTextureFeaturesFilter.h>
#include <itkRegionOfInterestImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>

#include <itkExtractImageFilter.h>

#include <itkImageDuplicator.h>
#include <itkImageFileWriter.h>

#include <iostream>
#include <sstream>
#include <sys/stat.h>

HaralickFeatures::HaralickFeatures(ImageType::Pointer img, UCharImageType::Pointer mask, std::vector<int> size){
    this->img = img;
    this->mask = mask;
    this->size = size;

    this->featureNames.push_back("energy");
    this->featureNames.push_back("entropy");
    this->featureNames.push_back("correlation");
    this->featureNames.push_back("inv_dif_moment");
    this->featureNames.push_back("inertia");
    this->featureNames.push_back("cluster_shade");
    this->featureNames.push_back("cluster_prom");
    this->featureNames.push_back("har_correlation");
}

HaralickFeatures::~HaralickFeatures(){

}

void HaralickFeatures::Run(){
    //Defines types used
    typedef itk::RescaleIntensityImageFilter<ImageType>                              RescaleFilterType;
    typedef itk::Statistics::ScalarImageToCooccurrenceMatrixFilter<ImageType>        Image2CoOcurrenceType;
    typedef itk::Statistics::HistogramToTextureFeaturesFilter<Image2CoOcurrenceType::HistogramType>
                                                                                     Hist2FeaturesType;
    typedef itk::RegionOfInterestImageFilter<ImageType, ImageType>                   ROIType;
    typedef itk::ImageRegionIterator<ImageType>                                      IteratorType;
    typedef itk::ImageRegionIterator<UCharImageType>                                 UCharIteratorType;

    //Rescales image for values between 0 and 255
    typename RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
    rescaleFilter->SetInput(this->img);
    rescaleFilter->SetOutputMinimum(0);
    rescaleFilter->SetOutputMaximum(255);
    rescaleFilter->Update();

    //Creates output images
    unsigned int numberOfOutputs = 8;
    for (unsigned int i=0; i<numberOfOutputs; i++){
        this->outputImages.push_back(ImageType::New());
        this->outputImages[i]->CopyInformation(this->img);
        this->outputImages[i]->SetRegions(this->img->GetRequestedRegion());
        this->outputImages[i]->Allocate();
        this->outputImages[i]->FillBuffer(0);
    }

    //Creates cooccurrence and texture filters
    typename Image2CoOcurrenceType::Pointer coocFilter = Image2CoOcurrenceType::New();
    typename Hist2FeaturesType::Pointer textFilter = Hist2FeaturesType::New();
    coocFilter->SetPixelValueMinMax(0, 255);
    coocFilter->SetNumberOfBinsPerAxis(16);

    //Defines offset for cooccurrence filter
    Image2CoOcurrenceType::OffsetVectorPointer offsetVec = Image2CoOcurrenceType::OffsetVector::New();
    bool breakLoops = false;
    cout<<"haralick offsets:"<<endl;
    for(int i=-1; i<=1; i++){
        for(int j=-1; j<=1; j++){
            for(int k=-1; k<=1; k++){
                if(i==0 && j==0 && k==0){
                    breakLoops = true;
                }
                if(breakLoops)
                    break;
                ImageType::OffsetType offset = {{i,j,k}};
                offsetVec->push_back(offset);
                cout<<offset;
            }
            if(breakLoops)
                break;
        }
        if(breakLoops)
            break;
    }
    cout<<endl;
    coocFilter->SetOffsets(offsetVec);

    //Creates roi to run through image
    ROIType::Pointer roiFilter = ROIType::New();
    roiFilter->SetInput(rescaleFilter->GetOutput());

    //Creates window to be used in roi
    ImageType::RegionType window;
    ImageType::RegionType::SizeType size;
    for(unsigned int i=0; i<this->size.size(); i++)
        size[i]=this->size[i];

    window.SetSize(size);

    //Create iterators
    IteratorType imgIt (this->img, this->img->GetRequestedRegion());
    UCharIteratorType mapIt (this->mask, this->mask->GetRequestedRegion());
    IteratorType engIt (this->outputImages[0], this->outputImages[0]->GetRequestedRegion());
    IteratorType entIt (this->outputImages[1], this->outputImages[1]->GetRequestedRegion());
    IteratorType corIt (this->outputImages[2], this->outputImages[2]->GetRequestedRegion());
    IteratorType dfmIt (this->outputImages[3], this->outputImages[3]->GetRequestedRegion());
    IteratorType ineIt (this->outputImages[4], this->outputImages[4]->GetRequestedRegion());
    IteratorType clsIt (this->outputImages[5], this->outputImages[5]->GetRequestedRegion());
    IteratorType clpIt (this->outputImages[6], this->outputImages[6]->GetRequestedRegion());
    IteratorType hcorIt(this->outputImages[7], this->outputImages[7]->GetRequestedRegion());

    //Runs Processing
    imgIt.GoToBegin();
    while(!imgIt.IsAtEnd()){
        mapIt.SetIndex(imgIt.GetIndex());
        if(mapIt.Get()!=0){
            window.SetIndex(imgIt.GetIndex());
            ImageType::RegionType region = this->img->GetRequestedRegion();

            if(region.IsInside(window)){
                roiFilter->SetRegionOfInterest(window);
                roiFilter->Update();

                coocFilter->SetInput(roiFilter->GetOutput());
                coocFilter->Update();
                textFilter->SetInput(coocFilter->GetOutput());
                textFilter->Update();

                engIt.SetIndex(imgIt.GetIndex());
                engIt.Set(textFilter->GetFeature(Hist2FeaturesType::Energy));
                entIt.SetIndex(imgIt.GetIndex());
                entIt.Set(textFilter->GetFeature(Hist2FeaturesType::Entropy));
                corIt.SetIndex(imgIt.GetIndex());
                corIt.Set(textFilter->GetFeature(Hist2FeaturesType::Correlation));
                dfmIt.SetIndex(imgIt.GetIndex());
                dfmIt.Set(textFilter->GetFeature(Hist2FeaturesType::InverseDifferenceMoment));
                ineIt.SetIndex(imgIt.GetIndex());
                ineIt.Set(textFilter->GetFeature(Hist2FeaturesType::Inertia));
                clsIt.SetIndex(imgIt.GetIndex());
                clsIt.Set(textFilter->GetFeature(Hist2FeaturesType::ClusterShade));
                clpIt.SetIndex(imgIt.GetIndex());
                clpIt.Set(textFilter->GetFeature(Hist2FeaturesType::ClusterProminence));
                hcorIt.SetIndex(imgIt.GetIndex());
                hcorIt.Set(textFilter->GetFeature(Hist2FeaturesType::HaralickCorrelation));
            }
        }
        ++imgIt;
    }
}

std::vector< HaralickFeatures::ImageType::Pointer> HaralickFeatures::GetOutput(){
    std::vector<ImageType::Pointer> resultImages;

    typedef itk::RescaleIntensityImageFilter<ImageType> RescaleFilterType;
    typename RescaleFilterType::Pointer rescale = RescaleFilterType::New();

    rescale->SetOutputMinimum(0);
    rescale->SetOutputMaximum(255);

    typedef itk::ImageDuplicator<ImageType> ImageDuplicatorType;

    unsigned int numberOfOutputs = 8;
    for(unsigned int i=0; i<numberOfOutputs; i++){
        rescale->SetInput(this->outputImages[i]);
        rescale->Update();
        
        typename ImageDuplicatorType::Pointer duplicator = ImageDuplicatorType::New();

        duplicator->SetInputImage(rescale->GetOutput());
        duplicator->Update();

        resultImages.push_back(duplicator->GetOutput());
    }

    return resultImages;
}

void HaralickFeatures::SaveImages(string prefix){
    typedef itk::ImageFileWriter<ImageType>   WriterType;
    typename WriterType::Pointer writer = WriterType::New();

    typedef itk::RescaleIntensityImageFilter<ImageType> RescaleFilterType;
    typename RescaleFilterType::Pointer rescale = RescaleFilterType::New();

    rescale->SetOutputMinimum(0);
    rescale->SetOutputMaximum(255);

    unsigned int numberOfOutputs = 8;
    for(unsigned int i=0; i<numberOfOutputs; i++){
        rescale->SetInput(this->outputImages[i]);
        rescale->Update();
        writer->SetInput(rescale->GetOutput());
        writer->SetFileName( prefix+"_"+this->featureNames[i]+".nii.gz" );
        writer->Update();
    }
}
