#include "HistogramFeatures.h"

#include <itkImageFileReader.h>
#include <itkImageRegionIterator.h>

#include <itkImageToHistogramFilter.h>
#include <itkRegionOfInterestImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>

#include <itkImageDuplicator.h>
#include <itkImageFileWriter.h>

#include <iostream>
#include <sstream>
//#include <ofstream.h>
#include <sys/stat.h>

HistogramFeatures::HistogramFeatures(ImageType::Pointer img, UCharImageType::Pointer mask, std::vector<int> size){
    this->img = img;
    this->mask = mask;
    this->size = size;

    this->featureNames.push_back("intensity");
    this->featureNames.push_back("mean");
    this->featureNames.push_back("std");
    this->featureNames.push_back("skewness");
    this->featureNames.push_back("kurtosis");
}

HistogramFeatures::~HistogramFeatures(){

}

void HistogramFeatures::Run(){
    //Defines types used
    typedef itk::RescaleIntensityImageFilter<ImageType>                              RescaleFilterType;
    typedef itk::RegionOfInterestImageFilter<ImageType, ImageType>                   ROIType;
    typedef itk::Statistics::ImageToHistogramFilter<UCharImageType>                  HistogramFilterType;
    typedef itk::ImageRegionIterator<ImageType>                                      IteratorType;
    typedef itk::ImageRegionIterator<UCharImageType>                                 UCharIteratorType;

    //Rescales image for values between 0 and 255
    typename RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
    rescaleFilter->SetInput(this->img);
    rescaleFilter->SetOutputMinimum(0);
    rescaleFilter->SetOutputMaximum(255);
    rescaleFilter->Update();

    //Creates output images
    unsigned int numberOfOutputs = 5;
    for (unsigned int i=0; i<numberOfOutputs; i++){
        this->outputImages.push_back(ImageType::New());
        this->outputImages[i]->CopyInformation(this->img);
        this->outputImages[i]->SetRegions(this->img->GetRequestedRegion());
        this->outputImages[i]->Allocate();
        this->outputImages[i]->FillBuffer(0);
    }

    //Creates histogram filter
    typename HistogramFilterType::Pointer histFilter = HistogramFilterType::New();
    int measurementVectorSize = 1;
    int binsPerDimension = 16;
    HistogramFilterType::HistogramType::MeasurementVectorType lowerBound(binsPerDimension);
    lowerBound.Fill(0);
    HistogramFilterType::HistogramType::MeasurementVectorType upperBound(binsPerDimension);
    upperBound.Fill(255) ;
    HistogramFilterType::HistogramType::SizeType histSize(measurementVectorSize);
    histSize.Fill(binsPerDimension);

    histFilter->SetHistogramBinMinimum( lowerBound );
    histFilter->SetHistogramBinMaximum( upperBound );
    histFilter->SetHistogramSize( histSize );

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
    IteratorType itIt (this->outputImages[0], this->outputImages[0]->GetRequestedRegion());
    IteratorType mnIt (this->outputImages[1], this->outputImages[1]->GetRequestedRegion());
    IteratorType stIt (this->outputImages[2], this->outputImages[2]->GetRequestedRegion());
    IteratorType skIt (this->outputImages[3], this->outputImages[3]->GetRequestedRegion());
    IteratorType ktIt (this->outputImages[4], this->outputImages[4]->GetRequestedRegion());

    //Runs processing
    imgIt.GoToBegin();
    while(!imgIt.IsAtEnd()){
        mapIt.SetIndex(imgIt.GetIndex());
        if(mapIt.Get()>0){
            window.SetIndex(imgIt.GetIndex());
            ImageType::RegionType region = this->img->GetRequestedRegion();

            if(region.IsInside(window)){
                roiFilter->SetRegionOfInterest(window);
                roiFilter->Update();

                IteratorType roiIt(roiFilter->GetOutput(), roiFilter->GetOutput()->GetRequestedRegion());

                double it, mn, st, sk, kt;

                //Intensity and first moment
                it=0;
                mn=0;
                int count=0;

                roiIt.GoToBegin();
                while(!roiIt.IsAtEnd()){
                    mn+=roiIt.Get();
                    it+=pow((double)roiIt.Get(),2.0);
                    ++count;
                    ++roiIt;
                }

                mn=mn/(double)count;
                it=it/(double)count;

                //Second moment
                st=0;

                roiIt.GoToBegin();
                while(!roiIt.IsAtEnd()){
                    st+=pow(((double)roiIt.Get()-mn),2.0);

                    ++roiIt;
                }
                st=(double)st/(double)count;

                //Third and fourth moment
                sk=0;
                kt=0;

                roiIt.GoToBegin();
                while(!roiIt.IsAtEnd()){
                    if(roiIt.Get()>0){
                        sk+=pow(((double)roiIt.Get()-mn),3.0)/((double)count*pow(st,3.0));
                        kt+=pow(((double)roiIt.Get()-mn),4.0)/((double)count*pow(st,4.0));
                    }
                    ++roiIt;
                }

                //Attribute values
                itIt.SetIndex(imgIt.GetIndex());
                itIt.Set(it);
                mnIt.SetIndex(imgIt.GetIndex());
                mnIt.Set(mn);
                stIt.SetIndex(imgIt.GetIndex());
                stIt.Set(st);
                skIt.SetIndex(imgIt.GetIndex());
                skIt.Set(sk);
                ktIt.SetIndex(imgIt.GetIndex());
                ktIt.Set(kt);

            }
        }
        ++imgIt;
    }
}

std::vector< HistogramFeatures::ImageType::Pointer > HistogramFeatures::GetOutput(){
    return this->outputImages;
}

void HistogramFeatures::SaveImages(string prefix){
    typedef itk::ImageFileWriter<ImageType>   WriterType;
    typename WriterType::Pointer writer = WriterType::New();

    typedef itk::RescaleIntensityImageFilter<ImageType> RescaleFilterType;
    typename RescaleFilterType::Pointer rescale = RescaleFilterType::New();

    rescale->SetOutputMinimum(0);
    rescale->SetOutputMaximum(255);

    unsigned int numberOfOutputs = 5;
    for(unsigned int i=0; i<numberOfOutputs; i++){
        rescale->SetInput(this->outputImages[i]);
        rescale->Update();
        writer->SetInput(rescale->GetOutput());
        writer->SetFileName( prefix+"_"+this->featureNames[i]+".nii.gz" );
        writer->Update();
    }
}
