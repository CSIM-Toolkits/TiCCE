#include "itkImageFileWriter.h"

#include "itkSmoothingRecursiveGaussianImageFilter.h"

#include "itkPluginUtilities.h"

#include "TextureMapExtractorCLP.h"

#include <itkImageRegionIterator.h>
#include <itkImageConstIterator.h>

#include <itkImageDuplicator.h>
#include <itkImageFileWriter.h>
#include <itkCastImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkComposeImageFilter.h>

#include "Texture/HaralickFeatures.h"
#include "Texture/HistogramFeatures.h"

#include <iostream>
#include <sstream>

#include <ctime>

// Use an anonymous namespace to keep class types and function names
// from colliding when module is used as shared object module.  Every
// thing should be in an anonymous namespace except for the module
// entry point, e.g. main()
//
namespace
{

template <typename TPixel>
int DoIt( int argc, char * argv[], TPixel )
{
    PARSE_ARGS;

    //Checks if at least one feature set is selected
    if(!doHaralick && !doHistogram){
        cout<<"ERROR: NO FEATURE SET WAS SELECTED. USE --help TO READ DESCRIPTION. AT LEAST ONE SHOULD BE USED."<<endl;
        return 1;
    }

    //Checks if no output methods are defined. If then, refer to the --help command, as both shouldn't be used.
    bool writeIndividually = (savingPrefix != "")?true:false;
    bool writeVectorImage = (string(outputVolume.c_str()) != "")?true:false;

    if(!writeIndividually && !writeVectorImage){
        cout<<"ERROR: NO OUTPUT METHODS WERE DEFINED. USE --help TO READ DESCRIPTION. AT LEAST ONE METHOD SHOULD BE USED."<<endl;
        return 1;
    }

    typedef    TPixel InputPixelType;

    typedef itk::Image<InputPixelType,  3> InputImageType;
    typedef itk::Image<double, 3>          ImageType;
    typedef itk::VectorImage<double, 3>    VectorImageType;
    typedef itk::Image<unsigned char, 3>   UCharImageType;

    typedef itk::ImageFileReader<InputImageType>  InputReaderType;

    typename InputReaderType::Pointer reader = InputReaderType::New();
    itk::PluginFilterWatcher watchReader(reader, "Read Origninal Volume",
                                          CLPProcessInformation);

    reader->SetFileName( inputVolume.c_str() );

    reader->Update();

    //For time measurements
    clock_t begin, end;

    /*###################################################################################
    * Pre processing.
    */

    typedef itk::CastImageFilter<InputImageType, ImageType> CastFilterType;
    typedef itk::CastImageFilter<InputImageType, UCharImageType> UCharCastFilterType;
    typename CastFilterType::Pointer castImage = CastFilterType::New();
    typename UCharCastFilterType::Pointer castMask = UCharCastFilterType::New();

    //Casts and saves the input image
    castImage->SetInput(reader->GetOutput());
    castImage->Update();

    //Casts and duplicate the input map
    typedef itk::RescaleIntensityImageFilter<InputImageType, InputImageType> RescaleFilterType;
    typename RescaleFilterType::Pointer rescale = RescaleFilterType::New();
    rescale->SetInput(reader->GetOutput());
    rescale->SetOutputMinimum(0);
    rescale->SetOutputMaximum(255);

    castMask->SetInput(rescale->GetOutput());
    castMask->Update();

    cout<<endl;

    //Defines vector image for possible output
    std::vector<ImageType::Pointer> features;

    /*###################################################################################
    * 1 step: Computes the haralick features for the image. The result is either returned as image maps
    * saved in the defined path directory or as specified individual output paths.
    */
    if(doHaralick){
        cout<<"**************************"<<endl<<"Doing Haralick step"<<endl;

        begin = clock();

        HaralickFeatures* haralick = new HaralickFeatures(castImage->GetOutput(), castMask->GetOutput(), window_size);
        haralick->Run();

        if(writeIndividually)
            haralick->SaveImages(savingPrefix);

        if(writeVectorImage){
            std::vector<ImageType::Pointer> harFeatures = haralick->GetOutput();

            features.insert(features.end(), harFeatures.begin(), harFeatures.end());
        }

        end = clock();
        double time = (double)(end-begin)/CLOCKS_PER_SEC;
        cout<<"Elapsed time = "<<time<<" seg = "<<time/60.0<<" min"<<endl;
    }

    /*###################################################################################
    * 2 step: Computes the histogram features for the image. The result is either returned as image maps
    * saved in the defined path directory or as specified individual output paths.
    */
    if(doHistogram){
        cout<<"**************************"<<endl<<"Doing histogram step"<<endl;

        begin = clock();

        HistogramFeatures* histogram = new HistogramFeatures(castImage->GetOutput(), castMask->GetOutput(), window_size);
        histogram->Run();

        if(writeIndividually)
            histogram->SaveImages(savingPrefix);

        if(writeVectorImage){
            std::vector<ImageType::Pointer> histFeatures = histogram->GetOutput();

            features.insert(features.end(), histFeatures.begin(), histFeatures.end());
        }

        end = clock();
        double time = (double)(end-begin)/CLOCKS_PER_SEC;
        cout<<"Elapsed time = "<<time<<" seg = "<<time/60.0<<" min"<<endl;
    }

    if(writeVectorImage){
        typedef itk::ComposeImageFilter<ImageType> ComposeFilterType;
        typename ComposeFilterType::Pointer compose = ComposeFilterType::New();

        unsigned int size = features.size();

        for(unsigned int i = 0; i<size; i++)
            compose->SetInput(i,features[i]);

        compose->Update();

        typedef itk::ImageFileWriter<VectorImageType> VectorWriterType;
        typename VectorWriterType::Pointer vecWriter = VectorWriterType::New();

        vecWriter->SetInput(compose->GetOutput());
        vecWriter->SetFileName(outputVolume.c_str());
        vecWriter->Update();;
    }

    return EXIT_SUCCESS;
}

} // end of anonymous namespace

int main( int argc, char * argv[] )
{
    PARSE_ARGS;

    itk::ImageIOBase::IOPixelType     pixelType;
    itk::ImageIOBase::IOComponentType componentType;

    try
    {
        itk::GetImageType(inputVolume, pixelType, componentType);

        // This filter handles all types on input, but only produces
        // signed types
        switch( componentType )
        {
        case itk::ImageIOBase::UCHAR:
            return DoIt( argc, argv, static_cast<unsigned char>(0) );
            break;
        case itk::ImageIOBase::CHAR:
            return DoIt( argc, argv, static_cast<signed char>(0) );
            break;
        case itk::ImageIOBase::USHORT:
            return DoIt( argc, argv, static_cast<unsigned short>(0) );
            break;
        case itk::ImageIOBase::SHORT:
            return DoIt( argc, argv, static_cast<short>(0) );
            break;
        case itk::ImageIOBase::UINT:
            return DoIt( argc, argv, static_cast<unsigned int>(0) );
            break;
        case itk::ImageIOBase::INT:
            return DoIt( argc, argv, static_cast<int>(0) );
            break;
        case itk::ImageIOBase::ULONG:
            return DoIt( argc, argv, static_cast<unsigned long>(0) );
            break;
        case itk::ImageIOBase::LONG:
            return DoIt( argc, argv, static_cast<long>(0) );
            break;
        case itk::ImageIOBase::FLOAT:
            return DoIt( argc, argv, static_cast<float>(0) );
            break;
        case itk::ImageIOBase::DOUBLE:
            return DoIt( argc, argv, static_cast<double>(0) );
            break;
        case itk::ImageIOBase::UNKNOWNCOMPONENTTYPE:
        default:
            std::cerr << "Unknown input image pixel component type: ";
            std::cerr << itk::ImageIOBase::GetComponentTypeAsString( componentType );
            std::cerr << std::endl;
            return EXIT_FAILURE;
            break;
        }
    }

    catch( itk::ExceptionObject & excep )
    {
        std::cerr << argv[0] << ": exception caught !" << std::endl;
        std::cerr << excep << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
