cmake_minimum_required(VERSION 3.5)

project(TiCCE)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "http://slicer.org/slicerWiki/index.php/Documentation/Nightly/Extensions/TiCCE")
set(EXTENSION_CATEGORY "CSIM")
set(EXTENSION_CONTRIBUTORS "Fabricio H Simozo (University of São Paulo)")
set(EXTENSION_DESCRIPTION "Extension containing modules for characterizing and classifying tissues from images.")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/TiCCE.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/TiCCE/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a space separated string, a list or 'NA' if any

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(TiCCE)
add_subdirectory(TextureMapExtractor)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
