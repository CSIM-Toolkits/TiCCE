#-----------------------------------------------------------------------------
set(MODULE_NAME TiCCE)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

file(GLOB TiCCE_FeatureTemplates_DATA RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "Resources/feature_templates/*.nii.gz")
file(GLOB TiCCE_ClassifiersJoblibs_DATA RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "Resources/classifier_joblibs/*.joblib")
set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  ${TiCCE_FeatureTemplates_DATA}
  ${TiCCE_ClassifiersJoblibs_DATA}
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
