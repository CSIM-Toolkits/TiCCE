import os
import unittest
import sys
import platform
import vtk, qt, ctk, slicer
import SimpleITK as sitk
from slicer.ScriptedLoadableModule import *
import logging

#
# TiCCE
#

class TiCCE(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "TiCCE"
    self.parent.categories = ["CSIM"]
    self.parent.dependencies = []
    self.parent.contributors = ["Fabricio H Simozo (University of Sao Paulo)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
This scripted module is the main module for Tissue Characterization and Classification Extension (TiCCE).
From this module, the full extension pipeline can be called, as well as any specific step.
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
This file was originally developed by Fabricio Henrique Simozo, CSIM, University of Sao Paulo,
and the development of the steps in this pipeline were part of the PhD project from Simozo.
""" # replace with organization, grant and thanks.

#
# TiCCEWidget
#

class TiCCEWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...

    self.stackedWidget = qt.QStackedWidget()
    self.layout.addWidget(self.stackedWidget)

    # Add vertical spacer
    self.layout.addStretch(1)

    #
    # Initialization Area
    #
    initialPage = slicer.qSlicerWidget()
    self.stackedWidget.addWidget(initialPage)
    # Layout within the dummy slicer widget page
    initialFormLayout = qt.QFormLayout(initialPage)
    initialLabel = qt.QLabel("Select which steps you would like to perform:")
    initialFormLayout.addRow(initialLabel)

    #
    # input volume selector
    #
    self.charInputSelector = slicer.qMRMLNodeComboBox()
    self.charInputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.charInputSelector.selectNodeUponCreation = True
    self.charInputSelector.addEnabled = False
    self.charInputSelector.removeEnabled = False
    self.charInputSelector.noneEnabled = True
    self.charInputSelector.showHidden = False
    self.charInputSelector.showChildNodeTypes = False
    self.charInputSelector.setMRMLScene(slicer.mrmlScene)
    self.charInputSelector.setToolTip( "Pick the input to the algorithm" )
    self.charInputSelectorText = qt.QLabel("Input Volume: ")
    initialFormLayout.addRow(self.charInputSelectorText, self.charInputSelector)

    #
    # Testing input thickness selector
    #
    self.testInputThicknessSelector = slicer.qMRMLNodeComboBox()
    self.testInputThicknessSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.testInputThicknessSelector.selectNodeUponCreation = True
    self.testInputThicknessSelector.addEnabled = False
    self.testInputThicknessSelector.removeEnabled = False
    self.testInputThicknessSelector.noneEnabled = True
    self.testInputThicknessSelector.showHidden = False
    self.testInputThicknessSelector.showChildNodeTypes = False
    self.testInputThicknessSelector.setMRMLScene(slicer.mrmlScene)
    self.testInputThicknessSelector.setToolTip("Pick the input image for thickness obtained from input image with ANTS algorithm."
                                               "MUST BE IN THE SAME SPACE AS INPUT IMAGE")
    self.testInputThicknessSelectorText = qt.QLabel("Input Thickness: ")
    initialFormLayout.addRow(self.testInputThicknessSelectorText, self.testInputThicknessSelector)

    #
    # Testing input label selector
    #
    self.testInputLabelsSelector = slicer.qMRMLNodeComboBox()
    self.testInputLabelsSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
    self.testInputLabelsSelector.selectNodeUponCreation = True
    self.testInputLabelsSelector.addEnabled = False
    self.testInputLabelsSelector.removeEnabled = False
    self.testInputLabelsSelector.noneEnabled = True
    self.testInputLabelsSelector.showHidden = False
    self.testInputLabelsSelector.showChildNodeTypes = False
    self.testInputLabelsSelector.setMRMLScene(slicer.mrmlScene)
    self.testInputLabelsSelector.setToolTip("Pick the input labels that will be used during testing. ")
    self.testInputLabelsSelectorText = qt.QLabel("Input Labels: ")
    initialFormLayout.addRow(self.testInputLabelsSelectorText, self.testInputLabelsSelector)

    #
    # Testing output label selector
    #
    self.outputLabelSelector = slicer.qMRMLNodeComboBox()
    self.outputLabelSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
    self.outputLabelSelector.selectNodeUponCreation = True
    self.outputLabelSelector.addEnabled = True
    self.outputLabelSelector.removeEnabled = True
    self.outputLabelSelector.noneEnabled = False
    self.outputLabelSelector.showHidden = False
    self.outputLabelSelector.showChildNodeTypes = False
    self.outputLabelSelector.setMRMLScene(slicer.mrmlScene)
    self.outputLabelSelector.setToolTip("Pick the output label to be used as result.")
    self.outputLabelSelectorText = qt.QLabel("Output Label: ")
    initialFormLayout.addRow(self.outputLabelSelectorText, self.outputLabelSelector)

    self.runPipelineButton = qt.QPushButton("Run")
    self.runPipelineButton.toolTip = "Run defined pipeline."
    self.runPipelineButton.enabled = True

    navigationArea = slicer.qSlicerWidget()
    self.navigationFormLayout = qt.QHBoxLayout(navigationArea)
    self.navigationFormLayout.addWidget(self.runPipelineButton)

    self.layout.addWidget(navigationArea)

    # Connections
    self.runPipelineButton.connect("clicked(bool)", self.onRunButton)

    # self.loadTrainedClassifier.connect("clicked(bool)", self.updateParameters)

  def cleanup(self):
    pass

  def onRunButton(self):
    logic = TiCCELogic()
    logic.run(self.charInputSelector.currentNode(),
              self.testInputThicknessSelector.currentNode(),
              self.testInputLabelsSelector.currentNode(),
              self.outputLabelSelector.currentNode())

# TiCCELogic
#

class TiCCELogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def isValidInputData(self, inputVolumeNodes):
    """Validates if the output is not the same as input
    """
    for node in inputVolumeNodes:
      if not self.isValidInputNode(node):
        return False

    return True

  def isValidInputNode(self, node):
    """Validates if the output is not the same as input
    """
    if not node:
      logging.debug('isValidInputData failed: no input volume node defined')
      return False

    if node.GetImageData() is None:
      logging.debug('isValidInputData failed: no image data in volume node')

    return True

  def run(self, charVolume, testThicknessVolume, testLabel, outputLabel):
    """
    Run characterization step
    """

    # Checks if parameters and conditions are set to run algorithm
    # Tests characterization input nodes:
    if not charVolume:
      slicer.util.errorDisplay('No input volumes were selected.')
      return False
    if not self.isValidInputNode(charVolume):
      slicer.util.errorDisplay('Characterization selected volumes is invalid.')
      return False
    if not outputLabel:
      slicer.util.errorDisplay('No output volume was selected.')
      return False

    # Default parameters
    doHaralick = True
    doHistogram = False
    logging.info('Processing started')

    # Runs characterization step ******************************
    vectorFeatureNodes = slicer.vtkMRMLVectorVolumeNode()
    slicer.mrmlScene.AddNode(vectorFeatureNodes)

    listFeatures = []
    listZFeatures = []
    if testThicknessVolume is None:
      doWithThickness = False
    else:
      doWithThickness = True

    self.runCharacterizationCLI(charVolume, vectorFeatureNodes, doHaralick, doHistogram)

    # Creates list from vector image
    self.createListFromVectorImage(vectorFeatureNodes, listFeatures)

    # Runs Z-Score step ***************************************
    self.generateZScore(charVolume, listFeatures, testThicknessVolume, listZFeatures)

    # Creates list with all features in correct order for classifier
    listFull = []
    self.composeListWithAllFeatures(listFeatures, listZFeatures, doWithThickness, listFull)

    # Check for dependencies
    # from pip._internal import main as pipmain
    import subprocess

    pip_modules = [['scipy','scipy'],['scikit-learn', 'sklearn'],['joblib', 'joblib']]
    for module in pip_modules:
      try:
        module_obj = __import__(module[1])
        logging.info("{0} was found. Successfully imported".format(module[0]))
      except ImportError:
        logging.info("{0} was not found. Attempting to install {0}.".format(module[0]))
        # pipmain(['install', '--user', module[0]])
        subprocess.call([sys.executable, "-m", "pip", "install", "--user", module[0]])

    self.doClassification(listFull, testLabel, outputLabel, doWithThickness)


    logging.info('Processing completed')

    return True

  def runCharacterizationCLI(self, inputNode, vectorFeatureNodes, doHaralick, doHistogram):
    print("Running characterization step")

    params = {}
    params["inputVolume"] = inputNode.GetID()
    params["outputVolume"] = vectorFeatureNodes.GetID()
    params["doHaralick"] = doHaralick
    params["doHistogram"] = doHistogram
    params["window_size"] = "3,3,1"

    slicer.cli.run(slicer.modules.texturemapextractor, None, params, wait_for_completion=True)

  def createListFromVectorImage(self, inputVectorNode, list):
    print("Creating list from vectorNode")

    inputImage = inputVectorNode.GetImageData()

    listFeatureName = [
      "energy",
      "entropy",
      "correlation",
      "inv_diff_moment",
      "inertia",
      "cluster_shade",
      "cluster_prom",
      "har_correlation"
    ]

    for i in range(inputImage.GetNumberOfScalarComponents()):
      print("Processing return from vector to list: " + str(i) + "...")
      outputNode = slicer.vtkMRMLScalarVolumeNode()
      slicer.mrmlScene.AddNode(outputNode)

      ijkToRAS = vtk.vtkMatrix4x4()
      inputVectorNode.GetIJKToRASMatrix(ijkToRAS)
      outputNode.SetIJKToRASMatrix(ijkToRAS)

      extract = vtk.vtkImageExtractComponents()
      extract.SetInputConnection(inputVectorNode.GetImageDataConnection())
      extract.SetComponents(i)
      extract.Update()
      outputNode.SetImageDataConnection(extract.GetOutputPort())

      list.append((listFeatureName[i], outputNode))

  def generateZScore(self, inputVolumeNode, listFeatures, thicknessVolumeNode, listZFeatures):
    print("Doing features step:")
    print("Registration:")
    # Load feature templates

    # To hold transform
    regMNItoRefTransform = slicer.vtkMRMLBSplineTransformNode()
    slicer.mrmlScene.AddNode(regMNItoRefTransform)

    print("Registering flair image...")
    outputFlair = slicer.vtkMRMLScalarVolumeNode()
    slicer.mrmlScene.AddNode(outputFlair)
    self.registerAndExtractZ(inputVolumeNode, outputFlair, "flair", regMNItoRefTransform, True)

    for featureNode in listFeatures:
      print("Registering " + featureNode[0] + "...")

      output = slicer.vtkMRMLScalarVolumeNode()
      slicer.mrmlScene.AddNode(output)
      self.registerAndExtractZ(featureNode[1], output, featureNode[0], regMNItoRefTransform, False)

      # Add to vectorFeatureNodes
      # vectorFeatureNodes.append(output)
      listZFeatures.append((featureNode[0] + "_z", output))

    # vectorFeatureNodes.append(inputVolumeNode)
    # vectorFeatureNodes.append(outputFlair)
    listFeatures.append(("flair", inputVolumeNode))
    listZFeatures.append(("flair_z", outputFlair))

    if thicknessVolumeNode is not None:
      print("Processing thickness...")
      outputThick = slicer.vtkMRMLScalarVolumeNode()
      slicer.mrmlScene.AddNode(outputThick)
      self.registerAndExtractZ(thicknessVolumeNode, outputThick, "thickness", regMNItoRefTransform, False)

      listFeatures.append(("thickness", thicknessVolumeNode))
      listZFeatures.append(("thickness_z", outputThick))

  def registerAndExtractZ(self, inputNode, outputNode, featureName, regMNItoRefTransform, generateRegistration):
    print("Register and extract Z...")

    modulePath = os.path.dirname(slicer.modules.ticce.path)
    import sitkUtils
    if platform.system() is "Windows":
      separator = "\\"
    else:
      separator = "/"

    templatePath = modulePath + separator + "Resources" + separator + "feature_templates"

    (readSuccess, meanNode) = slicer.util.loadVolume(templatePath + separator + featureName + "_mean.nii.gz", {}, True)
    (readSuccess, stdNode) = slicer.util.loadVolume(templatePath + separator + featureName + "_std.nii.gz", {}, True)

    mniMeanRef = slicer.vtkMRMLScalarVolumeNode()
    slicer.mrmlScene.AddNode(mniMeanRef)
    mniStdRef = slicer.vtkMRMLScalarVolumeNode()
    slicer.mrmlScene.AddNode(mniStdRef)

    if generateRegistration:
      # Get 0-255 images to perform registration:
      inputNodeResc = slicer.vtkMRMLScalarVolumeNode()
      slicer.mrmlScene.AddNode(inputNodeResc)
      self.rescaleImage(inputNode, inputNodeResc, 0, 255)

      meanNodeResc = slicer.vtkMRMLScalarVolumeNode()
      slicer.mrmlScene.AddNode(meanNodeResc)
      self.rescaleImage(meanNode, meanNodeResc, 0, 255)

      self.doNonLinearRegistration(inputNodeResc, meanNodeResc, None, regMNItoRefTransform, 0.10, "3,3,3", "useCenterOfHeadAlign", -1)

      slicer.mrmlScene.RemoveNode(inputNodeResc)
      slicer.mrmlScene.RemoveNode(meanNodeResc)

    self.applyRegistrationTransform(meanNode, inputNode, mniMeanRef, regMNItoRefTransform, False, False)

    self.applyRegistrationTransform(stdNode, inputNode, mniStdRef, regMNItoRefTransform, False, False)

    # Subtract and divide mean images
    featureImg = sitkUtils.PullVolumeFromSlicer(inputNode)
    mniMeanRefImg = sitkUtils.PullVolumeFromSlicer(mniMeanRef)
    mniStdRefImg = sitkUtils.PullVolumeFromSlicer(mniStdRef)

    # Prepare image types

    if featureImg.GetPixelIDTypeAsString() != "32-bit float":
      featureImg = sitk.Cast(featureImg, sitk.sitkFloat32)

    if mniMeanRefImg.GetPixelIDTypeAsString() != "32-bit float":
      mniMeanRefImg = sitk.Cast(mniMeanRefImg, sitk.sitkFloat32)

    if mniStdRefImg.GetPixelIDTypeAsString() != "32-bit float":
      mniStdRefImg = sitk.Cast(mniStdRefImg, sitk.sitkFloat32)

    zFeatureImg = (featureImg - mniMeanRefImg) / mniStdRefImg

    sitkUtils.PushVolumeToSlicer(zFeatureImg, outputNode)

    slicer.mrmlScene.RemoveNode(mniMeanRef)
    slicer.mrmlScene.RemoveNode(mniStdRef)
    slicer.mrmlScene.RemoveNode(meanNode)
    slicer.mrmlScene.RemoveNode(stdNode)


  def rescaleImage(self, inputNode, outputNode, minValue, maxValue):
    print("Rescale image...")
    import sitkUtils
    import SimpleITK as sitk

    sitkImg = sitkUtils.PullVolumeFromSlicer(inputNode)
    sitkImgResc = sitk.RescaleIntensity(sitkImg, minValue, maxValue)

    sitkUtils.PushVolumeToSlicer(sitkImgResc, outputNode)

  def doNonLinearRegistration(self, fixedNode, movingNode, resultNode, transform, samplePerc, grid, initiationMethod,
                              numberOfThreads):
    print("NonLinear Registration...")

    """
    Execute the BrainsFit registration
    :param fixedNode:
    :param movingNode:
    :param resultNode:
    :return:
    """
    regParams = {}
    regParams["fixedVolume"] = fixedNode.GetID()
    regParams["movingVolume"] = movingNode.GetID()
    regParams["samplingPercentage"] = samplePerc
    regParams["splineGridSize"] = grid

    if resultNode is not None:
      regParams["outputVolume"] = resultNode.GetID()
    # regParams["linearTransform"] = transform.GetID()
    if transform is not None:
      regParams["bsplineTransform"] = transform.GetID()

    regParams["initializeTransformMode"] = initiationMethod
    # regParams["histogramMatch"] = True
    regParams["useRigid"] = True
    regParams["useAffine"] = True
    regParams["useBSpline"] = True
    regParams["numberOfThreads"] = numberOfThreads

    slicer.cli.run(slicer.modules.brainsfit, None, regParams, wait_for_completion=True)

  def applyRegistrationTransform(self, inputVolume, referenceVolume, outputVolume, warpTransform, doInverse, isLabelMap):
    """
    Execute the Resample Volume CLI
    :param inputVolume:
    :param referenceVolume:
    :param outputVolume:
    :param pixelType:
    :param warpTransform:
    :param inverseTransform:
    :param interpolationMode:
    :return:
    """
    params = {}
    params["inputVolume"] = inputVolume.GetID()
    params["referenceVolume"] = referenceVolume.GetID()
    params["outputVolume"] = outputVolume.GetID()
    params["warpTransform"] = warpTransform.GetID()
    params["inverseTransform"] = doInverse
    if isLabelMap:
      params["interpolationMode"] = "NearestNeighbor"
      params["pixelType"] = "binary"
    else:
      params["interpolationMode"] = "Linear"
      params["pixelType"] = "float"

    slicer.cli.run(slicer.modules.brainsresample, None, params, wait_for_completion=True)


  def composeListWithAllFeatures(self, listFeatures, listZFeatures, doWithThickness, listFull):
    print("Composing full feature list")
    print("Length features: ")
    print(len(listFeatures))
    print("Length z features: ")
    print(len(listZFeatures))

    tempList = listFeatures + listZFeatures

    print("Length full: ")
    print(len(tempList))

    tempList = dict(tempList)

    listFeatureName = [
      "cluster_prom",
      "cluster_shade",
      "correlation",
      "energy",
      "entropy",
      "har_correlation",
      "inertia",
      "flair",
      "inv_diff_moment",
    ]

    if doWithThickness:
      listFeatureName.append("thickness")

    listFeatureName = listFeatureName + [s + "_z" for s in listFeatureName]

    for feature in listFeatureName:
      listFull.append(tempList[feature])

    print("Length result: ")
    print(len(listFull))


  def createTestingArrays(self, testVolumes, testLabel, testingArray, doWithThickness):
    import numpy as np
    modulePath = os.path.dirname(slicer.modules.ticce.path)
    # Loads scaler from resources
    if platform.system() is "Windows":
      separator = "\\"
    else:
      separator = "/"

    #First, concatenate information from all voxels from all images in a single matrix
    featureList = []
    classList = []
    for volume in testVolumes:
      dim = volume.GetImageData().GetDimensions()
      xs = range(0, dim[0])
      ys = range(0, dim[1])
      zs = range(0, dim[2])

      import itertools
      coords = itertools.product(xs, ys, zs)

      # Iterates through all coordinates and, if there is a label in that coordinate, adds feature values and class to
      # respective arrays
      for v in coords:
        if(testLabel is not None):
          if(testLabel.GetImageData().GetScalarComponentAsFloat(v[0], v[1], v[2],0) == 0):
            continue # Only interrupts if there IS testLabel and its value equals 0

        coordFeatures = []
        for i in range(volume.GetImageData().GetNumberOfScalarComponents()):
          coordFeatures.append(volume.GetImageData().GetScalarComponentAsFloat(v[0], v[1], v[2], i))

        featureList.append(coordFeatures)

    features = np.array(featureList)

    # Loads scaler from resources
    joblibPath = modulePath + separator + "Resources" + separator + "classifier_joblibs"
    picklePath = modulePath + separator + "Resources" + separator + "classifier_pickles"

    # Load scaler
    # from joblib import load
    import pickle
    if doWithThickness:
      scaler = load(joblibPath + separator + 'scaler.joblib')
      # scaler = pickle.load(picklePath + separator + 'scaler.p')
    else:
      scaler = load(joblibPath + separator + 'scaler_no_thickness.joblib')
      # scaler = pickle.load(picklePath + separator + 'scaler_no_thickness.p')

    testingArray.append(scaler.transform(features))

  def doClassification(self, testVolumes, testLabel, resultLabel, doWithThickness):
    print("Doing classification...")

    import numpy as np
    modulePath = os.path.dirname(slicer.modules.ticce.path)
    # Loads scaler from resources
    if platform.system() is "Windows":
      separator = "\\"
    else:
      separator = "/"

    # First, concatenate information from all voxels from all images in a single matrix
    featureList = []

    # Loads scaler from resources
    from joblib import load
    # import pickle

    joblibPath = modulePath + separator + "Resources" + separator + "classifier_joblibs"
    # picklePath = modulePath + separator + "Resources" + separator + "classifier_pickles"

    if doWithThickness:
      scaler = load(joblibPath + separator + 'scaler.joblib')
      clf = load(joblibPath + separator + 'classifier.joblib')
      # scaler = pickle.load(open(picklePath + separator + 'scaler.p', 'rb'))
      # clf = pickle.load(open(picklePath + separator + 'classifier.p', 'rb'))
    else:
      scaler = load(joblibPath + separator + 'scaler_no_thickness.joblib')
      clf = load(joblibPath + separator + 'classifier_no_thickness.joblib')
      # scaler = pickle.load(open(picklePath + separator + 'scaler_no_thickness.p', 'rb'))
      # clf = pickle.load(open(picklePath + separator + 'classifier_no_thickness.p', 'rb'))

    dim = testVolumes[0].GetImageData().GetDimensions()
    xs = range(0, dim[0])
    ys = range(0, dim[1])
    zs = range(0, dim[2])

    import itertools
    coords = itertools.product(xs, ys, zs)

    print("Length of list: ")
    print(len(testVolumes))

    # Set output image
    resultLabel.SetOrigin(testLabel.GetOrigin())
    resultLabel.SetSpacing(testLabel.GetSpacing())
    import sitkUtils

    tempSitk = sitkUtils.PullVolumeFromSlicer(testLabel)
    sitkUtils.PushVolumeToSlicer(tempSitk, resultLabel)


    # Iterates through all coordinates and, if there is a label in that coordinate, adds feature values and class to
    # respective arrays
    indexList = []
    testList = []
    for v in coords:
      if (testLabel is not None):
        if (testLabel.GetImageData().GetScalarComponentAsFloat(v[0], v[1], v[2], 0) == 0):
          continue  # Only interrupts if there IS testLabel and its value equals 0

      coordFeatures = []
      for volume in testVolumes:
        value = volume.GetImageData().GetScalarComponentAsFloat(v[0], v[1], v[2], 0)
        if np.isnan(value):
          value = 0
        elif value > 10000:
          value = 10000
        elif value < -10000:
          value = -10000

        coordFeatures.append(value)

      indexList.append([v[0], v[1], v[2]])
      testList.append(coordFeatures)

    print("Creating test array")
    testArray = np.array(testList)

    print("Predicting for array")
    pred = clf.predict(scaler.transform(testArray))

    print("Populating output label mask")
    for p, v in zip(pred, indexList):
      resultLabel.GetImageData().SetScalarComponentFromFloat(v[0], v[1], v[2], 0, p)

class TiCCETest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_TiCCE1()

  def test_TiCCE1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    import urllib
    downloads = (
      ('http://slicer.kitware.com/midas3/download?items=5767', 'FA.nrrd', slicer.util.loadVolume),
    )

    for url,name,loader in downloads:
      filePath = slicer.app.temporaryPath + '/' + name
      if not os.path.exists(filePath) or os.stat(filePath).st_size == 0:
        logging.info('Requesting download %s from %s...\n' % (name, url))
        urllib.urlretrieve(url, filePath)
      if loader:
        logging.info('Loading %s...' % (name,))
        loader(filePath)
    self.delayDisplay('Finished with download and loading')

    volumeNode = slicer.util.getNode(pattern="FA")
    logic = TiCCELogic()
    self.assertIsNotNone( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')
