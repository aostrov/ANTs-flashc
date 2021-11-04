
include(${CMAKE_CURRENT_LIST_DIR}/Common.cmake)

configure_file(${CMAKE_CURRENT_LIST_DIR}/CTestCustom.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/CTestCustom.cmake COPYONLY)

set(CMAKE_MODULE_PATH
    ${${PROJECT_NAME}_SOURCE_DIR}/CMake
    ${${PROJECT_NAME}_BINARY_DIR}/CMake
    ${CMAKE_MODULE_PATH}
    )

set (CMAKE_INCLUDE_DIRECTORIES_BEFORE ON)
#-----------------------------------------------------------------------------
# Version information
option( ${PROJECT_NAME}_BUILD_DISTRIBUTE "Remove '-g#####' from version. ( for official distribution only )" OFF )
mark_as_advanced( ${PROJECT_NAME}_BUILD_DISTRIBUTE )

include(Version.cmake)

#-----------------------------------------------------------------------------
# CPACK Version
#
set(CPACK_PACKAGE_NAME "ANTs")
set(CPACK_PACKAGE_VENDOR "CMake.org")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "ANTs - Advanced Normalization Tools")
set(CPACK_PACKAGE_VERSION_MAJOR ${${PROJECT_NAME}_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${${PROJECT_NAME}_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${${PROJECT_NAME}_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION ${${PROJECT_NAME}_VERSION})
set(CPACK_PACKAGE_INSTALL_DIRECTORY "ANTS_Install")
set(CPACK_BINARY_GENERATORS "DragNDrop TGZ TZ")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.txt")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "ANTs - robust image registration, segmentation and more")
set(CPACK_RESOURCE_FILE_LICENSE  "${CMAKE_CURRENT_SOURCE_DIR}/ANTSCopyright.txt")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.txt")

message(STATUS "Building ${PROJECT_NAME} version \"${${PROJECT_NAME}_VERSION}\"")

# Set up ITK
find_package(ITK ${ITK_VERSION_ID} REQUIRED)
include(${ITK_USE_FILE})
# FLASH EDIT
# TODO: still needs to be improved; paths should be exposed when running ccmake and required for proper config
#       if PyCAConfig.cmake is not found, ccmake should complain and fail the configure
#       eventually PyCA should be treated exactly like ITK in the way it is found and included
# looks for PyCA bin directory
# set(PyCA_DIR "" CACHE PATH "Path to PyCA build directory, where PyCAConfig.cmake is located")
# set(PyCA_SOURCE "" CACHE PATH "Path to PyCA source directory")
IF(PyCA_DIR)
  INCLUDE(${PyCA_DIR}/PyCAConfig.cmake)
  INCLUDE_DIRECTORIES(${PyCA_INCLUDE_DIRECTORIES})
ELSE(PyCA_DIR)
  MESSAGE("PyCA_DIR (binary dir for compiled PyCA library) required but missing")
ENDIF(PyCA_DIR)

# == Setup PyCA Include Directories (files missing in PyCA_DIR): ==
set(PyCA_SOURCE "" CACHE FILEPATH "PyCA source code path")
LINK_DIRECTORIES( ${PyCA_SOURCE} )
set(PyCA_SOURCE "" CACHE FILEPATH "PyCA header file path")
If(PyCA_SOURCE)
  LIST(APPEND PyCA_INCLUDE_DIRECTORIES
       ${PyCA_SOURCE}/Code/Cxx/inc/types
       ${PyCA_SOURCE}/Code/Cxx/inc/math
       ${PyCA_SOURCE}/Code/Cxx/inc/alg
       ${PyCA_SOURCE}/Code/Cxx/inc/io
       ${PyCA_SOURCE}/Code/Cxx/src/math
  )
  INCLUDE_DIRECTORIES(${PyCA_INCLUDE_DIRECTORIES})
ELSE(PyCA_SOURCE)
  MESSAGE("PyCA source code directory is required but missing")
ENDIF(PyCA_SOURCE)
# END: FLASH EDIT

# Set up which ANTs apps to build
option(BUILD_ALL_ANTS_APPS "Use All ANTs Apps" ON)

# Set up VTK
option(USE_VTK "Use VTK Libraries" OFF)
if(USE_VTK)
  find_package(VTK)

  if(VTK_VERSION_MAJOR GREATER 6)
    find_package(VTK 8.1.1 COMPONENTS vtkRenderingVolumeOpenGL2
   vtkCommonCore
   vtkCommonDataModel
   vtkIOGeometry
   vtkIOXML
   vtkIOLegacy
   vtkIOPLY
   vtkFiltersModeling
   vtkImagingStencil
   vtkImagingGeneral
   vtkRenderingAnnotation
   vtkFiltersExtraction
   )
  endif()

  if(VTK_FOUND)
    include(${VTK_USE_FILE})
  else()
     message("Cannot build some programs without VTK.  Please set VTK_DIR if you need these programs.")
  endif()
endif()

# With MS compilers on Win64, we need the /bigobj switch, else generated
# code results in objects with number of sections exceeding object file
# format.
# see http://msdn.microsoft.com/en-us/library/ms173499.aspx
if(CMAKE_CL_64 OR MSVC)
  add_definitions(/bigobj)
endif()

option(ITK_USE_FFTWD "Use double precision fftw if found" OFF)
option(ITK_USE_FFTWF "Use single precision fftw if found" OFF)
option(ITK_USE_SYSTEM_FFTW "Use an installed version of fftw" OFF)
if (ITK_USE_FFTWD OR ITK_USE_FFTWF)
  if(ITK_USE_SYSTEM_FFTW)
      find_package( FFTW )
      link_directories(${FFTW_LIBDIR})
  else()
      link_directories(${ITK_DIR}/fftw/lib)
      include_directories(${ITK_DIR}/fftw/include)
  endif()
endif()

# FLASH EDIT
option(USE_FFTWD "Use double precision fftw if found" OFF)
option(USE_FFTWF "Use single precision fftw if found" ON)
if (USE_FFTWD OR USE_FFTWF)
    find_package( FFTW )
    link_directories(${FFTW_LIBDIR})
endif()
# END: FLASH EDIT
# These are configure time options that specify which
# subset of tests should be run
option(RUN_SHORT_TESTS    "Run the quick unit tests."                                   ON  )
option(RUN_LONG_TESTS     "Run the time consuming tests. i.e. real world registrations" OFF )
option(OLD_BASELINE_TESTS "Use reported metrics from old tests"                         OFF )
#-----------------------------------------------------------------------------
include(CTest)
enable_testing()
#Set the global max TIMEOUT for CTest jobs.  This is very large for the moment
#and should be revisted to reduce based on "LONG/SHORT" test times, set to 1 hr for now
set(CTEST_TEST_TIMEOUT 1800 CACHE STRING "Maximum seconds allowed before CTest will kill the test." FORCE)
set(DART_TESTING_TIMEOUT ${CTEST_TEST_TIMEOUT} CACHE STRING "Maximum seconds allowed before CTest will kill the test." FORCE)

configure_file(${CMAKE_CURRENT_LIST_DIR}/CTestCustom.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/CTestCustom.cmake COPYONLY)

include_directories( ${BOOST_INCLUDE_DIR} ) #Define where to find Boost includes
link_directories( ${ITK_LIBRARY_PATH}  )
# message("${ITK_LIBRARIES}")

#----------------------------------------------------------------------------
# Setup ants build environment
set(PICSL_INCLUDE_DIRS
${CMAKE_CURRENT_SOURCE_DIR}/Utilities
${CMAKE_CURRENT_SOURCE_DIR}/ImageRegistration
${CMAKE_CURRENT_SOURCE_DIR}/ImageSegmentation
#  ${CMAKE_CURRENT_SOURCE_DIR}/GraphTheory
${CMAKE_CURRENT_SOURCE_DIR}/Tensor
${CMAKE_CURRENT_SOURCE_DIR}/Temporary
${CMAKE_CURRENT_SOURCE_DIR}/Examples
${CMAKE_CURRENT_BINARY_DIR}
${CMAKE_CURRENT_SOURCE_DIR}/flash  # FLASH EDIT
)
include_directories(${PICSL_INCLUDE_DIRS})

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/ANTsVersionConfig.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/ANTsVersionConfig.h" @ONLY IMMEDIATE)

add_subdirectory(Examples)

if (NOT ANTS_INSTALL_LIBS_ONLY) 
  install(PROGRAMS Scripts/ANTSpexec.sh
     Scripts/antsASLProcessing.sh
     Scripts/antsAtroposN4.sh
     Scripts/antsBOLDNetworkAnalysis.R
     Scripts/antsBrainExtraction.sh
     Scripts/antsCorticalThickness.sh
     Scripts/antsIntermodalityIntrasubject.sh
     Scripts/antsIntroduction.sh
     Scripts/antsLaplacianBoundaryCondition.R
     Scripts/antsLongitudinalCorticalThickness.sh
     Scripts/antsJointLabelFusion.sh
     Scripts/antsMultivariateTemplateConstruction.sh
     Scripts/antsMultivariateTemplateConstruction2.sh
     Scripts/antsNetworkAnalysis.R
     Scripts/antsNeuroimagingBattery
     Scripts/antsRegistrationSyN.sh
     Scripts/antsRegistrationSyNQuick.sh
     Scripts/waitForPBSQJobs.pl
     Scripts/waitForSGEQJobs.pl
     Scripts/waitForXGridJobs.pl
     Scripts/waitForSlurmJobs.pl
                DESTINATION bin
                PERMISSIONS  OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
                CONFIGURATIONS  Release
                COMPONENT SCRIPTS
     )
endif()

#Only install ITK/VTK libraries if shared build and superbuild is used
if(BUILD_SHARED_LIBS AND ((NOT USE_SYSTEM_ITK) OR ((NOT USE_SYSTEM_VTK) AND USE_VTK)))
  install(DIRECTORY ${CMAKE_BINARY_DIR}/../staging/lib/
          DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
