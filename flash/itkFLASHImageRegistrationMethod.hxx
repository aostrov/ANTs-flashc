/*=========================================================================
 *
 *  Copyright: Greg M. Fleishman (can change to ITK if merged)
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkFLASHImageRegistrationMethod_hxx
#define itkFLASHImageRegistrationMethod_hxx

#include "itkFLASHImageRegistrationMethod.h"

#include "FftOper.h"
#include "FieldComplex3D.h"
#include "ITKFileIO.h"
#include "IOpers.h"
#include "FOpers.h"
#include "IFOpers.h"
#include "HFOpers.h"
#include "Reduction.h"
#include "FluidKernelFFT.h"

using namespace PyCA;

namespace itk
{
/**
 * Constructor
 */
template<typename TFixedImage, typename TMovingImage, typename TOutputTransform, typename TVirtualImage, typename TPointSet>
FLASHImageRegistrationMethod<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>
::FLASHImageRegistrationMethod() :
  m_LearningRate( 0.25 ),
  m_ConvergenceThreshold( 1.0e-6 ),
  m_ConvergenceWindowSize( 10 ),
  m_GaussianSmoothingVarianceForTheUpdateField( 3.0 ),
  m_GaussianSmoothingVarianceForTheTotalField( 0.5 )
{
  this->m_NumberOfIterationsPerLevel.SetSize( 3 );
  this->m_NumberOfIterationsPerLevel[0] = 20;
  this->m_NumberOfIterationsPerLevel[1] = 30;
  this->m_NumberOfIterationsPerLevel[2] = 40;
  this->m_DownsampleImagesForMetricDerivatives = true;
  this->m_MovingToMiddleTransform = ITK_NULLPTR;
  this->m_RegularizerTermWeight = 0.03;
  this->m_LaplacianWeight = 3.0;
  this->m_IdentityWeight = 1.0;
  this->m_OperatorOrder = 6.0;
  this->m_NumberOfTimeSteps = 10;
}


/**
 * Destructor
 */
template<typename TFixedImage, typename TMovingImage, typename TOutputTransform, typename TVirtualImage, typename TPointSet>
FLASHImageRegistrationMethod<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>
::~FLASHImageRegistrationMethod()
{
}


/**
 * Initialize registration objects for optimization at a specific level
 */
template<typename TFixedImage, typename TMovingImage, typename TOutputTransform, typename TVirtualImage, typename TPointSet>
void
FLASHImageRegistrationMethod<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>
::InitializeRegistrationAtEachLevel( const SizeValueType level )
{
  Superclass::InitializeRegistrationAtEachLevel( level );

  // TODO: modify names and add other objects to if and code block
  if( level == 0 )
    {
    // If velocity and displacement objects are not initialized from a state restoration
    //
    if( this->m_FixedToMiddleTransform.IsNull() )
      {
      // Initialize the FixedToMiddleTransform as an Identity displacement field transform
      //
      // TODO: example of initializing a 0 value displacement field
      // TODO: initialize FLASH related objects
      this->m_FixedToMiddleTransform = OutputTransformType::New();

      VirtualImageBaseConstPointer virtualDomainImage = this->GetCurrentLevelVirtualDomainImage();

      const DisplacementVectorType zeroVector( 0.0 );

      typename DisplacementFieldType::Pointer fixedDisplacementField = DisplacementFieldType::New();
      fixedDisplacementField->CopyInformation( virtualDomainImage );
      fixedDisplacementField->SetRegions( virtualDomainImage->GetBufferedRegion() );
      fixedDisplacementField->Allocate();
      fixedDisplacementField->FillBuffer( zeroVector );

      typename DisplacementFieldType::Pointer fixedInverseDisplacementField = DisplacementFieldType::New();
      fixedInverseDisplacementField->CopyInformation( virtualDomainImage );
      fixedInverseDisplacementField->SetRegions( virtualDomainImage->GetBufferedRegion() );
      fixedInverseDisplacementField->Allocate();
      fixedInverseDisplacementField->FillBuffer( zeroVector );

      this->m_FixedToMiddleTransform->SetDisplacementField( fixedDisplacementField );
      this->m_FixedToMiddleTransform->SetInverseDisplacementField( fixedInverseDisplacementField );
      }
    else
      {
      // TODO: modify to check for an appropriate FLASH related object instead
      if( this->m_FixedToMiddleTransform->GetInverseDisplacementField()
         && this->m_FixedToMiddleTransform->GetInverseDisplacementField() )
         {
         // TODO: learn more about adaptor - set with appropriate object
         // TODO: maybe number of fourier coefficients and spatial subsampling should be separate entities?
         // TODO: maybe you want control over both for even more increased speed?
         // TODO: will need to update interface, and also think about the consequences of changing the number of
         // TODO: fourier coefficients and the spatial resolution simultaneously, but if things are in physical
         // TODO: units, then it shouldn't be too complex
         itkDebugMacro( "FLASH registration is initialized by restoring the state.");
         this->m_TransformParametersAdaptorsPerLevel[0]->SetTransform( this->m_FixedToMiddleTransform );
         this->m_TransformParametersAdaptorsPerLevel[0]->AdaptTransformParameters();
         }
      else
        {
        itkExceptionMacro( "Invalid state restoration." );
        }
      }
    }
  else if( this->m_TransformParametersAdaptorsPerLevel[level] )
    {
    // TODO: these transform adaptors may not be necessary if I'm just augmenting the num of fourier coefficients at each level
    // TODO: instead of resampling the image resolution
    // TODO: probably just have to pad the set of fourier coefficients used to determine deformation
    this->m_TransformParametersAdaptorsPerLevel[level]->SetTransform( this->m_FixedToMiddleTransform );
    this->m_TransformParametersAdaptorsPerLevel[level]->AdaptTransformParameters();
    }
}


/*
 * Start the optimization at each level.  We just do a basic gradient descent operation.
 */
template<typename TFixedImage, typename TMovingImage, typename TOutputTransform, typename TVirtualImage, typename TPointSet>
void
FLASHImageRegistrationMethod<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>
::StartOptimization()
{
  VirtualImageBaseConstPointer virtualDomainImage = this->GetCurrentLevelVirtualDomainImage();

  if( virtualDomainImage.IsNull() )
    {
    itkExceptionMacro( "The virtual domain image is not found." );
    }

  InitialTransformType* fixedInitialTransform = const_cast<InitialTransformType*>( this->GetFixedInitialTransform() );

  // Monitor the convergence
  typedef itk::Function::WindowConvergenceMonitoringFunction<RealType> ConvergenceMonitoringType;
  typename ConvergenceMonitoringType::Pointer convergenceMonitoring = ConvergenceMonitoringType::New();
  convergenceMonitoring->SetWindowSize( this->m_ConvergenceWindowSize );

  IterationReporter reporter( this, 0, 1 );  // TODO: should maybe read about this reporter function

  while( this->m_CurrentIteration++ < this->m_NumberOfIterationsPerLevel[this->m_CurrentLevel] && !this->m_IsConverged )
    {
    typename CompositeTransformType::Pointer fixedComposite = CompositeTransformType::New();
    if ( fixedInitialTransform != ITK_NULLPTR )
      {
      fixedComposite->AddTransform( fixedInitialTransform );
      }
    fixedComposite->FlattenTransformQueue();

    typename CompositeTransformType::Pointer movingComposite = CompositeTransformType::New();
    movingComposite->AddTransform( this->m_CompositeTransform );
    // TODO: change MovingToMiddle to \phi_{1,0}
    movingComposite->AddTransform( this->m_MovingToMiddleTransform->GetInverseTransform() );
    movingComposite->FlattenTransformQueue();
    movingComposite->SetOnlyMostRecentTransformToOptimizeOn();

    // Compute the update fields (to both moving and fixed images) and smooth

    MeasureType fixedMetricValue = 0.0;
    MeasureType movingMetricValue = 0.0;

    // TODO: this one function call will encapsulate both forward/backward shooting for me, the complete gradient calculation
    DisplacementFieldPointer movingToMiddleSmoothUpdateField = this->ComputeUpdateField(
      this->m_MovingSmoothImages, this->m_MovingPointSets, movingComposite,
      this->m_FixedSmoothImages, this->m_FixedPointSets, fixedComposite,
      this->m_MovingImageMasks, this->m_FixedImageMasks, fixedMetricValue );

    // Add the update field to both displacement fields (from fixed/moving to middle image) and then smooth

    // TODO: example of how to compose displacement field with update
    // TODO: this will eventually be removed, since the total transform is constructed from shooting, but I'll
    // TODO: keep it for now as an example, because the shooting methods will use it
    typedef ComposeDisplacementFieldsImageFilter<DisplacementFieldType> ComposerType;

    typename ComposerType::Pointer movingComposer = ComposerType::New();
    movingComposer->SetDisplacementField( movingToMiddleSmoothUpdateField );
    movingComposer->SetWarpingField( this->m_MovingToMiddleTransform->GetDisplacementField() );
    movingComposer->Update();

    DisplacementFieldPointer movingToMiddleSmoothTotalFieldTmp = this->GaussianSmoothDisplacementField(
      movingComposer->GetOutput(), this->m_GaussianSmoothingVarianceForTheTotalField );

    // Iteratively estimate the inverse fields.
    // TODO: these will be unnecessary, as inverse is computed during shooting using advection equation
    DisplacementFieldPointer movingToMiddleSmoothTotalFieldInverse = this->InvertDisplacementField( movingToMiddleSmoothTotalFieldTmp, this->m_MovingToMiddleTransform->GetInverseDisplacementField() );
    DisplacementFieldPointer movingToMiddleSmoothTotalField = this->InvertDisplacementField( movingToMiddleSmoothTotalFieldInverse, movingToMiddleSmoothTotalFieldTmp );

    // Assign the displacement fields and their inverses to the proper transforms.

    this->m_MovingToMiddleTransform->SetDisplacementField( movingToMiddleSmoothTotalField );
    this->m_MovingToMiddleTransform->SetInverseDisplacementField( movingToMiddleSmoothTotalFieldInverse );

    this->m_CurrentMetricValue = fixedMetricValue;

    // TODO: this part is totally fine, just need to supply the correct energy value based on FLASH objective function
    convergenceMonitoring->AddEnergyValue( this->m_CurrentMetricValue );
    this->m_CurrentConvergenceValue = convergenceMonitoring->GetConvergenceValue();

    if( this->m_CurrentConvergenceValue < this->m_ConvergenceThreshold )
      {
      this->m_IsConverged = true;
      }
    reporter.CompletedStep();
    }
}


// TODO: update for FLASH, this method will be much more involved, as getting the update is more complex in shooting
// TODO: but I'll use subroutines of course
template<typename TFixedImage, typename TMovingImage, typename TOutputTransform, typename TVirtualImage, typename TPointSet>
typename FLASHImageRegistrationMethod<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>::DisplacementFieldPointer
FLASHImageRegistrationMethod<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>
::ComputeUpdateField( const FixedImagesContainerType fixedImages, const PointSetsContainerType fixedPointSets,
  const TransformBaseType * fixedTransform, const MovingImagesContainerType movingImages, const PointSetsContainerType movingPointSets,
  const TransformBaseType * movingTransform, const FixedImageMasksContainerType fixedImageMasks, const MovingImageMasksContainerType movingImageMasks,
  MeasureType & value )
{
  DisplacementFieldPointer metricGradientField = this->ComputeMetricGradientField(
      fixedImages, fixedPointSets, fixedTransform, movingImages, movingPointSets, movingTransform,
      fixedImageMasks, movingImageMasks, value );

  DisplacementFieldPointer updateField = this->GaussianSmoothDisplacementField( metricGradientField,
    this->m_GaussianSmoothingVarianceForTheUpdateField );

  DisplacementFieldPointer scaledUpdateField = this->ScaleUpdateField( updateField );

  return scaledUpdateField;
}


// TODO: wow, this recomputes the downsampling of the fixed image and it's mask on every single iteration... so inefficient
// TODO: also, lots of code duplication here between multiMetric and singleMetric case, could be cleaned up with private subroutine
// TODO: eventually want to look at LCC computation to ensure it uses Summed Area Tables
template<typename TFixedImage, typename TMovingImage, typename TOutputTransform, typename TVirtualImage, typename TPointSet>
typename FLASHImageRegistrationMethod<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>::DisplacementFieldPointer
FLASHImageRegistrationMethod<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>
::ComputeMetricGradientField( const FixedImagesContainerType fixedImages, const PointSetsContainerType fixedPointSets,
  const TransformBaseType * fixedTransform, const MovingImagesContainerType movingImages, const PointSetsContainerType movingPointSets,
  const TransformBaseType * movingTransform, const FixedImageMasksContainerType fixedImageMasks, const MovingImageMasksContainerType movingImageMasks,
  MeasureType & value )
{
  typename MultiMetricType::Pointer multiMetric = dynamic_cast<MultiMetricType *>( this->m_Metric.GetPointer() );

  VirtualImageBaseConstPointer virtualDomainImage = this->GetCurrentLevelVirtualDomainImage();

  if( multiMetric )
    {
    for( SizeValueType n = 0; n < multiMetric->GetNumberOfMetrics(); n++ )
      {
      if( multiMetric->GetMetricQueue()[n]->GetMetricCategory() == MetricType::POINT_SET_METRIC )
        {
        multiMetric->GetMetricQueue()[n]->SetFixedObject( fixedPointSets[n] );
        multiMetric->GetMetricQueue()[n]->SetMovingObject( movingPointSets[n] );
        multiMetric->SetFixedTransform( const_cast<TransformBaseType *>( fixedTransform ) );
        multiMetric->SetMovingTransform( const_cast<TransformBaseType *>( movingTransform ) );

        dynamic_cast<PointSetMetricType *>( multiMetric->GetMetricQueue()[n].GetPointer() )->SetCalculateValueAndDerivativeInTangentSpace( true );
        }
      else if( multiMetric->GetMetricQueue()[n]->GetMetricCategory() == MetricType::IMAGE_METRIC )
        {
        if( !this->m_DownsampleImagesForMetricDerivatives )
          {
          multiMetric->GetMetricQueue()[n]->SetFixedObject( fixedImages[n] );
          multiMetric->GetMetricQueue()[n]->SetMovingObject( movingImages[n] );

          multiMetric->SetFixedTransform( const_cast<TransformBaseType *>( fixedTransform ) );
          multiMetric->SetMovingTransform( const_cast<TransformBaseType *>( movingTransform ) );

          dynamic_cast<ImageMetricType *>( multiMetric->GetMetricQueue()[n].GetPointer() )->SetFixedImageMask( fixedImageMasks[n] );
          dynamic_cast<ImageMetricType *>( multiMetric->GetMetricQueue()[n].GetPointer() )->SetMovingImageMask( movingImageMasks[n] );
          }
        else
          {
          typedef ResampleImageFilter<FixedImageType, FixedImageType, RealType> FixedResamplerType;
          typename FixedResamplerType::Pointer fixedResampler = FixedResamplerType::New();
          fixedResampler->SetInput( fixedImages[n] );
          fixedResampler->SetTransform( fixedTransform );
          fixedResampler->UseReferenceImageOn();
          fixedResampler->SetReferenceImage( virtualDomainImage );
          fixedResampler->SetDefaultPixelValue( 0 );
          fixedResampler->Update();

          typedef ResampleImageFilter<MovingImageType, MovingImageType, RealType> MovingResamplerType;
          typename MovingResamplerType::Pointer movingResampler = MovingResamplerType::New();
          movingResampler->SetInput( movingImages[n] );
          movingResampler->SetTransform( movingTransform );
          movingResampler->UseReferenceImageOn();
          movingResampler->SetReferenceImage( virtualDomainImage );
          movingResampler->SetDefaultPixelValue( 0 );
          movingResampler->Update();

          multiMetric->GetMetricQueue()[n]->SetFixedObject( fixedResampler->GetOutput() );
          multiMetric->GetMetricQueue()[n]->SetMovingObject( movingResampler->GetOutput() );

          if( fixedImageMasks[n] )
            {
            typedef NearestNeighborInterpolateImageFunction<FixedMaskImageType, RealType> NearestNeighborInterpolatorType;
            typename NearestNeighborInterpolatorType::Pointer nearestNeighborInterpolator = NearestNeighborInterpolatorType::New();
            nearestNeighborInterpolator->SetInputImage( dynamic_cast<ImageMaskSpatialObjectType *>( const_cast<FixedImageMaskType *>( fixedImageMasks[n].GetPointer() ) )->GetImage() );

            typedef ResampleImageFilter<FixedMaskImageType, FixedMaskImageType, RealType> FixedMaskResamplerType;
            typename FixedMaskResamplerType::Pointer fixedMaskResampler = FixedMaskResamplerType::New();
            fixedMaskResampler->SetInput( dynamic_cast<ImageMaskSpatialObjectType *>( const_cast<FixedImageMaskType *>( fixedImageMasks[n].GetPointer() ) )->GetImage() );
            fixedMaskResampler->SetTransform( fixedTransform );
            fixedMaskResampler->SetInterpolator( nearestNeighborInterpolator );
            fixedMaskResampler->UseReferenceImageOn();
            fixedMaskResampler->SetReferenceImage( virtualDomainImage );
            fixedMaskResampler->SetDefaultPixelValue( 0 );
            fixedMaskResampler->Update();

            typename ImageMaskSpatialObjectType::Pointer resampledFixedImageMask = ImageMaskSpatialObjectType::New();
            resampledFixedImageMask->SetImage( fixedMaskResampler->GetOutput() );

            dynamic_cast<ImageMetricType *>( multiMetric->GetMetricQueue()[n].GetPointer() )->SetFixedImageMask( resampledFixedImageMask );
            }

          if( movingImageMasks[n] )
            {
            typedef NearestNeighborInterpolateImageFunction<MovingMaskImageType, RealType> NearestNeighborInterpolatorType;
            typename NearestNeighborInterpolatorType::Pointer nearestNeighborInterpolator = NearestNeighborInterpolatorType::New();
            nearestNeighborInterpolator->SetInputImage( dynamic_cast<ImageMaskSpatialObjectType *>( const_cast<MovingImageMaskType *>( movingImageMasks[n].GetPointer() ) )->GetImage() );

            typedef ResampleImageFilter<MovingMaskImageType, MovingMaskImageType, RealType> MovingMaskResamplerType;
            typename MovingMaskResamplerType::Pointer movingMaskResampler = MovingMaskResamplerType::New();
            movingMaskResampler->SetInput( dynamic_cast<ImageMaskSpatialObjectType *>( const_cast<MovingImageMaskType *>( movingImageMasks[n].GetPointer() ) )->GetImage() );
            movingMaskResampler->SetTransform( movingTransform );
            movingMaskResampler->SetInterpolator( nearestNeighborInterpolator );
            movingMaskResampler->UseReferenceImageOn();
            movingMaskResampler->SetReferenceImage( virtualDomainImage );
            movingMaskResampler->SetDefaultPixelValue( 0 );
            movingMaskResampler->Update();

            typename ImageMaskSpatialObjectType::Pointer resampledMovingImageMask = ImageMaskSpatialObjectType::New();
            resampledMovingImageMask->SetImage( movingMaskResampler->GetOutput() );

            dynamic_cast<ImageMetricType *>( multiMetric->GetMetricQueue()[n].GetPointer() )->SetMovingImageMask( resampledMovingImageMask );
            }
          }
        }
      else
        {
        itkExceptionMacro( "Invalid metric." );
        }
      }
    }
  else
    {
    if( this->m_Metric->GetMetricCategory() == MetricType::POINT_SET_METRIC )
      {
      this->m_Metric->SetFixedObject( fixedPointSets[0] );
      this->m_Metric->SetMovingObject( movingPointSets[0] );

      dynamic_cast<PointSetMetricType *>( this->m_Metric.GetPointer() )->SetFixedTransform( const_cast<TransformBaseType *>( fixedTransform ) );
      dynamic_cast<PointSetMetricType *>( this->m_Metric.GetPointer() )->SetMovingTransform( const_cast<TransformBaseType *>( movingTransform ) );

      dynamic_cast<PointSetMetricType *>( this->m_Metric.GetPointer() )->SetCalculateValueAndDerivativeInTangentSpace( true );

      // The following boolean variable is on by default.  However, I set it explicitly here
      // to note behavioral differences between the Gaussian (original) SyN and B-spline
      // SyN.  A point-set is defined irregularly (i.e., not necessarily at voxel centers) over
      // the fixed and moving image domains.  For the Gaussian smoothing of the gradient field
      // with original SyN, the corresponding metric gradient values must be mapped to the closest
      // voxel locations in the reference domain.  The rest of the gradient values are zeroed
      // out prior to gaussian smoothing via convolution.  For the B-spline analog, the underlying
      // smoothing operation is done using the BSplineScatteredDataPointSettoImageFilter so we
      // don't need to artificially zero out "missing" values.

      dynamic_cast<PointSetMetricType *>( this->m_Metric.GetPointer() )->SetStoreDerivativeAsSparseFieldForLocalSupportTransforms( true );
      }
    else if( this->m_Metric->GetMetricCategory() == MetricType::IMAGE_METRIC )
      {

      if( !this->m_DownsampleImagesForMetricDerivatives )
        {
        this->m_Metric->SetFixedObject( fixedImages[0] );
        this->m_Metric->SetMovingObject( movingImages[0] );

        dynamic_cast<ImageMetricType *>( this->m_Metric.GetPointer() )->SetFixedTransform( const_cast<TransformBaseType *>( fixedTransform ) );
        dynamic_cast<ImageMetricType *>( this->m_Metric.GetPointer() )->SetMovingTransform( const_cast<TransformBaseType *>( movingTransform ) );

        dynamic_cast<ImageMetricType *>( this->m_Metric.GetPointer() )->SetFixedImageMask( fixedImageMasks[0] );
        dynamic_cast<ImageMetricType *>( this->m_Metric.GetPointer() )->SetMovingImageMask( movingImageMasks[0] );
        }
      else
        {
        typedef ResampleImageFilter<FixedImageType, FixedImageType, RealType> FixedResamplerType;
        typename FixedResamplerType::Pointer fixedResampler = FixedResamplerType::New();
        fixedResampler->SetInput( fixedImages[0] );
        fixedResampler->SetTransform( fixedTransform );
        fixedResampler->UseReferenceImageOn();
        fixedResampler->SetReferenceImage( virtualDomainImage );
        fixedResampler->SetDefaultPixelValue( 0 );
        fixedResampler->Update();

        typedef ResampleImageFilter<MovingImageType, MovingImageType, RealType> MovingResamplerType;
        typename MovingResamplerType::Pointer movingResampler = MovingResamplerType::New();
        movingResampler->SetInput( movingImages[0] );
        movingResampler->SetTransform( movingTransform );
        movingResampler->UseReferenceImageOn();
        movingResampler->SetReferenceImage( virtualDomainImage );
        movingResampler->SetDefaultPixelValue( 0 );
        movingResampler->Update();

        this->m_Metric->SetFixedObject( fixedResampler->GetOutput() );
        this->m_Metric->SetMovingObject( movingResampler->GetOutput() );

        if( fixedImageMasks[0] )
          {
          typedef NearestNeighborInterpolateImageFunction<FixedMaskImageType, RealType> NearestNeighborInterpolatorType;
          typename NearestNeighborInterpolatorType::Pointer nearestNeighborInterpolator = NearestNeighborInterpolatorType::New();
          nearestNeighborInterpolator->SetInputImage( dynamic_cast<ImageMaskSpatialObjectType *>( const_cast<FixedImageMaskType *>( fixedImageMasks[0].GetPointer() ) )->GetImage() );

          typedef ResampleImageFilter<FixedMaskImageType, FixedMaskImageType, RealType> FixedMaskResamplerType;
          typename FixedMaskResamplerType::Pointer fixedMaskResampler = FixedMaskResamplerType::New();
          fixedMaskResampler->SetInput( dynamic_cast<ImageMaskSpatialObjectType *>( const_cast<FixedImageMaskType *>( fixedImageMasks[0].GetPointer() ) )->GetImage() );
          fixedMaskResampler->SetTransform( fixedTransform );
          fixedMaskResampler->SetInterpolator( nearestNeighborInterpolator );
          fixedMaskResampler->UseReferenceImageOn();
          fixedMaskResampler->SetReferenceImage( virtualDomainImage );
          fixedMaskResampler->SetDefaultPixelValue( 0 );
          fixedMaskResampler->Update();

          typename ImageMaskSpatialObjectType::Pointer resampledFixedImageMask = ImageMaskSpatialObjectType::New();
          resampledFixedImageMask->SetImage( fixedMaskResampler->GetOutput() );

          dynamic_cast<ImageMetricType *>( this->m_Metric.GetPointer() )->SetFixedImageMask( resampledFixedImageMask );
          }

        if( movingImageMasks[0] )
          {
          typedef NearestNeighborInterpolateImageFunction<MovingMaskImageType, RealType> NearestNeighborInterpolatorType;
          typename NearestNeighborInterpolatorType::Pointer nearestNeighborInterpolator = NearestNeighborInterpolatorType::New();
          nearestNeighborInterpolator->SetInputImage( dynamic_cast<ImageMaskSpatialObjectType *>( const_cast<MovingImageMaskType *>( movingImageMasks[0].GetPointer() ) )->GetImage() );

          typedef ResampleImageFilter<MovingMaskImageType, MovingMaskImageType, RealType> MovingMaskResamplerType;
          typename MovingMaskResamplerType::Pointer movingMaskResampler = MovingMaskResamplerType::New();
          movingMaskResampler->SetInput( dynamic_cast<ImageMaskSpatialObjectType *>( const_cast<MovingImageMaskType *>( movingImageMasks[0].GetPointer() ) )->GetImage() );
          movingMaskResampler->SetTransform( movingTransform );
          movingMaskResampler->SetInterpolator( nearestNeighborInterpolator );
          movingMaskResampler->UseReferenceImageOn();
          movingMaskResampler->SetReferenceImage( virtualDomainImage );
          movingMaskResampler->SetDefaultPixelValue( 0 );
          movingMaskResampler->Update();

          typename ImageMaskSpatialObjectType::Pointer resampledMovingImageMask = ImageMaskSpatialObjectType::New();
          resampledMovingImageMask->SetImage( movingMaskResampler->GetOutput() );

          dynamic_cast<ImageMetricType *>( this->m_Metric.GetPointer() )->SetMovingImageMask( resampledMovingImageMask );
          }
        }
      }
    else
      {
      itkExceptionMacro( "Invalid metric." );
      }
    }

  if( this->m_DownsampleImagesForMetricDerivatives && this->m_Metric->GetMetricCategory() != MetricType::POINT_SET_METRIC )
    {
    const DisplacementVectorType zeroVector( 0.0 );

    typename DisplacementFieldType::Pointer identityField = DisplacementFieldType::New();
    identityField->CopyInformation( virtualDomainImage );
    identityField->SetRegions( virtualDomainImage->GetLargestPossibleRegion() );
    identityField->Allocate();
    identityField->FillBuffer( zeroVector );

    DisplacementFieldTransformPointer identityDisplacementFieldTransform = DisplacementFieldTransformType::New();
    identityDisplacementFieldTransform->SetDisplacementField( identityField );
    identityDisplacementFieldTransform->SetInverseDisplacementField( identityField );

    if( this->m_Metric->GetMetricCategory() == MetricType::MULTI_METRIC )
      {
      multiMetric->SetFixedTransform( identityDisplacementFieldTransform );
      multiMetric->SetMovingTransform( identityDisplacementFieldTransform );
      }
    else if( this->m_Metric->GetMetricCategory() == MetricType::IMAGE_METRIC )
      {
      dynamic_cast<ImageMetricType *>( this->m_Metric.GetPointer() )->SetFixedTransform( identityDisplacementFieldTransform );
      dynamic_cast<ImageMetricType *>( this->m_Metric.GetPointer() )->SetMovingTransform( identityDisplacementFieldTransform );
      }
    }

  this->m_Metric->Initialize();

  typedef typename ImageMetricType::DerivativeType MetricDerivativeType;
  const typename MetricDerivativeType::SizeValueType metricDerivativeSize = virtualDomainImage->GetLargestPossibleRegion().GetNumberOfPixels() * ImageDimension;
  MetricDerivativeType metricDerivative( metricDerivativeSize );

  metricDerivative.Fill( NumericTraits<typename MetricDerivativeType::ValueType>::ZeroValue() );
  this->m_Metric->GetValueAndDerivative( value, metricDerivative );

  // Ensure that the size of the optimizer weights is the same as the
  // number of local transform parameters (=ImageDimension)
  if( !this->m_OptimizerWeightsAreIdentity && this->m_OptimizerWeights.Size() == ImageDimension )
    {
    typename MetricDerivativeType::iterator it;
    for( it = metricDerivative.begin(); it != metricDerivative.end(); it += ImageDimension )
      {
      for( unsigned int d = 0; d < ImageDimension; d++ )
        {
        *(it + d) *= this->m_OptimizerWeights[d];
        }
      }
    }

  typename DisplacementFieldType::Pointer gradientField = DisplacementFieldType::New();
  gradientField->CopyInformation( virtualDomainImage );
  gradientField->SetRegions( virtualDomainImage->GetRequestedRegion() );
  gradientField->Allocate();

  ImageRegionIterator<DisplacementFieldType> ItG( gradientField, gradientField->GetRequestedRegion() );

  SizeValueType count = 0;
  for( ItG.GoToBegin(); !ItG.IsAtEnd(); ++ItG )
    {
    DisplacementVectorType displacement;
    for( SizeValueType d = 0; d < ImageDimension; d++ )
      {
      displacement[d] = metricDerivative[count++];
      }
    ItG.Set( displacement );
    }

  return gradientField;
}


// TODO: this method can remain exactly the same, should just change the terminology to reflect that it's the initial velocity
// TODO: field that is getting scaled. Fourier domain isn't relevant here because multiplication with a constant permutes with
// TODO: linear operators like the Fourier transform.
template<typename TFixedImage, typename TMovingImage, typename TOutputTransform, typename TVirtualImage, typename TPointSet>
typename FLASHImageRegistrationMethod<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>::DisplacementFieldPointer
FLASHImageRegistrationMethod<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>
::ScaleUpdateField( const DisplacementFieldType * updateField )
{
  typename DisplacementFieldType::SpacingType spacing = updateField->GetSpacing();
  ImageRegionConstIterator<DisplacementFieldType> ItF( updateField, updateField->GetLargestPossibleRegion() );

  RealType maxNorm = NumericTraits<RealType>::NonpositiveMin();
  for( ItF.GoToBegin(); !ItF.IsAtEnd(); ++ItF )
    {
    DisplacementVectorType vector = ItF.Get();

    RealType localNorm = 0;
    for( SizeValueType d = 0; d < ImageDimension; d++ )
      {
      localNorm += itk::Math::sqr( vector[d] / spacing[d] );
      }
    localNorm = std::sqrt( localNorm );

    if( localNorm > maxNorm )
      {
      maxNorm = localNorm;
      }
    }

  RealType scale = this->m_LearningRate;
  if( maxNorm > NumericTraits<RealType>::ZeroValue() )
    {
    scale /= maxNorm;
    }

  typedef Image<RealType, ImageDimension> RealImageType;

  typedef MultiplyImageFilter<DisplacementFieldType, RealImageType, DisplacementFieldType> MultiplierType;
  typename MultiplierType::Pointer multiplier = MultiplierType::New();
  multiplier->SetInput( updateField );
  multiplier->SetConstant( scale );

  typename DisplacementFieldType::Pointer scaledUpdateField = multiplier->GetOutput();
  scaledUpdateField->Update();
  scaledUpdateField->DisconnectPipeline();

  return scaledUpdateField;
}



// TODO: update for FLASH, inverted displacement automatically computed in shooting, so just return that?
// TODO: this could possibly be better than solving advection equation... I mean, I definitely don't want to implement
// TODO: that finive-volume method again
template<typename TFixedImage, typename TMovingImage, typename TOutputTransform, typename TVirtualImage, typename TPointSet>
typename FLASHImageRegistrationMethod<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>::DisplacementFieldPointer
FLASHImageRegistrationMethod<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>
::InvertDisplacementField( const DisplacementFieldType * field, const DisplacementFieldType * inverseFieldEstimate )
{
  typedef InvertDisplacementFieldImageFilter<DisplacementFieldType> InverterType;

  typename InverterType::Pointer inverter = InverterType::New();
  inverter->SetInput( field );
  inverter->SetInverseFieldInitialEstimate( inverseFieldEstimate );
  inverter->SetMaximumNumberOfIterations( 20 );
  inverter->SetMeanErrorToleranceThreshold( 0.001 );
  inverter->SetMaxErrorToleranceThreshold( 0.1 );
  inverter->Update();

  DisplacementFieldPointer inverseField = inverter->GetOutput();

  return inverseField;
}



/**
 * Smooth a field, ensure boundary is zeroVector, if variance is very small, weighted average smooth and original field
 */
template<typename TFixedImage, typename TMovingImage, typename TOutputTransform, typename TVirtualImage, typename TPointSet>
typename FLASHImageRegistrationMethod<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>::DisplacementFieldPointer
FLASHImageRegistrationMethod<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>
::GaussianSmoothDisplacementField( const DisplacementFieldType * field, const RealType variance )
{
  typedef ImageDuplicator<DisplacementFieldType> DuplicatorType;
  typename DuplicatorType::Pointer duplicator = DuplicatorType::New();
  duplicator->SetInputImage( field );
  duplicator->Update();

  DisplacementFieldPointer smoothField = duplicator->GetModifiableOutput();

  if( variance <= 0.0 )
    {
    return smoothField;
    }

  typedef GaussianOperator<RealType, ImageDimension> GaussianSmoothingOperatorType;
  GaussianSmoothingOperatorType gaussianSmoothingOperator;

  typedef VectorNeighborhoodOperatorImageFilter<DisplacementFieldType, DisplacementFieldType> GaussianSmoothingSmootherType;
  typename GaussianSmoothingSmootherType::Pointer smoother = GaussianSmoothingSmootherType::New();

  // TODO: investigate if these operators do smoothing in spatial or Fourier domain, could be significant bottleneck
  for( SizeValueType d = 0; d < ImageDimension; d++ )
    {
    // smooth along this dimension
    gaussianSmoothingOperator.SetDirection( d );
    gaussianSmoothingOperator.SetVariance( variance );
    gaussianSmoothingOperator.SetMaximumError( 0.001 );
    gaussianSmoothingOperator.SetMaximumKernelWidth( smoothField->GetRequestedRegion().GetSize()[d] );
    gaussianSmoothingOperator.CreateDirectional();

    // todo: make sure we only smooth within the buffered region
    smoother->SetOperator( gaussianSmoothingOperator );
    smoother->SetInput( smoothField );
    try
      {
      smoother->Update();
      }
    catch( ExceptionObject & exc )
      {
      std::string msg( "Caught exception: " );
      msg += exc.what();
      itkExceptionMacro( << msg );
      }

    smoothField = smoother->GetOutput();
    smoothField->Update();
    smoothField->DisconnectPipeline();
    }

  const DisplacementVectorType zeroVector( 0.0 );

  //make sure boundary does not move
  RealType weight1 = 1.0;
  if( variance < 0.5 )
    {
    weight1 = 1.0 - 1.0 * ( variance / 0.5 );
    }
  RealType weight2 = 1.0 - weight1;

  const typename DisplacementFieldType::RegionType region = field->GetLargestPossibleRegion();
  const typename DisplacementFieldType::SizeType size = region.GetSize();
  const typename DisplacementFieldType::IndexType startIndex = region.GetIndex();

  ImageRegionConstIteratorWithIndex<DisplacementFieldType> ItF( field, field->GetLargestPossibleRegion() );
  ImageRegionIteratorWithIndex<DisplacementFieldType> ItS( smoothField, smoothField->GetLargestPossibleRegion() );
  for( ItF.GoToBegin(), ItS.GoToBegin(); !ItF.IsAtEnd(); ++ItF, ++ItS )
    {
    typename DisplacementFieldType::IndexType index = ItF.GetIndex();
    bool isOnBoundary = false;
    for ( unsigned int d = 0; d < ImageDimension; d++ )
      {
      if( index[d] == startIndex[d] || index[d] == static_cast<IndexValueType>( size[d] ) - startIndex[d] - 1 )
        {
        isOnBoundary = true;
        break;
        }
      }
    if( isOnBoundary )
      {
      ItS.Set( zeroVector );
      }
    else
      {
      ItS.Set( ItS.Get() * weight1 + ItF.Get() * weight2 );
      }
    }

  return smoothField;
}


/*
 * Start the registration
 */
template<typename TFixedImage, typename TMovingImage, typename TOutputTransform, typename TVirtualImage, typename TPointSet>
void
FLASHImageRegistrationMethod<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>
::GenerateData()
{
  this->AllocateOutputs();  // TODO: may need to double check this doesn't interfere with FLASH type things

  for( this->m_CurrentLevel = 0; this->m_CurrentLevel < this->m_NumberOfLevels; this->m_CurrentLevel++ )
    {
    this->InitializeRegistrationAtEachLevel( this->m_CurrentLevel );

    // The base class adds the transform to be optimized at initialization.
    // However, since this class handles its own optimization, we remove it
    // to optimize separately.  We then add it after the optimization loop.

    this->m_CompositeTransform->RemoveTransform();

    this->StartOptimization();

    this->m_CompositeTransform->AddTransform( this->m_OutputTransform );
    }

  // TODO: assign appropriate transforms as output
  this->m_OutputTransform->SetDisplacementField(/* \phi_{1,0} here */);
  this->m_OutputTransform->SetInverseDisplacementField(/* \phi_{0,1} here  */);

  this->GetTransformOutput()->Set(this->m_OutputTransform);
}


/*
 * PrintSelf
 */
template<typename TFixedImage, typename TMovingImage, typename TOutputTransform, typename TVirtualImage, typename TPointSet>
void
FLASHImageRegistrationMethod<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  os << indent << "Number of current iterations per level: " << this->m_NumberOfIterationsPerLevel << std::endl;
  os << indent << "Learning rate: " << this->m_LearningRate << std::endl;
  os << indent << "Convergence threshold: " << this->m_ConvergenceThreshold << std::endl;
  os << indent << "Convergence window size: " << this->m_ConvergenceWindowSize << std::endl;
  os << indent << "Gaussian smoothing variance for the update field: " << this->m_GaussianSmoothingVarianceForTheUpdateField << std::endl;
  os << indent << "Gaussian smoothing variance for the total field: " << this->m_GaussianSmoothingVarianceForTheTotalField << std::endl;
  // TODO: put in the correct variable names here once they've been added to the registration object
  os << indent << "regularizer term weight: " << this->m_RegularizerTermWeight << std::endl;
  os << indent << "Laplacian term weight: " << this->m_LaplacianWeight << std::endl;
  os << indent << "identity term weight: " << this->m_IdentityWeight << std::endl;
  os << indent << "differential operator order: " << this->m_OperatorOrder << std::endl;
  os << indent << "number of time steps in integration: " << this->m_NumberOfTimeSteps << std::endl;
}

} // end namespace itk

#endif
