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
  m_RegularizerTermWeight( 0.03 ),
  m_LaplacianWeight( 3.0 ),
  m_IdentityWeight( 1.0 ),
  m_OperatorOrder( 6.0 ),
  m_NumberOfTimeSteps( 10 ),
  m_TimeStepSize( 1.0/10 )
{
  this->m_DownsampleImagesForMetricDerivatives = true;
  this->m_DoRungeKuttaForIntegration = false;
  this->m_NumberOfIterationsPerLevel.SetSize( 3 );
  this->m_NumberOfIterationsPerLevel[0] = 20;
  this->m_NumberOfIterationsPerLevel[1] = 30;
  this->m_NumberOfIterationsPerLevel[2] = 40;
  this->m_FourierSizes = {32, 32, 32};

  this->m_v0 = ITK_NULLPTR;
  this->m_completeTransform = ITK_NULLPTR;
  this->m_mType = MEM_HOST;
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
 * Utility to change ITK image type to PyCA image type
 */
template<typename TFixedImage, typename TMovingImage, typename TOutputTransform, typename TVirtualImage, typename TPointSet>
Image3D *
FLASHImageRegistrationMethod<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>
::itkToPycaImage(int x, int y, int z, TMovingImage * itkTypeImage)
{
  Image3D * pycaTypeImage = new Image3D(x, y, z, this->m_mType);
  std::copy(itkTypeImage->GetBufferPointer(),
            itkTypeImage->GetBufferPointer() + x*y*z,
            pycaTypeImage->get());
  return pycaTypeImage;
}


/**
 * Initialize registration objects for optimization at a specific level
 */
template<typename TFixedImage, typename TMovingImage, typename TOutputTransform, typename TVirtualImage, typename TPointSet>
void
FLASHImageRegistrationMethod<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>
::InitializeRegistrationAtEachLevel( const SizeValueType level )
{
  std::cout << "INITIALIZING AT LEVEL: " << level << std::endl;
  Superclass::InitializeRegistrationAtEachLevel( level );

  /*** create FLASH required objects ***/
  // images from first input channel will be I0 and I1; TODO: generalize I0 and I1 to multichannel registration
  typename TMovingImage::SizeType dims = this->m_MovingSmoothImages[level]->GetLargestPossibleRegion().GetSize();  // TODO: this line seg faults on level 1 (i.e. 2nd level) when running InitializeRegistrationAtEachLevel on it's own (i.e. without rest of method)
  this->m_I0 = this->itkToPycaImage(dims[0], dims[1], dims[2], this->m_MovingSmoothImages[level]);
  this->m_I1 = this->itkToPycaImage(dims[0], dims[1], dims[2], this->m_FixedSmoothImages[level]);

  // FFT operator, the differential operator/Riemannian metric kernel in the Fourier domain
  GridInfo grid = this->m_I0->grid();
  unsigned int numFourierCoeff = this->m_FourierSizes[level];
  if (numFourierCoeff % 2 == 0) numFourierCoeff -= 1;
  this->m_fftoper = new FftOper(this->m_LaplacianWeight, this->m_IdentityWeight, this->m_OperatorOrder,
                                grid, numFourierCoeff, numFourierCoeff, numFourierCoeff);
  this->m_fftoper->FourierCoefficient();

  // the velocity flow field in time
  this->m_VelocityFlowField = new FieldComplex3D * [this->m_NumberOfTimeSteps+1];
  for (int i = 0; i <= this->m_NumberOfTimeSteps; i++)
    this->m_VelocityFlowField[i] = new FieldComplex3D(numFourierCoeff, numFourierCoeff, numFourierCoeff);

  // fields to hold gradient information
  this->m_imMatchGradient = new FieldComplex3D(numFourierCoeff, numFourierCoeff, numFourierCoeff);
  this->m_gradv = new FieldComplex3D(numFourierCoeff, numFourierCoeff, numFourierCoeff);
  this->m_fwdgradvfft = new FieldComplex3D(numFourierCoeff, numFourierCoeff, numFourierCoeff);

  // fields to hold Jacobian components
  this->m_JacX = new FieldComplex3D(numFourierCoeff, numFourierCoeff, numFourierCoeff);
  this->m_JacY = new FieldComplex3D(numFourierCoeff, numFourierCoeff, numFourierCoeff);
  this->m_JacZ = new FieldComplex3D(numFourierCoeff, numFourierCoeff, numFourierCoeff);

  // some scratch memory to help with computations later
  this->m_adScratch1 = new FieldComplex3D(numFourierCoeff, numFourierCoeff, numFourierCoeff);
  this->m_adScratch2 = new FieldComplex3D(numFourierCoeff, numFourierCoeff, numFourierCoeff);
  this->m_scratch1 = new FieldComplex3D(numFourierCoeff, numFourierCoeff, numFourierCoeff);
  this->m_scratch2 = new FieldComplex3D(numFourierCoeff, numFourierCoeff, numFourierCoeff);
  this->m_scratch3 = new FieldComplex3D(numFourierCoeff, numFourierCoeff, numFourierCoeff);

  // some scratch memory in the spatial domain
  // TODO: spatial domain objects maybe initialized outside of this function? Just once for all scales?
  this->m_v0Spatial = new Field3D(grid, this->m_mType);
  this->m_scratchV1 = new Field3D(grid, this->m_mType);
  this->m_scratchV2 = new Field3D(grid, this->m_mType);
  this->m_phiinv = new Field3D(grid, this->m_mType);
  this->m_deformIm = new Image3D(grid, this->m_mType);
  this->m_splatI = new Image3D(grid, this->m_mType);
  this->m_splatOnes = new Image3D(grid, this->m_mType);
  this->m_residualIm = new Image3D(grid, this->m_mType);

  // the identity field in the Fourier domain
  this->m_identity = new Field3D(grid, this->m_mType);
  Opers::SetToIdentity(*(this->m_identity));
  this->idxf = new float[2 * this->m_fftoper->fsxFFT * this->m_fftoper->fsy * this->m_fftoper->fsz];
  this->idyf = new float[2 * this->m_fftoper->fsxFFT * this->m_fftoper->fsy * this->m_fftoper->fsz];
  this->idzf = new float[2 * this->m_fftoper->fsxFFT * this->m_fftoper->fsy * this->m_fftoper->fsz];
  this->m_fftoper->spatial2fourier_F(this->idxf, this->idyf, this->idzf, *(this->m_identity));


  if( level == 0 )
    {
    // If velocity and displacement objects are not initialized from a state restoration
    if( this->m_completeTransform.IsNull() || !this->m_v0 )
      {
      // Initialize complex field for initial velocity and a completeTransform object
      this->m_v0 = new FieldComplex3D(numFourierCoeff, numFourierCoeff, numFourierCoeff);
      this->m_completeTransform = OutputTransformType::New();
      }
    else
      {
      // TODO: should probably double check that m_v0 is initialized (need to update this code in state restoration)
      if( this->m_completeTransform->GetInverseDisplacementField() )
        {
        itkDebugMacro( "FLASH registration is initialized by restoring the state.");
        this->m_TransformParametersAdaptorsPerLevel[0]->SetTransform( this->m_completeTransform );
        this->m_TransformParametersAdaptorsPerLevel[0]->AdaptTransformParameters();
        }
      else
        {
        itkExceptionMacro( "Invalid state restoration." )
        }
      }
    }
  else
    {
    if( this->m_TransformParametersAdaptorsPerLevel[level] )
      {
      this->m_TransformParametersAdaptorsPerLevel[level]->SetTransform( this->m_completeTransform );
      this->m_TransformParametersAdaptorsPerLevel[level]->AdaptTransformParameters();
      }
    this->m_v0 = Pad_FieldComplex( this->m_v0, numFourierCoeff );
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

  // get initial fixed transform if there is one
  InitialTransformType * fixedInitialTransform = const_cast<InitialTransformType*>( this->GetFixedInitialTransform() );

  // Monitor the convergence
  typedef itk::Function::WindowConvergenceMonitoringFunction<RealType> ConvergenceMonitoringType;
  typename ConvergenceMonitoringType::Pointer convergenceMonitoring = ConvergenceMonitoringType::New();
  convergenceMonitoring->SetWindowSize( this->m_ConvergenceWindowSize );
  IterationReporter reporter( this, 0, 1 );  // TODO: should maybe read about this reporter function

  int my_iteration = 0;
  while( this->m_CurrentIteration++ < this->m_NumberOfIterationsPerLevel[this->m_CurrentLevel] && !this->m_IsConverged )
    {
    // create fixed and moving composite transform objects
    typename CompositeTransformType::Pointer fixedComposite = CompositeTransformType::New();
    if ( fixedInitialTransform != ITK_NULLPTR )
      {
      fixedComposite->AddTransform( fixedInitialTransform );
      }
    fixedComposite->FlattenTransformQueue();
    // typename CompositeTransformType::Pointer movingComposite = CompositeTransformType::New();
    // movingComposite->AddTransform( this->m_CompositeTransform );

    // Compute the update field
    MeasureType metricValue = 0.0;
    FieldComplex3D * smoothUpdateField = this->ComputeUpdateField(
      this->m_FixedSmoothImages, this->m_FixedPointSets, fixedComposite,
      this->m_MovingSmoothImages, this->m_MovingPointSets, this->m_CompositeTransform, // movingComposite
      this->m_FixedImageMasks, this->m_MovingImageMasks, metricValue );

    // update initial velocity and completeTransform, scaling of update field already happened
    // Add_FieldComplex(*(this->m_v0), *(this->m_v0), smoothUpdateField, 1);

    // // monitor convergence information
    // this->m_CurrentMetricValue = metricValue;
    // convergenceMonitoring->AddEnergyValue( this->m_CurrentMetricValue );
    // this->m_CurrentConvergenceValue = convergenceMonitoring->GetConvergenceValue();
    // if( this->m_CurrentConvergenceValue < this->m_ConvergenceThreshold )
    //   {
    //   this->m_IsConverged = true;
    //   }
    // reporter.CompletedStep();
    }
}


template<typename TFixedImage, typename TMovingImage, typename TOutputTransform, typename TVirtualImage, typename TPointSet>
FieldComplex3D *
FLASHImageRegistrationMethod<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>
::ComputeUpdateField( const FixedImagesContainerType fixedImages, const PointSetsContainerType fixedPointSets,
  const TransformBaseType * fixedTransform, const MovingImagesContainerType movingImages, const PointSetsContainerType movingPointSets,
  TransformBaseType * movingTransform, const FixedImageMasksContainerType fixedImageMasks, const MovingImageMasksContainerType movingImageMasks,
  MeasureType & value )
{
  this->ForwardIntegration();

  // TODO: at least initially the virtual domain will have to be the same as full res image, i.e. all shrink factors = 1
  VirtualImageBaseConstPointer virtualDomainImage = this->GetCurrentLevelVirtualDomainImage();
  typename DisplacementFieldType::Pointer movingToFixedInverseDisplacement = DisplacementFieldType::New();
  movingToFixedInverseDisplacement->CopyInformation( virtualDomainImage );
  movingToFixedInverseDisplacement->SetRegions( virtualDomainImage->GetBufferedRegion() );
  movingToFixedInverseDisplacement->Allocate();
  typename DisplacementFieldType::IndexType voxelIndex;
  DisplacementVectorType dispVector;
  for (int x; x < 1000; x++)    // TODO: TOTALLY MADE UP NUMBERS HERE, NEED SIZE EXTENTS FOR IMAGE GRID!!!!!!!!
  {                             // TODO: ALSO NEED TO SUBTRACT OFF IDENTITY!!!!!!
    for (int y; y < 1000; y++)
    {
      for (int z; z < 1000; z++)
      {
        Vec3Df dispVector_pyca = this->m_phiinv->get(x, y, z);
        dispVector[0] = dispVector_pyca[0];
        dispVector[1] = dispVector_pyca[1];
        dispVector[2] = dispVector_pyca[2];
        voxelIndex[0] = x;
        voxelIndex[1] = y;
        voxelIndex[2] = z;
        movingToFixedInverseDisplacement->SetPixel(voxelIndex, dispVector);
      }
    }
  }
  this->m_completeTransform->SetInverseDisplacementField(movingToFixedInverseDisplacement);

  typename CompositeTransformType::Pointer movingComposite = CompositeTransformType::New();
  movingComposite->AddTransform( movingTransform );
  movingComposite->AddTransform( this->m_completeTransform->GetInverseTransform() );
  movingComposite->FlattenTransformQueue();
  movingComposite->SetOnlyMostRecentTransformToOptimizeOn();

  DisplacementFieldPointer metricGradientField = this->ComputeMetricGradientField(
                                                  fixedImages, fixedPointSets, fixedTransform,
                                                  movingImages, movingPointSets, movingComposite,
                                                  fixedImageMasks, movingImageMasks, value );



  // set m_residualIm to data in metricGradientField

  this->BackwardIntegration();

  // // TODO: ensure correct objects are passed here
  // AddI_FieldComplex(this->m_v0, imMatchGradient,
  //                   1.0/(this->m_RegularizerTermWeight*this->m_RegularizerTermWeight))
  // DisplacementFieldPointer scaledUpdateField = this->ScaleUpdateField( updateField );
  // return scaledUpdateField;
}


template<typename TFixedImage, typename TMovingImage, typename TOutputTransform, typename TVirtualImage, typename TPointSet>
void
FLASHImageRegistrationMethod<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>
::RungeKuttaStep(FieldComplex3D * sf1, FieldComplex3D * sf2, FieldComplex3D * sf3,
                 FieldComplex3D * vff1, FieldComplex3D * vff2, RealType dt)
// sf: scratch field (preallocated memory to store intermediate calculations)
// vff: velocity flow field (vff1: i-1, vff2: i)
// dt: time step size
{
  // v1 = v0 - (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
  // k1
  adTranspose(*sf1, *vff1, *vff1);
  // partially update v1 = v0 - (dt/6)*k1
  Copy_FieldComplex(*vff2, *vff1);
  AddI_FieldComplex(*vff2, *sf1, -dt / 6.0);
  // k2
  Copy_FieldComplex(*sf3, *vff1);
  AddI_FieldComplex(*sf3, *sf1, -0.5*dt);
  adTranspose(*sf2, *sf3, *sf3);
  // partially update v1 = v1 - (dt/3)*k2
  AddI_FieldComplex(*vff2, *sf2, -dt / 3.0);
  // k3 (stored in scratch1)
  Copy_FieldComplex(*sf3, *vff1);
  AddI_FieldComplex(*sf3, *sf2, -0.5*dt);
  adTranspose(*sf1, *sf3, *sf3);
  // partially update v1 = v1 - (dt/3)*k3
  AddI_FieldComplex(*vff2, *sf1, -dt / 3.0);
  // k4 (stored in scratch2)
  Copy_FieldComplex(*sf3, *vff1);
  AddI_FieldComplex(*sf3, *sf1, -dt);
  adTranspose(*sf2, *sf3, *sf3);
  // finish updating v1 = v1 - (dt/6)*k4
  AddI_FieldComplex(*vff2, *sf2, -dt / 6.0);
}


template<typename TFixedImage, typename TMovingImage, typename TOutputTransform, typename TVirtualImage, typename TPointSet>
void
FLASHImageRegistrationMethod<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>
::EulerStep(FieldComplex3D * sf1, FieldComplex3D * vff1, FieldComplex3D * vff2, RealType dt)
// sf: scratch field (preallocated memory to store intermediate calculations)
// vff: velocity flow field (vff1: i-1, vff2: i)
// dt: time step size
{
  // v0 = v0 - dt * adTranspose(v0, v0)
  Copy_FieldComplex(*vff2, *vff1);
  adTranspose(*sf1, *vff2, *vff2);
  AddI_FieldComplex(*vff2, *sf1, -dt);
}


template<typename TFixedImage, typename TMovingImage, typename TOutputTransform, typename TVirtualImage, typename TPointSet>
void
FLASHImageRegistrationMethod<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>
::AdvectionStep(FieldComplex3D * JacX, FieldComplex3D * JacY, FieldComplex3D * JacZ,
                FieldComplex3D * sf1, FieldComplex3D * sf2, FieldComplex3D * vff,
                RealType dt)
// sf: scratch field (preallocated memory to store intermediate calculations)
// vff: velocity flow field
// dt: time step size
{
  Jacobian(*JacX, *JacY, *JacZ, *(this->m_fftoper->CDcoeff), *sf1);
  this->m_fftoper->ConvolveComplexFFT(*sf2, 0, *JacX, *JacY, *JacZ, *vff);
  AddIMul_FieldComplex(*sf1, *sf2, *vff, -dt);
}


template<typename TFixedImage, typename TMovingImage, typename TOutputTransform, typename TVirtualImage, typename TPointSet>
void
FLASHImageRegistrationMethod<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>
::ForwardIntegration()
{
  // obtain velocity flow with EPDiff in Fourier domain
  Copy_FieldComplex(*(this->m_VelocityFlowField[0]), *(this->m_v0));
  if (this->m_DoRungeKuttaForIntegration)
  {
    for (int i = 1; i <= this->m_NumberOfTimeSteps; i++)
      RungeKuttaStep(this->m_scratch1, this->m_scratch2, this->m_scratch3,
                     this->m_VelocityFlowField[i-1], this->m_VelocityFlowField[i],
                     this->m_TimeStepSize);
  }
  else
  {
    for (int i = 1; i <= this->m_NumberOfTimeSteps; i++)
      EulerStep(this->m_scratch1,
                this->m_VelocityFlowField[i-1],
                this->m_VelocityFlowField[i],
                this->m_TimeStepSize);
  }

  // integrate velocity flow through advection equation to obtain inverse of path endpoint
  this->m_scratch1->initVal(complex<float>(0.0, 0.0)); // displacement field
  for (int i = 0; i < this->m_NumberOfTimeSteps; i++)
    AdvectionStep(this->m_JacX, this->m_JacY, this->m_JacZ,
                  this->m_scratch1, this->m_scratch2,
                  this->m_VelocityFlowField[i],
                  this->m_TimeStepSize);

  // obtain spatial domain transform, apply to image
  this->m_fftoper->fourier2spatial_addH(*(this->m_phiinv),
                                        *(this->m_scratch1),
                                        this->idxf, this->idyf, this->idzf);
  Opers::ApplyH(*(this->m_deformIm),
                *(this->m_I0),
                *(this->m_phiinv),
                BACKGROUND_STRATEGY_WRAP);
}


template<typename TFixedImage, typename TMovingImage, typename TOutputTransform, typename TVirtualImage, typename TPointSet>
void
FLASHImageRegistrationMethod<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>
::AdjointStep(FieldComplex3D * sf1, FieldComplex3D * sf2, FieldComplex3D * vff,
              FieldComplex3D * vadj, FieldComplex3D * dvadj,
              RealType dt)
// sf: scratch field (preallocated memory to store intermediate calculations)
// vff: velocity flow field
// vadj: adjoint image
// dvadj: adjoint velocity
// dt: time step size
{
  ad(*sf1, *vff, *dvadj);
  adTranspose(*sf2, *dvadj, *vff);
  for (int i = 0; i < this->m_fourierSizes[this->m_CurrentLevel] * 3 * 3; i++) // wow, per voxel per component loop!
    dvadj->data[i] +=  dt * (vadj->data[i] - sf1->data[i] + sf2->data[i]);
  adTranspose(*sf1, *vff, *vadj);
  AddI_FieldComplex(*vadj, *sf1, dt);
}


template<typename TFixedImage, typename TMovingImage, typename TOutputTransform, typename TVirtualImage, typename TPointSet>
void
FLASHImageRegistrationMethod<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>
::BackwardIntegration()
{
  // initialize backward integration
  Opers::Gradient(*(this->m_scratchV1),
                  *(this->m_deformIm),
                  DIFF_CENTRAL, BC_WRAP);
  // residualIm should be the image matching gradient,
  Opers::MulMulC_I(*(this->m_scratchV1),
                   *(this->m_residualIm), -1.0);
  this->m_fftoper->spatial2fourier(this->m_fwdgradvfft,
                                 *(this->m_scratchV1));
  MulI_FieldComplex(this->m_fwdgradvfft,
                  *(this->m_fftoper->Kcoeff));
  // integrate adjoint system entirely in Lie algebra
  this->m_imMatchGradient->initVal(complex<float>(0.0, 0.0)); // backward to t=0
  for (int i = this->m_NumberOfTimeSteps; i > 0; i--) // reduced adjoint jacobi fields
    AdjointStep(this->m_scratch1, this->m_scratch2, this->m_VelocityFlowField[i],
                this->m_fwdgradvfft, this->m_imMatchGradient,
                this->m_TimeStepSize);
}


// ad operator
// CD: central difference
// ConvolveComplexFFT(CD*v, w)-ConvolveComplexFFT(CD*w, v))
template<typename TFixedImage, typename TMovingImage, typename TOutputTransform, typename TVirtualImage, typename TPointSet>
void
FLASHImageRegistrationMethod<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>
::ad(FieldComplex3D & advw, const FieldComplex3D & v, const FieldComplex3D & w)
{
     Jacobian(*(this->m_JacX),
              *(this->m_JacY),
              *(this->m_JacZ),
              *(this->fftoper->CDcoeff), v); 
     this->m_fftoper->ConvolveComplexFFT(advw, 0,
                                         *(this->m_JacX),
                                         *(this->m_JacY),
                                         *(this->m_JacZ), w);
     Jacobian(*(this->m_JacX),
              *(this->m_JacY),
              *(this->m_JacZ),
              *(this->fftoper->CDcoeff), w); 
     this->m_fftoper->ConvolveComplexFFT(*(this->m_adScratch1), 0,
                                         *(this->m_JacX),
                                         *(this->m_JacY),
                                         *(this->m_JacZ), v);
     AddI_FieldComplex(advw, *(this->m_adScratch1), -1.0);
}


// adTranspose
// spatial domain: K(Dv^T Lw + div(L*w x v))
// K*(CorrComplexFFT(CD*v^T, L*w) + TensorCorr(L*w, v) * D)
template<typename TFixedImage, typename TMovingImage, typename TOutputTransform, typename TVirtualImage, typename TPointSet>
void
FLASHImageRegistrationMethod<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>
::adTranspose(FieldComplex3D & adTransvw, const FieldComplex3D & v, const FieldComplex3D & w)
{
     Mul_FieldComplex(*(this->m_adScratch1),
                      *(this->m_fftoper->Lcoeff), w);
     JacobianT(*(this->m_JacX),
               *(this->m_JacY),
               *(this->m_JacZ),
               *(this->m_fftoper->CDcoeff), v); 
     this->m_fftoper->ConvolveComplexFFT(adTransvw, 1,
                                         *(this->m_JacX),
                                         *(this->m_JacY),
                                         *(this->m_JacZ),
                                         *(this->m_adScratch1));
     this->m_fftoper->CorrComplexFFT(*(this->m_adScratch2), v,
                                     *(this->m_adScratch1),
                                     *(this->m_fftoper->CDcoeff));
     AddI_FieldComplex(adTransvw, *(this->m_adScratch2), 1.0);
     MulI_FieldComplex(adTransvw, *(this->m_fftoper->Kcoeff));
}


// // TODO: wow, this recomputes the downsampling of the fixed image and it's mask on every single iteration... so inefficient
// // TODO: also, lots of code duplication here between multiMetric and singleMetric case, could be cleaned up with private subroutine
// // TODO: eventually want to look at LCC computation to ensure it uses Summed Area Tables
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


// // TODO: this method can remain exactly the same, should just change the terminology to reflect that it's the initial velocity
// // TODO: field that is getting scaled. Fourier domain isn't relevant here because multiplication with a constant permutes with
// // TODO: linear operators like the Fourier transform.
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

    // this->m_CompositeTransform->AddTransform( this->m_OutputTransform );
    std::cout << "Completed resample iteration: " << this->m_CurrentLevel << std::endl;
    }

  // TODO: assign appropriate transforms as output
  // this->m_OutputTransform->SetDisplacementField(/* \phi_{1,0} here */);
  // this->m_OutputTransform->SetInverseDisplacementField(/* \phi_{0,1} here  */);

  // this->GetTransformOutput()->Set(this->m_OutputTransform);
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
  os << indent << "regularizer term weight: " << this->m_RegularizerTermWeight << std::endl;
  os << indent << "Laplacian term weight: " << this->m_LaplacianWeight << std::endl;
  os << indent << "identity term weight: " << this->m_IdentityWeight << std::endl;
  os << indent << "differential operator order: " << this->m_OperatorOrder << std::endl;
  os << indent << "number of time steps in integration: " << this->m_NumberOfTimeSteps << std::endl;
}


} // end namespace itk

#endif
