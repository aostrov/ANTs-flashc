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
#ifndef itkFLASHImageRegistrationMethod_h
#define itkFLASHImageRegistrationMethod_h

#include "itkImageRegistrationMethodv4.h"

#include "FftOper.h"
#include "FieldComplex3D.h"
#include "ITKFileIO.h"
#include "IOpers.h"
#include "FOpers.h"
#include "IFOpers.h"
#include "HFOpers.h"
#include "Reduction.h"
#include "FluidKernelFFT.h"

namespace itk
{

/** \class FLASHImageRegistrationMethod
 * \brief Interface method for the current registration framework
 * using the FLASH geodesic shooting registration method
 *
 * Output: The output is the updated transform which has been added to the
 * composite transform, the inverse transform, and the initial velocity
 * in on the low-dimensional Fourier basis
 *
 * This derived class from the ImageRegistrationMethodv4 class
 * is an implementation of the FLASH algorithm detailed in:
 *
 * Zhang, M., Liao, R., Dalca, A. V., Turk, E. A., Luo, J.,
 * Grant, P. E., Golland, P. (2017). Frequency diffeomorphisms for
 * efficient image registration. In International conference on information
 * processing in medical imaging, Springer (pp. 559–570).
 *
 * with more background on a highly related but slightly different method in:
 *
 * Zhang, Miaomiao, Thomas Fletcher, P. (2015). Finite-dimensional lie algebras
 * for fast diffeomorphic image registration.
 * Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial
 * Intelligence and Lecture Notes in Bioinformatics), 9123, 249-260.
 *
 * The object to be optimized is an initial velocity field, represented
 * in Fourier space as a band limited set of spatial frequency coefficients
 * of which there are many fewer than there are voxels in the fixed and moving
 * images. The initial velocity parameterizes a path of diffeomorphic transforms
 * which is computed by integrating the Euler-Poincare differential equation.
 * The moving image is transported by the endpoint of this path. The residual
 * image matching is backpropagated through the geodesic path and used to
 * update the initial velocity.
 *
 * \author Greg M. Fleishman
 * \FLASH method creator: Miaomiao Zhang
 *
 * \ingroup ITKRegistrationMethodsv4
 */
template<typename TFixedImage, typename TMovingImage, typename TOutputTransform =
  DisplacementFieldTransform<double, TFixedImage::ImageDimension>,
  typename TVirtualImage = TFixedImage,
  typename TPointSet = PointSet<unsigned int, TFixedImage::ImageDimension> >
class ITK_TEMPLATE_EXPORT FLASHImageRegistrationMethod
: public ImageRegistrationMethodv4<TFixedImage, TMovingImage, TOutputTransform, TVirtualImage, TPointSet>
{
public:
  /** Standard class typedefs. */
  typedef FLASHImageRegistrationMethod                                            Self;
  typedef ImageRegistrationMethodv4<TFixedImage, TMovingImage, TOutputTransform,
                                                       TVirtualImage, TPointSet>  Superclass;
  typedef SmartPointer<Self>                                                      Pointer;
  typedef SmartPointer<const Self>                                                ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** ImageDimension constants */
  itkStaticConstMacro( ImageDimension, unsigned int, TFixedImage::ImageDimension );

  /** Run-time type information (and related methods). */
  itkTypeMacro( FLASHImageRegistrationMethod, ImageRegistrationMethodv4 );  // might need SimpleImageRegistrationMethod for 2nd arg instead

  // TODO: some of these typedefs may be unnecessary, review after I've built itkFLASHImageRegistrationMethod.hxx
  // TODO: may need some additional typedefs for objects specific to FLASH method, review after ...
  /** Input typedefs for the images. */
  typedef TFixedImage                                                 FixedImageType;
  typedef typename FixedImageType::Pointer                            FixedImagePointer;
  typedef typename Superclass::FixedImagesContainerType               FixedImagesContainerType;
  typedef TMovingImage                                                MovingImageType;
  typedef typename MovingImageType::Pointer                           MovingImagePointer;
  typedef typename Superclass::MovingImagesContainerType              MovingImagesContainerType;

  typedef typename Superclass::PointSetType                           PointSetType;
  typedef typename PointSetType::Pointer                              PointSetPointer;
  typedef typename Superclass::PointSetsContainerType                 PointSetsContainerType;
  
  typedef typename MovingImageType::RegionType                        RegionType;

  /** Metric and transform typedefs */
  typedef typename Superclass::ImageMetricType                        ImageMetricType;
  typedef typename ImageMetricType::Pointer                           ImageMetricPointer;
  typedef typename ImageMetricType::MeasureType                       MeasureType;

  typedef ImageMaskSpatialObject<ImageDimension>                      ImageMaskSpatialObjectType;
  typedef typename Superclass::FixedImageMaskType                     FixedImageMaskType;
  typedef typename ImageMaskSpatialObjectType::ImageType              FixedMaskImageType;
  typedef typename Superclass::FixedImageMasksContainerType           FixedImageMasksContainerType;
  typedef typename Superclass::MovingImageMaskType                    MovingImageMaskType;
  typedef typename ImageMaskSpatialObjectType::ImageType              MovingMaskImageType;
  typedef typename Superclass::MovingImageMasksContainerType          MovingImageMasksContainerType;

  typedef typename Superclass::VirtualImageType                       VirtualImageType;
  typedef typename Superclass::VirtualImageBaseType                   VirtualImageBaseType;
  typedef typename Superclass::VirtualImageBaseConstPointer           VirtualImageBaseConstPointer;

  typedef typename Superclass::MultiMetricType                        MultiMetricType;
  typedef typename Superclass::MetricType                             MetricType;
  typedef typename MetricType::Pointer                                MetricPointer;
  typedef typename Superclass::PointSetMetricType                     PointSetMetricType;

  typedef typename Superclass::InitialTransformType                   InitialTransformType;
  typedef TOutputTransform                                            OutputTransformType;
  typedef typename OutputTransformType::Pointer                       OutputTransformPointer;
  typedef typename OutputTransformType::ScalarType                    RealType;
  typedef typename OutputTransformType::DerivativeType                DerivativeType;
  typedef typename DerivativeType::ValueType                          DerivativeValueType;
  typedef typename OutputTransformType::DisplacementFieldType         DisplacementFieldType;
  typedef typename DisplacementFieldType::Pointer                     DisplacementFieldPointer;
  typedef typename DisplacementFieldType::PixelType                   DisplacementVectorType;

  typedef typename Superclass::CompositeTransformType                 CompositeTransformType;
  typedef typename CompositeTransformType::TransformType              TransformBaseType;

  typedef typename Superclass::DecoratedOutputTransformType           DecoratedOutputTransformType;
  typedef typename DecoratedOutputTransformType::Pointer              DecoratedOutputTransformPointer;

  typedef DisplacementFieldTransform<RealType, ImageDimension>        DisplacementFieldTransformType;
  typedef typename DisplacementFieldTransformType::Pointer            DisplacementFieldTransformPointer;

  typedef Array<SizeValueType>                                        NumberOfIterationsArrayType;

  /** Set/Get the learning rate. */
  itkSetMacro( LearningRate, RealType );
  itkGetConstMacro( LearningRate, RealType );

  /** Set/Get the number of iterations per level. */
  itkSetMacro( NumberOfIterationsPerLevel, NumberOfIterationsArrayType );
  itkGetConstMacro( NumberOfIterationsPerLevel, NumberOfIterationsArrayType );

  /** Set/Get the convergence threshold */
  itkSetMacro( ConvergenceThreshold, RealType );
  itkGetConstMacro( ConvergenceThreshold, RealType );

  /** Set/Get the convergence window size */
  itkSetMacro( ConvergenceWindowSize, unsigned int );
  itkGetConstMacro( ConvergenceWindowSize, unsigned int );

  /** Set/Get downsample for metric derivatives option */
  itkSetMacro( DownsampleImagesForMetricDerivatives, bool );
  itkGetConstMacro( DownsampleImagesForMetricDerivatives, bool );

  /** Set/Get modifiable initial velocity to save/restore the current state of the registration. */
 itkSetObjectMacro( FixedToMiddleTransform, OutputTransformType );
 itkGetModifiableObjectMacro( FixedToMiddleTransform, OutputTransformType );

 /** Set/Get modifiable initial velocity to save/restore the current state of the registration. */
 itkSetObjectMacro( MovingToMiddleTransform, OutputTransformType );
 itkGetModifiableObjectMacro( MovingToMiddleTransform, OutputTransformType );

protected:
  FLASHImageRegistrationMethod();
  virtual ~FLASHImageRegistrationMethod();
  virtual void PrintSelf( std::ostream & os, Indent indent ) const ITK_OVERRIDE;

  /** Perform the registration. */
  virtual void  GenerateData() ITK_OVERRIDE;

  /** Handle optimization internally */
  virtual void StartOptimization();

  virtual void InitializeRegistrationAtEachLevel( const SizeValueType ) ITK_OVERRIDE;

  virtual FieldComplex3D * ComputeUpdateField( const FixedImagesContainerType, const PointSetsContainerType,  // FLASH EDIT, return type
    const TransformBaseType *, const MovingImagesContainerType, const PointSetsContainerType,
    TransformBaseType *, const FixedImageMasksContainerType, const MovingImageMasksContainerType, MeasureType & );
  virtual DisplacementFieldPointer ComputeMetricGradientField( const FixedImagesContainerType,
    const PointSetsContainerType, const TransformBaseType *, const MovingImagesContainerType,
    const PointSetsContainerType, const TransformBaseType *, const FixedImageMasksContainerType,
    const MovingImageMasksContainerType, MeasureType & );
  virtual DisplacementFieldPointer ScaleUpdateField( const DisplacementFieldType * );

  // dummies for observer
  OutputTransformPointer                                          m_MovingToMiddleTransform;
  OutputTransformPointer                                          m_FixedToMiddleTransform;

  // FLASH EDIT
  Image3D * itkToPycaImage(int, int, int, TMovingImage *);
  void EulerStep(FieldComplex3D *, FieldComplex3D *, FieldComplex3D *, float);
  void RungeKuttaStep(FieldComplex3D *, FieldComplex3D *, FieldComplex3D *,
                      FieldComplex3D *, FieldComplex3D *, float);
  void AdvectionStep(FieldComplex3D *, FieldComplex3D *, FieldComplex3D *,
                     FieldComplex3D *, FieldComplex3D *, FieldComplex3D *,
                     float);
  void AdjointStep(FieldComplex3D *, FieldComplex3D *, FieldComplex3D *,
                   FieldComplex3D *, FieldComplex3D *,
                   float);
  void ad(FieldComplex3D &, const FieldComplex3D &, const FieldComplex3D &);
  void adTranspose(FieldComplex3D &, const FieldComplex3D &, const FieldComplex3D &);
  void ForwardIntegration();
  void BackwardIntegration();
  // END: FLASH EDIT
  
  RealType                                                        m_LearningRate;

  OutputTransformPointer                                          m_completeTransform;

  RealType                                                        m_ConvergenceThreshold;
  unsigned int                                                    m_ConvergenceWindowSize;

  NumberOfIterationsArrayType                                     m_NumberOfIterationsPerLevel;
  bool                                                            m_DownsampleImagesForMetricDerivatives;

  // FLASH EDIT
  // FLASH specific variables
  RealType                                                        m_RegularizerTermWeight;
  RealType                                                        m_LaplacianWeight;
  RealType                                                        m_IdentityWeight;
  int                                                             m_OperatorOrder;
  int                                                             m_NumberOfTimeSteps;
  float                                                           m_TimeStepSize;
  std::vector<int>                                                m_FourierSizes;
  bool                                                            m_DoRungeKuttaForIntegration;

  Image3D *                                                       m_I0;
  Image3D *                                                       m_I1;

  FieldComplex3D *                                                m_v0;
  FieldComplex3D *                                                m_gradv;
  FieldComplex3D *                                                m_imMatchGradient;
  FieldComplex3D *                                                m_fwdgradvfft;
  FieldComplex3D *                                                m_JacX;
  FieldComplex3D *                                                m_JacY;
  FieldComplex3D *                                                m_JacZ;
  FieldComplex3D *                                                m_adScratch1;
  FieldComplex3D *                                                m_adScratch2;
  FieldComplex3D *                                                m_scratch1;
  FieldComplex3D *                                                m_scratch2;
  FieldComplex3D *                                                m_scratch3;
  FieldComplex3D **                                               m_VelocityFlowField;

  Field3D *                                                       m_scratchV1;
  Field3D *                                                       m_phiinv;
  Field3D *                                                       m_identity;

  float *                                                         idxf;
  float *                                                         idyf;
  float *                                                         idzf;

  MemoryType                                                      m_mType;
  FftOper *                                                       m_fftoper;

public:
  // Set/Get for the FLASH specific variables
  void SetRegularizerTermWeight( RealType weight)
    {
      m_RegularizerTermWeight = weight;
    }
  RealType GetRegularizerTermWeight() const
    {
      return m_RegularizerTermWeight;
    }

  void SetLaplacianWeight( RealType weight)
    {
      m_LaplacianWeight = weight;
    }
  RealType GetLaplacianWeight() const
    {
      return m_LaplacianWeight;
    }

  void SetIdentityWeight( RealType weight)
    {
      m_IdentityWeight = weight;
    }
  RealType GetIdentityWeight() const
    {
      return m_IdentityWeight;
    }

  void SetOperatorOrder( int weight)
    {
      m_OperatorOrder = weight;
    }
  int GetOperatorOrder() const
    {
      return m_OperatorOrder;
    }

  void SetNumberOfTimeSteps( int steps)
    {
      m_NumberOfTimeSteps = steps;
    }
  int GetNumberOfTimeSteps() const
    {
      return m_NumberOfTimeSteps;
    }

  void SetFourierSizes( std::vector<int> fourierSizes)
    {
      m_FourierSizes = fourierSizes;
    }
  std::vector<int> GetFourierSizes() const
    {
      return m_FourierSizes;
    }

  void SetCompleteTransform(OutputTransformType * completeTransform)
  {
    m_completeTransform = completeTransform;
  }

  // END: FLASH EDIT

private:
  ITK_DISALLOW_COPY_AND_ASSIGN(FLASHImageRegistrationMethod);

};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkFLASHImageRegistrationMethod.hxx"
#endif

#endif
