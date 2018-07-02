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
  typedef FLASHImageRegistrationMethod                                          Self;
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

  // TODO: may need additional getter/setter methods for FLASH specific options, review after building hxx
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

  /**
   * Set/Get the Gaussian smoothing variance for the update field.
   * Default = 1.75.
   */
  itkSetMacro( GaussianSmoothingVarianceForTheUpdateField, RealType );
  itkGetConstReferenceMacro( GaussianSmoothingVarianceForTheUpdateField, RealType );

  /**
   * Set/Get the Gaussian smoothing variance for the total field.
   * Default = 0.5.
   */
  itkSetMacro( GaussianSmoothingVarianceForTheTotalField, RealType );
  itkGetConstReferenceMacro( GaussianSmoothingVarianceForTheTotalField, RealType );

  /** Set/Get modifiable initial velocity to save/restore the current state of the registration. */
  itkSetObjectMacro( /* initial velocity field, initial velocity field type */ );
  itkGetModifiableObjectMacro( /* initial velocity field, initial velocity field type */ );

protected:
  FLASHImageRegistrationMethod();
  virtual ~FLASHImageRegistrationMethod();
  virtual void PrintSelf( std::ostream & os, Indent indent ) const ITK_OVERRIDE;

  /** Perform the registration. */
  virtual void  GenerateData() ITK_OVERRIDE;

  /** Handle optimization internally */
  virtual void StartOptimization();

  virtual void InitializeRegistrationAtEachLevel( const SizeValueType ) ITK_OVERRIDE;

  virtual DisplacementFieldPointer ComputeUpdateField( const FixedImagesContainerType, const PointSetsContainerType,
    const TransformBaseType *, const MovingImagesContainerType, const PointSetsContainerType,
    const TransformBaseType *, const FixedImageMasksContainerType, const MovingImageMasksContainerType, MeasureType & );
  virtual DisplacementFieldPointer ComputeMetricGradientField( const FixedImagesContainerType,
    const PointSetsContainerType, const TransformBaseType *, const MovingImagesContainerType,
    const PointSetsContainerType, const TransformBaseType *, const FixedImageMasksContainerType,
    const MovingImageMasksContainerType, MeasureType & );

  virtual DisplacementFieldPointer ScaleUpdateField( const DisplacementFieldType * );
  virtual DisplacementFieldPointer GaussianSmoothDisplacementField( const DisplacementFieldType *, const RealType );
  virtual DisplacementFieldPointer InvertDisplacementField( const DisplacementFieldType *, const DisplacementFieldType * = ITK_NULLPTR );
  
  // TODO: add FLASH specific member variables
  RealType                                                        m_LearningRate;

  OutputTransformPointer                                          m_MovingToMiddleTransform;

  RealType                                                        m_ConvergenceThreshold;
  unsigned int                                                    m_ConvergenceWindowSize;

  NumberOfIterationsArrayType                                     m_NumberOfIterationsPerLevel;
  bool                                                            m_DownsampleImagesForMetricDerivatives;

  // FLASH EDIT
  // FLASH specific variables
  RealType                                                        m_RegularizerTermWeight;
  RealType                                                        m_LaplacianWeight;
  RealType                                                        m_IdentityWeight;
  RealType                                                        m_OperatorOrder;
  unsigned int                                                    m_NumberOfTimeSteps;

  // Set/Get for the FLASH specific variables
  void SetRegularizerTermWeight( RealType weight)
    {
      this->m_RegularizerTermWeight = weight;
    }
  RealType GetRegularizerTermWeight() const
    {
      return this->m_RegularizerTermWeight;
    }

  void SetLaplacianWeight( RealType weight)
    {
      this->m_LaplacianWeight = weight;
    }
  RealType GetLaplacianWeight() const
    {
      return this->m_LaplacianWeight;
    }

  void SetIdentityWeight( RealType weight)
    {
      this->m_IdentityWeight = weight;
    }
  RealType GetIdentityWeight() const
    {
      return this->m_IdentityWeight;
    }

  void SetOperatorOrder( RealType weight)
    {
      this->m_OperatorOrder = weight;
    }
  RealType GetOperatorOrder() const
    {
      return this->m_OperatorOrder;
    }

  void SetNumberOfTimeSteps( unsigned int steps)
    {
      this->m_NumberOfTimeSteps = steps;
    }
  unsigned int GetNumberOfTimeSteps() const
    {
      return this->m_NumberOfTimeSteps;
    }
  // END: FLASH EDIT

private:
  ITK_DISALLOW_COPY_AND_ASSIGN(FLASHImageRegistrationMethod);

  RealType                                                        m_GaussianSmoothingVarianceForTheUpdateField;
  RealType                                                        m_GaussianSmoothingVarianceForTheTotalField;
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkFLASHImageRegistrationMethod.hxx"
#endif

#endif
