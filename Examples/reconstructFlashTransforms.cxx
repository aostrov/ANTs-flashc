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

#include "antsUtilities.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "FieldComplex3D.h"


namespace ants
{

const int ImageDimension = 3; // TODO: generalize
typedef itk::Vector< float, ImageDimension*2 >                           ComplexVectorType;
typedef itk::Image< ComplexVectorType, ImageDimension >                  ComplexFieldType;
typedef itk::Vector< float, ImageDimension >                             PixelType;
typedef itk::Image< PixelType, ImageDimension >                          ImageType;


void itkToPycaComplexVectorField(ComplexFieldType & itkField, FieldComplex3D & pycaField)
{
  itk::ImageRegionIterator<ComplexFieldType> Iter(&itkField, (&itkField)->GetRequestedRegion());
  ComplexVectorType itkVector;
  complex<float> component;
  itk::SizeValueType count = 0;
  for( Iter.GoToBegin(); !Iter.IsAtEnd(); ++Iter )
    {
    itkVector = Iter.Get();
    for( itk::SizeValueType d = 0; d < ImageDimension; d++ )
      {
      component.real(itkVector[2*d]);
      component.imag(itkVector[2*d+1]);
      pycaField.data[count] = component;
      count++;
      }
    }
}

// void pycaToItkVectorField(DisplacementFieldType & itkField, Field3D & pycaField)
// {
//   ImageRegionIterator<DisplacementFieldType> Iter(&itkField, (&itkField)->GetRequestedRegion());
//   SizeValueType count = 0;
//   DisplacementVectorType itkVector;
//   Vec3Df pycaVector, pycaIdentity;
//   for( Iter.GoToBegin(); !Iter.IsAtEnd(); ++Iter )
//     {
//     pycaVector = pycaField.get(count);
//     pycaIdentity = this->m_identity->get(count++);
//     for( SizeValueType d = 0; d < ImageDimension; d++ )
//       itkVector[d] = pycaVector[d] - pycaIdentity[d];
//     Iter.Set(itkVector);
//     }
// }


/************************* geodesic shooting functions *****************************************
************************************************************************************************/

void ForwardIntegration()
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

  // obtain spatial domain transform, convert to ITK field
  this->m_fftoper->fourier2spatial_addH(*(this->m_phiinv),
                                        *(this->m_scratch1),
                                        this->idxf, this->idyf, this->idzf);
  pycaToItkVectorField(*(this->m_movingToFixedInverseDisplacement), *(this->m_phiinv));

	for (int i = 0; i < this->m_NumberOfTimeSteps; i++)
	  ForwardTransformStep(this->m_fixedToMovingInverseDisplacement,
	                       this->m_spatialVitk,
	                       this->m_scratchV1,
	                       this->m_VelocityFlowField[i],
	                       this->m_TimeStepSize);
}


// /************************* numerical method and operator functions *****************************
// ************************************************************************************************/

// void RungeKuttaStep(FieldComplex3D * sf1, FieldComplex3D * sf2, FieldComplex3D * sf3,
//                  FieldComplex3D * vff1, FieldComplex3D * vff2, float dt)
// // sf: scratch field (preallocated memory to store intermediate calculations)
// // vff: velocity flow field (vff1: i-1, vff2: i)
// // dt: time step size
// {
//   // v1 = v0 - (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
//   // k1
//   adTranspose(*sf1, *vff1, *vff1);
//   // partially update v1 = v0 - (dt/6)*k1
//   Copy_FieldComplex(*vff2, *vff1);
//   AddI_FieldComplex(*vff2, *sf1, -dt / 6.0);
//   // k2
//   Copy_FieldComplex(*sf3, *vff1);
//   AddI_FieldComplex(*sf3, *sf1, -0.5*dt);
//   adTranspose(*sf2, *sf3, *sf3);
//   // partially update v1 = v1 - (dt/3)*k2
//   AddI_FieldComplex(*vff2, *sf2, -dt / 3.0);
//   // k3 (stored in scratch1)
//   Copy_FieldComplex(*sf3, *vff1);
//   AddI_FieldComplex(*sf3, *sf2, -0.5*dt);
//   adTranspose(*sf1, *sf3, *sf3);
//   // partially update v1 = v1 - (dt/3)*k3
//   AddI_FieldComplex(*vff2, *sf1, -dt / 3.0);
//   // k4 (stored in scratch2)
//   Copy_FieldComplex(*sf3, *vff1);
//   AddI_FieldComplex(*sf3, *sf1, -dt);
//   adTranspose(*sf2, *sf3, *sf3);
//   // finish updating v1 = v1 - (dt/6)*k4
//   AddI_FieldComplex(*vff2, *sf2, -dt / 6.0);
// }


// void EulerStep(FieldComplex3D * sf1, FieldComplex3D * vff1, FieldComplex3D * vff2, float dt)
// // sf: scratch field (preallocated memory to store intermediate calculations)
// // vff: velocity flow field (vff1: i-1, vff2: i)
// // dt: time step size
// {
//   // v0 = v0 - dt * adTranspose(v0, v0)
//   Copy_FieldComplex(*vff2, *vff1);
//   adTranspose(*sf1, *vff2, *vff2);
//   AddI_FieldComplex(*vff2, *sf1, -dt);
// }


// void AdvectionStep(FieldComplex3D * JacX, FieldComplex3D * JacY, FieldComplex3D * JacZ,
//                 FieldComplex3D * sf1, FieldComplex3D * sf2, FieldComplex3D * vff,
//                 float dt)
// // sf: scratch field (preallocated memory to store intermediate calculations)
// // vff: velocity flow field
// // dt: time step size
// {
//   Jacobian(*JacX, *JacY, *JacZ, *(this->m_fftoper->CDcoeff), *sf1);
//   this->m_fftoper->ConvolveComplexFFT(*sf2, 0, *JacX, *JacY, *JacZ, *vff);
//   AddIMul_FieldComplex(*sf1, *sf2, *vff, -dt);
// }


// void ForwardTransformStep(DisplacementFieldType * displacement, DisplacementFieldType * spatialVitk, Field3D * spatialV, FieldComplex3D * vff, float dt)
// // displacement: ITK displacement field object holding current forward displacement field
// // spatialVitk: ITK displacement field object to hold spatial velocity as an ITK object
// // spatialV: pyca Field3D to hold velocity in spatial domain
// // vff: velocity flow field for current time step
// // dt: time step size
// {
//   // phi_t = phi_{t-1} + dt * invFFT(v_t) o phi_{t-1}
//   // get spatial velocity as ITK object
//   // adding identity, then stripping it off in pycaToItkVectorField is wasteful, but we only compute forward transform once
//   this->m_fftoper->fourier2spatial_addH(*spatialV, *vff, this->idxf, this->idyf, this->idzf);
//   pycaToItkVectorField(*spatialVitk, *spatialV);
//   // move Eulerian velocity to Lagrangian velocity with current value of displacement
//   using warpType = WarpVectorImageFilter<DisplacementFieldType, DisplacementFieldType, DisplacementFieldType>;
//   typename warpType::Pointer warper = warpType::New();
//   warper->SetOutputSpacing(displacement->GetSpacing());
//   warper->SetOutputOrigin(displacement->GetOrigin());
//   warper->SetOutputDirection(displacement->GetDirection());
//   warper->SetInput( spatialVitk ); // possibly need to dereference these?
//   warper->SetDisplacementField( displacement );
//   // multiply by time step
//   using RealImageType = Image<float, ImageDimension>;
//   using MultiplierType = MultiplyImageFilter<DisplacementFieldType, RealImageType, DisplacementFieldType>;
//   typename MultiplierType::Pointer multiplier = MultiplierType::New();
//   multiplier->SetInput( warper->GetOutput() );
//   multiplier->SetConstant( dt );
//   multiplier->Update();
//   // add to current value of transform
//   using ComposerType = ComposeDisplacementFieldsImageFilter<DisplacementFieldType>;
//   typename ComposerType::Pointer composer = ComposerType::New();
//   composer->SetDisplacementField( multiplier->GetOutput() );
//   composer->SetWarpingField( displacement );
//   composer->Update();
//   this->m_fixedToMovingInverseDisplacement = composer->GetOutput();
// }


// // adTranspose
// // spatial domain: K(Dv^T Lw + div(L*w x v))
// // K*(CorrComplexFFT(CD*v^T, L*w) + TensorCorr(L*w, v) * D)
// void adTranspose(FieldComplex3D & adTransvw, const FieldComplex3D & v, const FieldComplex3D & w)
// {
//      Mul_FieldComplex(*(this->m_adScratch1),
//                       *(this->m_fftoper->Lcoeff), w);
//      JacobianT(*(this->m_JacX),
//                *(this->m_JacY),
//                *(this->m_JacZ),
//                *(this->m_fftoper->CDcoeff), v); 
//      this->m_fftoper->ConvolveComplexFFT(adTransvw, 1,
//                                          *(this->m_JacX),
//                                          *(this->m_JacY),
//                                          *(this->m_JacZ),
//                                          *(this->m_adScratch1));
//      this->m_fftoper->CorrComplexFFT(*(this->m_adScratch2), v,
//                                      *(this->m_adScratch1),
//                                      *(this->m_fftoper->CDcoeff));
//      AddI_FieldComplex(adTransvw, *(this->m_adScratch2), 1.0);
//      MulI_FieldComplex(adTransvw, *(this->m_fftoper->Kcoeff));
// }


void reconstructFlashTransforms(char * velocity_field_filename,
      	                        char * reference_image_filename,
      	                        char * output_prefix,
      	                        int time_steps,
      	                        float laplace_weight,
      	                        int operator_order)
{

  // Read initial velocity
  typedef itk::ImageFileReader<ComplexFieldType> ComplexFieldReaderType;
  typename ComplexFieldReaderType::Pointer velocity_reader = ComplexFieldReaderType::New();
  velocity_reader->SetFileName(velocity_field_filename);
  velocity_reader->Update();
  typename ComplexFieldType::Pointer v0_itk = velocity_reader->GetOutput();
  typename ComplexFieldType::SizeType v0_dims = v0_itk->GetLargestPossibleRegion().GetSize();

  // Read reference image
  // TODO: may not actually need to read in the data, maybe IOFactory thing (see WarpImageMultiTransform, before call to actual function)
  //       it can grab the kinds of params I need without reading in the image data
  typedef itk::ImageFileReader<ImageType> ImageReaderType;
  typename ImageReaderType::Pointer reference_reader = ImageReaderType::New();
  reference_reader->SetFileName(reference_image_filename);
  reference_reader->Update();
  typename ImageType::Pointer reference_image = reference_reader->GetOutput();
  typename ImageType::SizeType reference_dims = reference_image->GetLargestPossibleRegion().GetSize();
  typename ImageType::SpacingType reference_spacing = reference_image->GetSpacing();
  typename ImageType::PointType reference_origin = reference_image->GetOrigin();

  // convert V0 to pyca FieldComplex3D object
  FieldComplex3D * v0 = new FieldComplex3D(v0_dims[0], v0_dims[1], v0_dims[2]);  // TODO: generalize for image dimension?
  itkToPycaComplexVectorField(*v0_itk, *v0);

  // set up objects for forward integration



 

  // typedef itk::ImageFileWriter<ImageType> ImageFileWriterType;
  // typename ImageFileWriterType::Pointer writer_img = ImageFileWriterType::New();
  // if( img_ref )
  //   {
  //   img_output->SetDirection(img_ref->GetDirection() );
  //   }
  // writer_img->SetFileName(output_image_filename);
  // writer_img->SetInput(img_output);
  // writer_img->Update();
}


// entry point for the library; parameter 'args' is equivalent to 'argv' in (argc,argv) of commandline parameters to
// 'main()'
int reconstructFlashTransforms( std::vector<std::string> args, std::ostream* /*out_stream = ITK_NULLPTR */ )
{
  // put the arguments coming in as 'args' into standard (argc,argv) format;
  // 'args' doesn't have the command name as first, argument, so add it manually;
  // 'args' may have adjacent arguments concatenated into one argument,
  // which the parser should handle
  args.insert( args.begin(), "reconstructFlashTransforms" );

  int     argc = args.size();
  char* * argv = new char *[args.size() + 1];
  for( unsigned int i = 0; i < args.size(); ++i )
    {
    // allocate space for the string plus a null character
    argv[i] = new char[args[i].length() + 1];
    std::strncpy( argv[i], args[i].c_str(), args[i].length() );
    // place the null character in the end
    argv[i][args[i].length()] = '\0';
    }
  argv[argc] = ITK_NULLPTR;
    // class to automatically cleanup argv upon destruction
  class Cleanup_argv
  {
	public:
	    Cleanup_argv( char* * argv_, int argc_plus_one_ ) : argv( argv_ ), argc_plus_one( argc_plus_one_ )
	    {
	    }

	    ~Cleanup_argv()
	    {
	      for( unsigned int i = 0; i < argc_plus_one; ++i )
	        {
	        delete[] argv[i];
	        }
	      delete[] argv;
	    }

	private:
	    char* *      argv;
	    unsigned int argc_plus_one;
  };
    Cleanup_argv cleanup_argv( argv, argc + 1 );

  // antscout->set_stream( out_stream );

  if( argc <= 3 )
    {
    std::cout <<  "\nUsage:\n" << std::endl;
    std::cout << argv[0] << "V0Path ReferencePath outputs_prefix\n" << std::endl;
    std::cout << "Reconstructs Warp and InverseWarp from FLASH transform initial velocity\nrequires reference image\n" << std::endl;
    if( argc >= 2 &&
        ( std::string( argv[1] ) == std::string("--help") || std::string( argv[1] ) == std::string("-h") ) )
      {
      return EXIT_SUCCESS;
      }
    return EXIT_FAILURE;
    }

  // TODO: make interface better, e.g. use flags, enforce correct input types, pull argument parsing into separate function
  char *         velocity_field_filename = argv[1];
  char *         reference_image_filename = argv[2];
  char *         output_prefix = argv[3];
  const int      time_steps = atoi(argv[4]);
  const float    laplace_weight = atof(argv[5]);
  const int      operator_order = atoi(argv[6]);

  std::cout << "velocity_field_filename: " << velocity_field_filename << std::endl;
  std::cout << "reference_image_filename: " << reference_image_filename << std::endl;
  std::cout << "output_prefix: " << output_prefix << std::endl;
  std::cout << "time_steps: " << time_steps << std::endl;
  std::cout << "laplace_weight: " << laplace_weight << std::endl;
  std::cout << "operator_order: " << operator_order << std::endl;

  try
    {
      reconstructFlashTransforms(velocity_field_filename,
      	                         reference_image_filename,
      	                         output_prefix,
      	                         time_steps,
      	                         laplace_weight,
      	                         operator_order);
    }
  catch( itk::ExceptionObject & e )
    {
    std::cout << "Exception caught during reconstructFlashTransforms." << std::endl;
    std::cout << e << std::endl;
    return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
}
} // namespace ants
