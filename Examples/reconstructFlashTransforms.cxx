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
#include "FftOper.h"
#include "IOpers.h"
#include "FOpers.h"
#include "IFOpers.h"
#include "HFOpers.h"
#include "Reduction.h"
#include "FluidKernelFFT.h"

#include "itkWarpVectorImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkComposeDisplacementFieldsImageFilter.h"


namespace ants
{

const int ImageDimension = 3; // TODO: generalize
bool DoRungeKuttaForIntegration = false; // TODO: add to interface
using ComplexVectorType = itk::Vector< float, ImageDimension*2 > ;
using ComplexFieldType = itk::Image< ComplexVectorType, ImageDimension >;
using ReferenceImageType = itk::Image< float, ImageDimension >;
using DisplacementVectorType = itk::Vector< float, ImageDimension >;
using DisplacementFieldType = itk::Image< DisplacementVectorType, ImageDimension >;

// v0, reference image, their spatial info, and fftoper must be global (used across entire program)
FieldComplex3D * v0;
typename ComplexFieldType::SizeType v0_dims;
typename ReferenceImageType::Pointer reference_image;
GridInfo grid;
FftOper * fftoper;

FieldComplex3D * JacX;
FieldComplex3D * JacY;
FieldComplex3D * JacZ;
Field3D * identity;
float * idxf;
float * idyf;
float * idzf;
FieldComplex3D * adScratch1;
FieldComplex3D * adScratch2;

typename DisplacementFieldType::Pointer forwardDisplacement;


// /************************* numerical method and operator functions *****************************
// ************************************************************************************************/
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

void pycaToItkVectorField(DisplacementFieldType & itkField, Field3D & pycaField)
{
  itk::ImageRegionIterator<DisplacementFieldType> Iter(&itkField, (&itkField)->GetRequestedRegion());
  itk::SizeValueType count = 0;
  DisplacementVectorType itkVector;
  Vec3Df pycaVector, pycaIdentity;
  for( Iter.GoToBegin(); !Iter.IsAtEnd(); ++Iter )
    {
    pycaVector = pycaField.get(count);
    pycaIdentity = identity->get(count++);
    for( itk::SizeValueType d = 0; d < ImageDimension; d++ )
      itkVector[d] = pycaVector[d] - pycaIdentity[d];
    Iter.Set(itkVector);
    }
}


// adTranspose
// spatial domain: K(Dv^T Lw + div(L*w x v))
// K*(CorrComplexFFT(CD*v^T, L*w) + TensorCorr(L*w, v) * D)
void adTranspose(FieldComplex3D & adTransvw, const FieldComplex3D & v, const FieldComplex3D & w)
{
     Mul_FieldComplex(*(adScratch1),
                      *(fftoper->Lcoeff), w);
     JacobianT(*(JacX),
               *(JacY),
               *(JacZ),
               *(fftoper->CDcoeff), v); 
     fftoper->ConvolveComplexFFT(adTransvw, 1,
                                         *(JacX),
                                         *(JacY),
                                         *(JacZ),
                                         *(adScratch1));
     fftoper->CorrComplexFFT(*(adScratch2), v,
                                     *(adScratch1),
                                     *(fftoper->CDcoeff));
     AddI_FieldComplex(adTransvw, *(adScratch2), 1.0);
     MulI_FieldComplex(adTransvw, *(fftoper->Kcoeff));
}


void RungeKuttaStep(FieldComplex3D * sf1, FieldComplex3D * sf2, FieldComplex3D * sf3,
                 FieldComplex3D * vff1, FieldComplex3D * vff2, float dt)
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


void EulerStep(FieldComplex3D * sf1, FieldComplex3D * vff1, FieldComplex3D * vff2, float dt)
// sf: scratch field (preallocated memory to store intermediate calculations)
// vff: velocity flow field (vff1: i-1, vff2: i)
// dt: time step size
{
  // v0 = v0 - dt * adTranspose(v0, v0)
  Copy_FieldComplex(*vff2, *vff1);
  adTranspose(*sf1, *vff2, *vff2);
  AddI_FieldComplex(*vff2, *sf1, -dt);
}


void AdvectionStep(FieldComplex3D * JacX, FieldComplex3D * JacY, FieldComplex3D * JacZ,
                FieldComplex3D * sf1, FieldComplex3D * sf2, FieldComplex3D * vff,
                float dt)
// sf: scratch field (preallocated memory to store intermediate calculations)
// vff: velocity flow field
// dt: time step size
{
  Jacobian(*JacX, *JacY, *JacZ, *(fftoper->CDcoeff), *sf1);
  fftoper->ConvolveComplexFFT(*sf2, 0, *JacX, *JacY, *JacZ, *vff);
  AddIMul_FieldComplex(*sf1, *sf2, *vff, -dt);
}


void ForwardTransformStep(DisplacementFieldType * displacement,
	                        DisplacementFieldType * spatialVitk,
	                        Field3D * spatialV,
	                        FieldComplex3D * vff,
	                        float dt)
// displacement: ITK displacement field object holding current forward displacement field
// spatialVitk: ITK displacement field object to hold spatial velocity as an ITK object
// spatialV: pyca Field3D to hold velocity in spatial domain
// vff: velocity flow field for current time step
// dt: time step size
{
  // phi_t = phi_{t-1} + dt * invFFT(v_t) o phi_{t-1}
  // get spatial velocity as ITK object
  // adding identity, then stripping it off in pycaToItkVectorField is wasteful, but we only compute forward transform once
  fftoper->fourier2spatial_addH(*spatialV, *vff, idxf, idyf, idzf);
  pycaToItkVectorField(*spatialVitk, *spatialV);
  // move Eulerian velocity to Lagrangian velocity with current value of displacement
  using warpType = itk::WarpVectorImageFilter<DisplacementFieldType, DisplacementFieldType, DisplacementFieldType>;
  typename warpType::Pointer warper = warpType::New();
  warper->SetOutputSpacing(displacement->GetSpacing());
  warper->SetOutputOrigin(displacement->GetOrigin());
  warper->SetOutputDirection(displacement->GetDirection());
  warper->SetInput( spatialVitk ); // possibly need to dereference these?
  warper->SetDisplacementField( displacement );
  // multiply by time step
  using MultiplierType = itk::MultiplyImageFilter<DisplacementFieldType, ReferenceImageType, DisplacementFieldType>;
  typename MultiplierType::Pointer multiplier = MultiplierType::New();
  multiplier->SetInput( warper->GetOutput() );
  multiplier->SetConstant( dt );
  multiplier->Update();
  // add to current value of transform
  using ComposerType = itk::ComposeDisplacementFieldsImageFilter<DisplacementFieldType>;
  typename ComposerType::Pointer composer = ComposerType::New();
  composer->SetDisplacementField( multiplier->GetOutput() );
  composer->SetWarpingField( displacement );
  composer->Update();
  forwardDisplacement = composer->GetOutput();
}


/************************* geodesic shooting functions *****************************************
************************************************************************************************/

void ForwardIntegration(Field3D * inverseTransformSpatial,
	                      FieldComplex3D * v0,
	                      int time_steps)
{
	// initialize intermediate variables
  float time_step_size = 1.0/time_steps;

  FieldComplex3D ** velocityFlowField = new FieldComplex3D * [time_steps + 1];
  for (int i = 0; i <= time_steps; i++)
    velocityFlowField[i] = new FieldComplex3D(v0_dims[0], v0_dims[1], v0_dims[2]);

  JacX = new FieldComplex3D(v0_dims[0], v0_dims[1], v0_dims[2]);
  JacY = new FieldComplex3D(v0_dims[0], v0_dims[1], v0_dims[2]);
  JacZ = new FieldComplex3D(v0_dims[0], v0_dims[1], v0_dims[2]);

  // identity field in the Fourier domain, pyca
  identity = new Field3D(grid, MEM_HOST);
  Opers::SetToIdentity(*(identity));
  idxf = new float[2 * fftoper->fsxFFT * fftoper->fsy * fftoper->fsz];
  idyf = new float[2 * fftoper->fsxFFT * fftoper->fsy * fftoper->fsz];
  idzf = new float[2 * fftoper->fsxFFT * fftoper->fsy * fftoper->fsz];
  fftoper->spatial2fourier_F(idxf, idyf, idzf, *(identity));

  // extra space to help adTranspose calculations
  adScratch1 = new FieldComplex3D(v0_dims[0], v0_dims[1], v0_dims[2]);
  adScratch2 = new FieldComplex3D(v0_dims[0], v0_dims[1], v0_dims[2]);

  // velocity in spatial domain, itk
  typename DisplacementFieldType::Pointer spatialVitk = DisplacementFieldType::New();
  spatialVitk->CopyInformation( reference_image );
  spatialVitk->SetRegions( reference_image->GetRequestedRegion() );
  spatialVitk->Allocate();

  // Fourier domain scratch fields, pyca
  FieldComplex3D * inverseTransformFourier = new FieldComplex3D(v0_dims[0], v0_dims[1], v0_dims[2]);
  FieldComplex3D * scratchFieldFourier = new FieldComplex3D(v0_dims[0], v0_dims[1], v0_dims[2]);

  // some scratch memory in the spatial domain, pyca
  Field3D * scratchV1 = new Field3D(grid, MEM_HOST);
  Field3D * inverseTransformSpatialDomain = new Field3D(grid, MEM_HOST);


  // obtain velocity flow with EPDiff in Fourier domain
  Copy_FieldComplex(*(velocityFlowField[0]), *v0);
  if (DoRungeKuttaForIntegration)
  {
  	FieldComplex3D * scratchFieldFourier2 = new FieldComplex3D(v0_dims[0], v0_dims[1], v0_dims[2]);
  	FieldComplex3D * scratchFieldFourier3 = new FieldComplex3D(v0_dims[0], v0_dims[1], v0_dims[2]);
    for (int i = 1; i <= time_steps; i++)
      RungeKuttaStep(scratchFieldFourier, scratchFieldFourier2, scratchFieldFourier3,
                     velocityFlowField[i-1], velocityFlowField[i],
                     time_step_size);
  }
  else
  {
    for (int i = 1; i <= time_steps; i++)
      EulerStep(scratchFieldFourier,
                velocityFlowField[i-1],
                velocityFlowField[i],
                time_step_size);
  }

  // integrate velocity flow through advection equation to obtain inverse of path endpoint
  inverseTransformFourier->initVal(complex<float>(0.0, 0.0)); // displacement field
  for (int i = 0; i < time_steps; i++)
    AdvectionStep(JacX, JacY, JacZ,
                  inverseTransformFourier,
                  scratchFieldFourier,
                  velocityFlowField[i],
                  time_step_size);

  // obtain spatial domain transform, convert to ITK field
  fftoper->fourier2spatial_addH(*inverseTransformSpatial,
                                *inverseTransformFourier,
                                idxf, idyf, idzf);

	for (int i = 0; i < time_steps; i++)
	  ForwardTransformStep(forwardDisplacement,
	                       spatialVitk,
	                       scratchV1,
	                       velocityFlowField[i],
	                       time_step_size);
}





void reconstructFlashTransforms(char * velocity_field_filename,
      	                        char * reference_image_filename,
      	                        std::string output_prefix,
      	                        int time_steps,
      	                        float laplace_weight,
      	                        int operator_order)
{
  // Read initial velocity, convert to pyca object
  using ComplexFieldReaderType = itk::ImageFileReader<ComplexFieldType> ;
  typename ComplexFieldReaderType::Pointer velocity_reader = ComplexFieldReaderType::New();
  velocity_reader->SetFileName(velocity_field_filename);
  velocity_reader->Update();
  typename ComplexFieldType::Pointer v0_itk = velocity_reader->GetOutput();
  v0_dims = v0_itk->GetLargestPossibleRegion().GetSize();
  v0 = new FieldComplex3D(v0_dims[0], v0_dims[1], v0_dims[2]);
  itkToPycaComplexVectorField(*v0_itk, *v0);

  // Read reference image, convert spatial domain parameters to pyca GridInfo object
  // TODO: may not actually need to read in the data, maybe IOFactory thing (see WarpImageMultiTransform, before call to actual function)
  //       it can grab the kinds of params I need without reading in the image data
  using ImageReaderType = itk::ImageFileReader<ReferenceImageType> ;
  typename ImageReaderType::Pointer reference_reader = ImageReaderType::New();
  reference_reader->SetFileName(reference_image_filename);
  reference_reader->Update();
  reference_image = reference_reader->GetOutput();
  typename ReferenceImageType::SizeType reference_dims = reference_image->GetLargestPossibleRegion().GetSize();
  typename ReferenceImageType::SpacingType reference_spacing = reference_image->GetSpacing();
  typename ReferenceImageType::PointType reference_origin = reference_image->GetOrigin();
  grid = GridInfo(Vec3Di(reference_dims[0], reference_dims[1], reference_dims[2]),
                  Vec3Df(reference_spacing[0], reference_spacing[1], reference_spacing[2]),
                  Vec3Df(reference_origin[0], reference_origin[1], reference_origin[2]));

  // initialize pyca field for inverse transform and itk field for forward transform, also fft operator
  Field3D * inverseTransformSpatial = new Field3D(grid, MEM_HOST);
  forwardDisplacement = DisplacementFieldType::New();
  forwardDisplacement->CopyInformation( reference_image );
  forwardDisplacement->SetRegions( reference_image->GetRequestedRegion() );
  forwardDisplacement->Allocate();
  float identity_weight = 1.0;
  fftoper = new FftOper(laplace_weight, identity_weight, operator_order,
                        grid, v0_dims[0], v0_dims[1], v0_dims[2]);
  fftoper->FourierCoefficient();

  // run forward integration
  ForwardIntegration(inverseTransformSpatial,
  	                 v0, time_steps);

  // convert results to itk objects
  typename DisplacementFieldType::Pointer inverseDisplacement = DisplacementFieldType::New();
  inverseDisplacement->CopyInformation( reference_image );
  inverseDisplacement->SetRegions( reference_image->GetRequestedRegion() );
  inverseDisplacement->Allocate();
  pycaToItkVectorField(*inverseDisplacement, *inverseTransformSpatial);

  // write out reconstructed displacement fields
  using DisplacementFieldWriterType = itk::ImageFileWriter<DisplacementFieldType> ;
  typename DisplacementFieldWriterType::Pointer writer = DisplacementFieldWriterType::New();
  writer->SetFileName(output_prefix + "-Warp.nii.gz");
  writer->SetInput(inverseDisplacement);
  writer->Update();
  writer->SetFileName(output_prefix + "-InverseWarp.nii.gz");
  writer->SetInput(forwardDisplacement);
  writer->Update();
}


// entry point for the library; parameter 'args' is equivalent to 'argv' in (argc,argv) of commandline parameters to
// 'main()'
int reconstructFlashTransforms( std::vector<std::string> args, std::ostream* /*out_stream = nullptr */ )
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
  argv[argc] = nullptr;
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
    std::cout << argv[0] << " V0Path ReferencePath outputs_prefix time_steps laplace_weight operator_order\n" << std::endl;
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
  std::string    output_prefix = argv[3];
  const int      time_steps = std::stoi(argv[4]);
  const float    laplace_weight = atof(argv[5]);
  const int      operator_order = std::stoi(argv[6]);

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
