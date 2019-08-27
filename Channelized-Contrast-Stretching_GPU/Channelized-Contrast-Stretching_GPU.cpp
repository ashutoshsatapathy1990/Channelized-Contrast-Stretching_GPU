
/* -------------------------------------------------------------------------------------------------|
IMAGE CONTRAST ENHANCEMENT IN SPATIAL DOMAIN USING LINEAR, PIECEWISE LINEAR, EXP., LOG. STRETCHING. |
IMAGE CONTRAST ENHANCEMENT IN SPATIAL DOMAIN USING POWER LAW TRANSFORMATION.						|			    |
Contrast Enhancement.cpp : Defines the entry point for the console application.					    |
...................................................................................................*/

//++++++++++++++++++++++++++++++++++ START HEADER FILES +++++++++++++++++++++++++++++++++++++++++++++
// Include The Necesssary Header Files
// [Both In std. and non std. path]
#include "stdafx.h"
#include<stdio.h>
#include<conio.h>
#include<string.h>
#include<stdlib.h>
#include<complex>
#ifdef __APPLE__
#include<OpenCL\cl.h>
#else
#include<CL\cl.h>
#endif
#include<opencv\cv.h>
#include<opencv\highgui.h>
using namespace std;
using namespace cv;
//++++++++++++++++++++++++++++++++++ END HEADER FILES +++++++++++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++ LINEAR STRETCHING KERNEL ++++++++++++++++++++++++++++++++++++++++++
// OpenCL Linear Stretching Kernel Which Is Run For Every Work Item Created
const char *linear_kernel =
"#pragma OPENCL EXTENSION cl_khr_fp32 : enable													\n"	\
"#pragma OPENCL EXTENSION cl_khr_printf : enable												\n"	\
"__kernel																						\n"	\
"void linear_kernel (__global float* data,														\n"	\
"		double max,																				\n"	\
"		double min)																				\n"	\
"{																								\n"	\
"	// Get the index of work items																\n"	\
"	uint index = get_global_id(0);																\n"	\
"	data[index] = (data[index] - min) / (max - min) * 255.0;									\n"	\
"}																								\n"	\
"\n";
//+++++++++++++++++++++++++++++ END LINEAR STRETCHING KERNEL ++++++++++++++++++++++++++++++++++++++++

//++++++++++++++++++++++++++ PIECEWISE LINEAR STRETCHING KERNEL +++++++++++++++++++++++++++++++++++++
// OpenCL Piece wise Linear Kernel Which Is Run For Every Work Item Created
const char *pw_linear_kernel =
"#pragma OPENCL EXTENSION cl_khr_fp32 : enable													\n"	\
"#pragma OPENCL EXTENSION cl_khr_printf : enable												\n"	\
"__kernel																						\n"	\
"void pw_linear_kernel (__global float* data,													\n"	\
"		double max,																				\n"	\
"		double min,																				\n"	\
"		int Seg)																				\n"	\
"{																								\n"	\
"	// Get the index of work items																\n"	\
"	uint index = get_global_id(0);																\n"	\
"	float range = max - min + 1.0;																\n"	\
"	float pixel, final;																			\n"	\
"	pixel = data[index];																		\n"	\
"	if (Seg == 3){																				\n" \
"		float fold = range / 3;																	\n"	\
"		if (pixel < min && pixel <= min + fold - 1){											\n"	\
"			final = 63 / (fold - 1) * (pixel - min); }											\n"	\
"		else if (pixel > min + fold - 1 && pixel <= min + 2*fold - 1){							\n"	\
"			final = 128 / (fold) * (pixel - min - fold + 1) + 63; }								\n"	\
"		else{ final = 64 / (fold) * (pixel - min - 2 * fold + 1) + 191; } }						\n"	\
"	else if (Seg == 4){																			\n" \
"		float fold = range / 4;																	\n"	\
"		if (pixel < min && pixel <= min + fold - 1){											\n"	\
"			final = 31 / (fold - 1) * (pixel - min); }											\n"	\
"		else if (pixel > min + fold - 1 && pixel <= min + 2 * fold - 1){						\n"	\
"			final = 96 / (fold) * (pixel - min - fold + 1) + 31; }								\n"	\
"		else if (pixel > min + 2 * fold - 1 && pixel <= min + 3 * fold - 1){					\n"	\
"			final = 96 / (fold) * (pixel - min - 2 * fold + 1) + 127; }							\n"	\
"		else{ final = 32 / (fold) * (pixel - min - 3 * fold + 1) + 191; } }						\n"	\
"	else{																						\n" \
"		float fold = range / 5;																	\n"	\
"		if (pixel < min && pixel <= min + fold - 1){											\n"	\
"			final = 15 / (fold - 1) * (pixel - min); }											\n"	\
"		else if (pixel > min + fold - 1 && pixel <= min + 2 * fold - 1){						\n"	\
"			final = 64 / (fold) * (pixel - min - fold + 1) + 15; }								\n"	\
"		else if (pixel > min + 2 * fold - 1 && pixel <= min + 3 * fold - 1){					\n"	\
"			final = 96 / (fold) * (pixel - min - 2 * fold + 1) + 79; }							\n"	\
"		else if (pixel > min + 3 * fold - 1 && pixel <= min + 4 * fold - 1){					\n"	\
"			final = 64 / (fold) * (pixel - min - 3 * fold + 1) + 175; }							\n"	\
"		else{ final = 16 / (fold) * (pixel - min - 4 * fold + 1) + 239; } }						\n"	\
"	data[index] = final;																		\n"	\
"}																								\n"	\
"\n";
//+++++++++++++++++++++++ END PIECEWISE LINEAR STRETCHING KERNEL ++++++++++++++++++++++++++++++++++++

//++++++++++++++++++++++++++++ LOGARITHIMIC STRETCHING KERNEL +++++++++++++++++++++++++++++++++++++++
// OpenCL Logarithimic Kernel Which Is Run For Every Work Item Created
const char *log_kernel =
"#pragma OPENCL EXTENSION cl_khr_fp32 : enable													\n"	\
"#pragma OPENCL EXTENSION cl_khr_printf : enable												\n"	\
"__kernel																						\n"	\
"void log_kernel (__global float* data)															\n"	\
"{																								\n"	\
"	// Get the index of work items																\n"	\
"	uint index = get_global_id(0);																\n"	\
"	data[index] = 46.0 * log(data[index] + 1);													\n"	\
"}																								\n"	\
"\n";
//+++++++++++++++++++++++++ END LOGARITHIMIC STRETCHING KERNEL ++++++++++++++++++++++++++++++++++++++

//++++++++++++++++++++++++++++ EXPONENTIAL STRETCHING KERNEL +++++++++++++++++++++++++++++++++++++++
// OpenCL Exponential Kernel Which Is Run For Every Work Item Created
const char *exp_kernel =
"#define EXP 2.72																				\n" \
"#pragma OPENCL EXTENSION cl_khr_fp32 : enable													\n"	\
"#pragma OPENCL EXTENSION cl_khr_printf : enable												\n"	\
"__kernel																						\n"	\
"void exp_kernel (__global float* data)															\n"	\
"{																								\n"	\
"	// Get the index of work items																\n"	\
"	uint index = get_global_id(0);																\n"	\
"	data[index] = pow(EXP, 0.02173 * data[index]);												\n"	\
"}																								\n"	\
"\n";
//+++++++++++++++++++++++++ END EXPONENTIAL STRETCHING KERNEL ++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++ POWERLAW TRANSFORMATION KERNEL +++++++++++++++++++++++++++++++++++++++
// OpenCL Powerlaw Transformation Kernel Which Is Run For Every Work Item Created
const char *plt_kernel =
"#pragma OPENCL EXTENSION cl_khr_fp32 : enable													\n"	\
"#pragma OPENCL EXTENSION cl_khr_printf : enable												\n"	\
"__kernel																						\n"	\
"void plt_kernel (__global float* data,															\n"	\
"		float power,																			\n"	\
"		float Const)																			\n"	\
"{																								\n"	\
"	// Get the index of work items																\n"	\
"	uint index = get_global_id(0);																\n"	\
"	data[index] = Const * pow(data[index], power);												\n"	\
"}																								\n"	\
"\n";
//+++++++++++++++++++++++ END POWERLAW TRANSFORMATION KERNEL ++++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++ CREATE AND BUILD PROGRAM ++++++++++++++++++++++++++++++++++++++++
// Create and Build The Program Using Selected Enhancement's OpenCL Kernel Source Code.
cl_program enhancement_Program(cl_context context, const char *name, cl_device_id* device_list, cl_int clStatus)
{
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&name, NULL, &clStatus);
	clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
	return program;
}
//++++++++++++++++++++++++++++++ END CREATE AND BUILD PROGRAM +++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++ CREATE OPENCL KERNELS +++++++++++++++++++++++++++++++++++++++++++
// Create Multiple Kernels from The Program dedicated for [R, G, B] Channels of The Input Image.
cl_kernel* Enhancement_Kernel(cl_program program, cl_kernel* kernel, char* name, cl_int clStatus)
{
	kernel[0] = clCreateKernel(program, name, &clStatus);
	kernel[1] = clCreateKernel(program, name, &clStatus);
	kernel[2] = clCreateKernel(program, name, &clStatus);
	return kernel;
}
//+++++++++++++++++++++++++++++++ END CREATE OPENCL KERNELS +++++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++ SET KERNELS' ARGUMENTS ++++++++++++++++++++++++++++++++++++++++++
// Passing Arguments to Each Kernels Dedicated for Red, Green and Blue Channels [Contrast Enhancement]
void Kernel_Arg(cl_kernel* kernel, cl_mem RED_clmem, cl_mem GREEN_clmem, cl_mem BLUE_clmem, double* maxi, double* mini, int Seg, float power, float Const, int Select, cl_int clStatus)
{
	clStatus = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void *)&RED_clmem);
	clStatus = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), (void *)&GREEN_clmem);
	clStatus = clSetKernelArg(kernel[2], 0, sizeof(cl_mem), (void *)&BLUE_clmem);
	if (Select == 1)
	{
		clStatus = clSetKernelArg(kernel[0], 1, sizeof(double), (void *)&maxi[0]);
		clStatus = clSetKernelArg(kernel[1], 1, sizeof(double), (void *)&maxi[1]);
		clStatus = clSetKernelArg(kernel[2], 1, sizeof(double), (void *)&maxi[2]);
		clStatus = clSetKernelArg(kernel[0], 2, sizeof(double), (void *)&mini[0]);
		clStatus = clSetKernelArg(kernel[1], 2, sizeof(double), (void *)&mini[1]);
		clStatus = clSetKernelArg(kernel[2], 2, sizeof(double), (void *)&mini[2]);
	}
	else if (Select == 2)
	{
		clStatus = clSetKernelArg(kernel[0], 1, sizeof(double), (void *)&maxi[0]);
		clStatus = clSetKernelArg(kernel[1], 1, sizeof(double), (void *)&maxi[1]);
		clStatus = clSetKernelArg(kernel[2], 1, sizeof(double), (void *)&maxi[2]);
		clStatus = clSetKernelArg(kernel[0], 2, sizeof(double), (void *)&mini[0]);
		clStatus = clSetKernelArg(kernel[1], 2, sizeof(double), (void *)&mini[1]);
		clStatus = clSetKernelArg(kernel[2], 2, sizeof(double), (void *)&mini[2]);
		clStatus = clSetKernelArg(kernel[0], 3, sizeof(int), (void *)&Seg);
		clStatus = clSetKernelArg(kernel[1], 3, sizeof(int), (void *)&Seg);
		clStatus = clSetKernelArg(kernel[2], 3, sizeof(int), (void *)&Seg);
	}
	else if (Select == 5)
	{
		clStatus = clSetKernelArg(kernel[0], 1, sizeof(float), (void *)&power);
		clStatus = clSetKernelArg(kernel[1], 1, sizeof(float), (void *)&power);
		clStatus = clSetKernelArg(kernel[2], 1, sizeof(float), (void *)&power);
		clStatus = clSetKernelArg(kernel[0], 2, sizeof(float), (void *)&Const);
		clStatus = clSetKernelArg(kernel[1], 2, sizeof(float), (void *)&Const);
		clStatus = clSetKernelArg(kernel[2], 2, sizeof(float), (void *)&Const);
	}
}
//+++++++++++++++++++++++++++++++ END OPENCL KERNELS' ARGUMENTS +++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++++++++ EXECUTE KERNELS +++++++++++++++++++++++++++++++++++++++++++
//Execute the OpenCL Kernels for Enhancement of Each Channels Independetly.
void Exec_Kernel(cl_command_queue command_queue, cl_kernel* kernel, size_t global, size_t local, cl_int clStatus)
{
	clStatus = clEnqueueNDRangeKernel(command_queue, kernel[0], 1, NULL, &global, &local, 0, NULL, NULL);
	clStatus = clEnqueueNDRangeKernel(command_queue, kernel[1], 1, NULL, &global, &local, 0, NULL, NULL);
	clStatus = clEnqueueNDRangeKernel(command_queue, kernel[2], 1, NULL, &global, &local, 0, NULL, NULL);
}
//+++++++++++++++++++++++++++++++++++++ END EXECUTE KERNELS +++++++++++++++++++++++++++++++++++++++++

//++++++++++++++++++++++++++++++++++++++ RELEASE KERNELS ++++++++++++++++++++++++++++++++++++++++++++
//Release the OpenCL Kernels After The Image Enhancement.
void releaseKernel(cl_kernel* kernel, cl_int clStatus)
{
	clStatus = clReleaseKernel(kernel[0]);
	clStatus = clReleaseKernel(kernel[1]);
	clStatus = clReleaseKernel(kernel[2]);
}
//++++++++++++++++++++++++++++++++++  END RELEASE KERNELS ++++++++++++++++++++++++++++++++++++++++++++

//++++++++++++++++++++++++++++++++++ START MAIN PROGRAM +++++++++++++++++++++++++++++++++++++++++++++
int main()
{

	// Create Variables to Store Actual Values of [R, G, B] Cheannels.
	// 'Select' Variable f-or selection of The Enhancement Method.  
	// Arrays to store min and max intensities of each channels [R, G, B].
	Mat RGB_Image[3]; int Select; double min[3], max[3];

	// Initialize Clock Variable to compute Time Taken in millisecs
	clock_t start, end;
	float Time_Used;

	printf("Selection of The Enhancement Methods:\n [1]Linear\n [2]Piece-Wise Linear\n [3]Logarithimic\n [4]Exponential\n [5]Power-Law\n");
	scanf("%d", &Select);

	// Get The Platforms' Information
	cl_platform_id* platforms = NULL;
	cl_uint num_platforms;

	// Set up The Platforms
	cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
	clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);

	// Get The Device Lists and Choose The Device You Want to Run on.
	cl_device_id* device_list = NULL;
	cl_uint num_devices;
	clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
	device_list = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
	clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, num_devices, device_list, NULL);

	// Create an OpenCL Context for Each Device in The Platform
	cl_context context;
	context = clCreateContext(NULL, num_devices, device_list, NULL, NULL, &clStatus);

	// Create a Command Queue for Out of Order Execution in 0th Device.
	cl_command_queue command_queue_enhancement = clCreateCommandQueue(context, device_list[0], 0, &clStatus);

	// Read Original Image from The Given Path
	Mat Image = imread("img1.bmp", CV_LOAD_IMAGE_COLOR);

	// Check The Status of Image <Mat Variable>
	if (!Image.data)
	{
		printf("COULDN'T OPEN OR READ INPUT FILE");
		return -1;
	}

	// Display The Input Blurred Image
	namedWindow("ORIGINAL IMAGE", WINDOW_NORMAL);
	imshow("ORIGINAL IMAGE", Image);

	// Convert Each Channel Pixel Values from 8U to 32F
	// Copy Image.data values to RGB_Image for Extracting All Three Frames [R, G, B] 
	Image.convertTo(Image, CV_32F);
	split(Image, RGB_Image);

	// Store minimum and maximum values of [R, G, B] channels separately.
	for (int i = 0; i < 3; i++)
	{
		minMaxLoc(RGB_Image[i], &min[i], &max[i]);
		//RGB_Image[i].convertTo(RGB_Image[i], CV_8U);
	}

	// Create Host Buffers for Each Channels [R, G, B] as Image Size [H x W]
	float* RED_Frame = (float*)malloc(sizeof(float) * Image.rows * Image.cols);
	float* GREEN_Frame = (float*)malloc(sizeof(float) * Image.rows * Image.cols);
	float* BLUE_Frame = (float*)malloc(sizeof(float) * Image.rows * Image.cols);

	// Create OpenCL Device Buffers and Map to Host Buffers Separately Created for Each Color Channel.
	cl_mem RED_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * Image.rows * Image.cols, RED_Frame, &clStatus);
	cl_mem GREEN_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * Image.rows * Image.cols, GREEN_Frame, &clStatus);
	cl_mem BLUE_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * Image.rows * Image.cols, BLUE_Frame, &clStatus);

	// Get The Program's and Kernels' Information
	cl_program program_enhancement = NULL; 
	cl_kernel* kernel_enhancement = (cl_kernel*)malloc(sizeof(cl_kernel) * 3);

	// Selection of Contrast Enhancement : [1]Linear [2]PieceWise Linear [3]Logarithmic [4]Exponential [5]Power-Law.
	// Creation and Building of OpenCL Programs and Kernels for Each Channels [R, G, B].
	// Passing Arguments to Kernels for Contrast Enhancement based on Selected Method.
	switch (Select)
	{
	case 1: printf("You Selected Linear Stretching \n");
		program_enhancement = enhancement_Program(context, linear_kernel, device_list, clStatus);
		kernel_enhancement = Enhancement_Kernel(program_enhancement, kernel_enhancement, "linear_kernel", clStatus);
		Kernel_Arg(kernel_enhancement, RED_clmem, GREEN_clmem, BLUE_clmem, max, min, 3, 0.75, 3.996, Select, clStatus);
		break;
	case 2: printf("You Selected Piece-Wise Linear Stretching \n");
		program_enhancement = enhancement_Program(context, pw_linear_kernel, device_list, clStatus);
		kernel_enhancement = Enhancement_Kernel(program_enhancement, kernel_enhancement, "pw_linear_kernel", clStatus);
		Kernel_Arg(kernel_enhancement, RED_clmem, GREEN_clmem, BLUE_clmem, max, min, 3, 0.75, 3.996, Select, clStatus);
		break;
	case 3:	printf("You Selected Logarithimic Stretching \n");
		program_enhancement = enhancement_Program(context, log_kernel, device_list, clStatus);
		kernel_enhancement = Enhancement_Kernel(program_enhancement, kernel_enhancement, "log_kernel", clStatus);
		Kernel_Arg(kernel_enhancement, RED_clmem, GREEN_clmem, BLUE_clmem, max, min, 3, 0.75, 3.996, Select, clStatus);
		break;
	case 4:	printf("You Selected Exponential Stretching \n");
		program_enhancement = enhancement_Program(context, exp_kernel, device_list, clStatus);
		kernel_enhancement = Enhancement_Kernel(program_enhancement, kernel_enhancement, "exp_kernel", clStatus);
		Kernel_Arg(kernel_enhancement, RED_clmem, GREEN_clmem, BLUE_clmem, max, min, 3, 0.75, 3.996, Select, clStatus);
		break;
	case 5:	printf("You Selected Power-Law Transformation \n");
		program_enhancement = enhancement_Program(context, plt_kernel, device_list, clStatus);
		kernel_enhancement = Enhancement_Kernel(program_enhancement, kernel_enhancement, "plt_kernel", clStatus);
		Kernel_Arg(kernel_enhancement, RED_clmem, GREEN_clmem, BLUE_clmem, max, min, 3, 0.75, 3.996, Select, clStatus);
		break;
	default: printf("YOU HAVE SELECTED WRONG FILTER \n");
		break;
	}

	// Copy Image Data to Host Buffers for Each Channels Separately.
	memcpy(RED_Frame, RGB_Image[0].data, Image.rows * Image.cols * sizeof(float));
	memcpy(GREEN_Frame, RGB_Image[1].data, Image.rows * Image.cols * sizeof(float));
	memcpy(BLUE_Frame, RGB_Image[2].data, Image.rows * Image.cols * sizeof(float));

	//Execute the OpenCL Kernels for Boosting of Each Channels Independetly.
	Exec_Kernel(command_queue_enhancement, kernel_enhancement, Image.rows*Image.cols, 1024, clStatus);

	// Copy from Device Buffers to Host Buffers after Image Boosting Operation.
	clStatus = clEnqueueReadBuffer(command_queue_enhancement, RED_clmem, CL_TRUE, 0, Image.rows * Image.cols * sizeof(float), RED_Frame, 0, NULL, NULL);
	clStatus = clEnqueueReadBuffer(command_queue_enhancement, GREEN_clmem, CL_TRUE, 0, Image.rows * Image.cols * sizeof(float), GREEN_Frame, 0, NULL, NULL);
	clStatus = clEnqueueReadBuffer(command_queue_enhancement, BLUE_clmem, CL_TRUE, 0, Image.rows * Image.cols * sizeof(float), BLUE_Frame, 0, NULL, NULL);

	// Copy from Host Buffers to Image Variables for Each Channels Separately.
	memcpy(RGB_Image[0].data, RED_Frame, Image.rows * Image.cols * sizeof(float));
	memcpy(RGB_Image[1].data, GREEN_Frame, Image.rows * Image.cols * sizeof(float));
	memcpy(RGB_Image[2].data, BLUE_Frame, Image.rows * Image.cols * sizeof(float));

	// Merge All Three Channels to Construct Final Image
	merge(RGB_Image, 3, Image);
	Image.convertTo(Image, CV_8U);

	// Display Final Enhanced Image By High Boost FIlter
	namedWindow("ENHANCED IMAGE", WINDOW_NORMAL);
	imshow("ENHANCED IMAGE", Image);

	// Finally Release All OpenCL Allocated Objects and Buffers [Host & Device].
	releaseKernel(kernel_enhancement, clStatus);
	clStatus = clReleaseProgram(program_enhancement);
	clStatus = clReleaseMemObject(RED_clmem);
	clStatus = clReleaseMemObject(GREEN_clmem);
	clStatus = clReleaseMemObject(BLUE_clmem);
	clStatus = clReleaseCommandQueue(command_queue_enhancement);
	clStatus = clReleaseContext(context);
	free(RED_Frame);
	free(GREEN_Frame);
	free(BLUE_Frame);
	free(platforms);
	free(device_list);

	waitKey(0);
	return 0;
}
//+++++++++++++++++++++++++++++++++++ END MAIN PROGRAM ++++++++++++++++++++++++++++++++++++++++++++++
