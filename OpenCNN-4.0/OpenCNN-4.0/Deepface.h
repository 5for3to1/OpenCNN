#pragma once
#include <stdio.h>
#include <stdlib.h>
#include<iostream>
#include<fstream>
#include<string>
#include<iomanip>

//多线程相关头文件
#include <thread>  
#include <Windows.h>

#include<sstream>

#include<opencv2/opencv.hpp>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

using namespace std;
using namespace cv;

class Deepface
{
public:
	Deepface();
	~Deepface();

	void forward(string);

	void convolution(int, uchar *, bool debug);
	void convolution(int, bool debug);
	void slice_etlwise();
	void pooling();
	void full_connection(int, float *);

	cl_platform_id *platform_id;					//OpenCL平台
	cl_device_id device_id;							//OpenCL设备
	cl_uint ret_num_devices;						//平台数量
	cl_uint ret_num_platforms;						//设备数量

	cl_int ret;										//
	cl_context context;								//上下文
	cl_command_queue command_queue;					//命令队列
	cl_program program;								//内核程序对象
	

	int num_param_layer;							//所有的卷积层和全连接层
	int * N;										//卷积核的数量
	int * C;										//卷积核的通道数
	int * K;										//卷积核的尺寸
	float ** ptr_layers_params_w;					//各层 w 参数的指针
	float ** ptr_layers_params_b;					//各层 b 参数的指针
	cl_mem * layers_params_w_mem_obj;				//各层 w 缓冲区的指针
	cl_mem * layers_params_b_mem_obj;				//各层 b 缓冲区的指针

	int num_all_layers;								//所有的层数
	int index_current_layer;						//当前层的号码
	cl_mem * layers_data;							//各层的数据输入输出缓冲区

	int ** layer_param;								//每一层的规格参数
	cl_mem * layer_param_mem_obj;					//设备内存中参数缓冲区

	clock_t total_start, total_end;
};
