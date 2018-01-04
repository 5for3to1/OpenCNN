#pragma once
#include <stdio.h>
#include <stdlib.h>
#include<iostream>
#include<fstream>
#include<string>
#include<iomanip>

//���߳����ͷ�ļ�
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

	cl_platform_id *platform_id;					//OpenCLƽ̨
	cl_device_id device_id;							//OpenCL�豸
	cl_uint ret_num_devices;						//ƽ̨����
	cl_uint ret_num_platforms;						//�豸����

	cl_int ret;										//
	cl_context context;								//������
	cl_command_queue command_queue;					//�������
	cl_program program;								//�ں˳������
	

	int num_param_layer;							//���еľ�����ȫ���Ӳ�
	int * N;										//����˵�����
	int * C;										//����˵�ͨ����
	int * K;										//����˵ĳߴ�
	float ** ptr_layers_params_w;					//���� w ������ָ��
	float ** ptr_layers_params_b;					//���� b ������ָ��
	cl_mem * layers_params_w_mem_obj;				//���� w ��������ָ��
	cl_mem * layers_params_b_mem_obj;				//���� b ��������ָ��

	int num_all_layers;								//���еĲ���
	int index_current_layer;						//��ǰ��ĺ���
	cl_mem * layers_data;							//����������������������

	int ** layer_param;								//ÿһ��Ĺ�����
	cl_mem * layer_param_mem_obj;					//�豸�ڴ��в���������

	clock_t total_start, total_end;
};
