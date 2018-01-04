#include "Deepface.h"


//网络前向操作
void Deepface::forward(string output_feature_name)
{
	int index_params_layer;							//带参数的层号码

	//命令队列同步
	ret = clFinish(command_queue);

	//开始计时
	clock_t start, end;								//时钟计时
	start = clock();

	//计时
	clock_t begin_count, end_count;
	double time_count;

	float * full_connection_result = (float*)malloc(sizeof(float)* 512);			//最终的512维特征向量

	//循环 iters 次
	int iters = 1;
	for (int i = 0; i < iters; i++)
	{

		/***************卷积层: conv1*************/
		//初始化带参数的层号
		index_params_layer = 0;

		//计时开始
		begin_count = clock();

		//调用Mat类读图片
		Mat src_img = imread("data_jjw/gray_128_128.jpg", 0);			//读入一张灰度图
		if (src_img.empty())
		{
			cout << "image read failure" << endl;
			return;
		}

		convolution(index_params_layer, src_img.data, 0);					//第一次卷积

		cout << endl << "------conv1------" << endl;

		end_count = clock();
		time_count = (double)(end_count - begin_count) / CLOCKS_PER_SEC;
		cout << "conv cost " << time_count << " s" << endl << endl;


		/***************slice1和etlwise1层**************/

		slice_etlwise();

		//输出slice和etlwise层计算结果
		cout << "------slice11 etlwise11------" << endl;


		/***************池化层: pool1************/

		pooling();

		cout << endl << "------pool1------" << endl;


		/*********************conv2a*********************/
		//切换卷积层
		index_params_layer++;

		begin_count = clock();

		convolution(index_params_layer, 0);

		cout << endl << "------conv2a------" << endl;

		end_count = clock();
		time_count = (double)(end_count - begin_count) / CLOCKS_PER_SEC;
		cout << "conv cost " << time_count << " s" << endl << endl;


		/*******************slice2a和etlwise2a****************/

		slice_etlwise();

		cout << "------slice2a etlwise2a------" << endl;


		/*******************conv2*********************/
		//切换卷积层
		index_params_layer++;

		begin_count = clock();

		convolution(index_params_layer, 0);

		cout << endl << "------conv2------" << endl;

		end_count = clock();
		time_count = (double)(end_count - begin_count) / CLOCKS_PER_SEC;
		cout << "conv cost " << time_count << " s" << endl << endl;


		/*******************slice2和etlwise2****************/

		slice_etlwise();

		cout << "------slice2 etlwise2------" << endl;


		/****************pool2************/

		pooling();

		cout << endl << "------pool2------" << endl;


		/*********************conv3a*********************/
		//切换卷积层
		index_params_layer++;

		begin_count = clock();

		convolution(index_params_layer, 0);

		cout << endl << "------conv3a------" << endl;

		end_count = clock();
		time_count = (double)(end_count - begin_count) / CLOCKS_PER_SEC;
		cout << "conv cost " << time_count << " s" << endl << endl;


		/*******************slice3a和etlwise3a****************/

		slice_etlwise();

		cout << "------slice3a etlwise3a------" << endl;


		/*******************conv3*********************/

		//切换卷积层
		index_params_layer++;

		begin_count = clock();

		convolution(index_params_layer, 0);

		cout << endl << "------conv3------" << endl;

		end_count = clock();
		time_count = (double)(end_count - begin_count) / CLOCKS_PER_SEC;
		cout << "conv cost " << time_count << " s" << endl << endl;


		/*******************slice3和etlwise3****************/

		slice_etlwise();

		cout << "------slice3 etlwise3------" << endl;


		/***************pool3************/

		pooling();

		cout << endl << "------pool3------" << endl;



		/*********************conv4a*********************/
		//切换卷积层
		index_params_layer++;

		begin_count = clock();

		convolution(index_params_layer, 0);

		cout << endl << "------conv4a------" << endl;

		end_count = clock();
		time_count = (double)(end_count - begin_count) / CLOCKS_PER_SEC;
		cout << "conv cost " << time_count << " s" << endl << endl;


		/*******************slice4a和etlwise4a****************/

		slice_etlwise();

		cout << "------slice4a etlwise4a------" << endl;


		/*******************conv4*********************/
		//切换卷积层
		index_params_layer++;

		begin_count = clock();

		convolution(index_params_layer, 0);

		cout << endl << "------conv4------" << endl;

		end_count = clock();
		time_count = (double)(end_count - begin_count) / CLOCKS_PER_SEC;
		cout << "conv cost " << time_count << " s" << endl << endl;

		/*******************slice4和etlwise4****************/

		slice_etlwise();

		cout << "------slice4 etlwise4------" << endl;


		/*********************conv5a*********************/
		//切换卷积层
		index_params_layer++;

		begin_count = clock();

		convolution(index_params_layer, 0);

		cout << endl << "------conv5a------" << endl;

		end_count = clock();
		time_count = (double)(end_count - begin_count) / CLOCKS_PER_SEC;
		cout << "conv cost " << time_count << " s" << endl << endl;

		/*******************slice5a和etlwise5a****************/

		slice_etlwise();

		cout << "------slice5a etlwise5a------" << endl;


		/*******************conv5*********************/
		//切换卷积层
		index_params_layer++;

		begin_count = clock();

		convolution(index_params_layer, 0);

		cout << endl << "------conv5------" << endl;

		end_count = clock();
		time_count = (double)(end_count - begin_count) / CLOCKS_PER_SEC;
		cout << "conv cost " << time_count << " s" << endl << endl;

		/*******************slice5和etlwise5****************/

		slice_etlwise();

		cout << "------slice5 etlwise5------" << endl;


		/***************pool4************/

		pooling();

		//输出池化层结果
		cout << endl << "------pool4------" << endl;


		/****************fc1***************/
		//切换至全连接层
		index_params_layer++;

		//全连接层计时
		clock_t fc_begin, fc_end;
		fc_begin = clock();

		full_connection(index_params_layer, full_connection_result);

		fc_end = clock();

		//输出全连接层计算结果
		cout << endl << "------fc1------" << endl;

		double fc_time = (double)(fc_end - fc_begin) / CLOCKS_PER_SEC;
		cout << "fc layer cost " << fc_time << " s" << endl;

		//向txt中写512维特征
		ofstream out("data_jjw/" + output_feature_name + ".txt");
		for (int j = 0; j < 512; j++)
		{
			out << setiosflags(ios::fixed) << setprecision(7) << full_connection_result[j] << endl;
		}
		out.close();

	}

	//命令队列同步
	ret = clFinish(command_queue);

	//网络运行结束
	free(full_connection_result);
	cout << endl << "nets have executed " << index_current_layer + 1 << " layers" << endl;
	cout << endl << "lighten-CNN finish!" << endl;

	end = clock();
	double time = (double)(end - start) / CLOCKS_PER_SEC;
	cout << "lighten-CNN运行时间：" << time / iters << "s" << endl;

}

Deepface::~Deepface()
{
	//释放参数
	free(N);
	free(C);
	free(K);
	//释放设备端参数缓冲区
	for (int i = 0; i < num_param_layer; i++)
	{
		ret = clReleaseMemObject(layers_params_w_mem_obj[i]);
		ret = clReleaseMemObject(layers_params_b_mem_obj[i]);
	}
	free(layers_params_w_mem_obj);
	free(layers_params_b_mem_obj);

	//释放所有层的输入输出以及参数
	for (int i = 0; i < num_all_layers; i++)
	{
		ret = clReleaseMemObject(layers_data[i]);
		ret = clReleaseMemObject(layer_param_mem_obj[i]);
		free(layer_param[i]);
	}
	ret = clReleaseMemObject(layers_data[num_all_layers]);
	free(layers_data);
	free(layer_param);
	free(layer_param_mem_obj);

	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
	free(platform_id);

	total_end = clock();
	double time = (double)(total_end - total_start) / CLOCKS_PER_SEC;
	cout << "lighten-CNN total time: " << time << " s" << endl;
}

Deepface::Deepface()
{
	total_start = clock();

	//获取平台和设备
	ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
	platform_id = (cl_platform_id*)malloc(ret_num_platforms*sizeof(cl_platform_id));

	ret = clGetPlatformIDs(ret_num_platforms, platform_id, &ret_num_platforms);

	char dname[512];
	clGetPlatformInfo(platform_id[1], CL_PLATFORM_NAME, 512, dname, NULL);
	cout << "CL_PLATFORM_NAME:" << dname << endl;

	char vendor[512];//供应商
	char version[512];//OpenCL版本
	clGetPlatformInfo(platform_id[1], CL_PLATFORM_VENDOR, 512, vendor, NULL);
	clGetPlatformInfo(platform_id[1], CL_PLATFORM_VERSION, 512, version, NULL);

	cout << "CL_PLATFORM_VENDOR:" << vendor << endl;
	cout << "CL_PLATFORM_VERSION:" << version << endl;


	ret = clGetDeviceIDs(platform_id[1], CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

	ret = clGetDeviceInfo(device_id, CL_DEVICE_NAME, 512, dname, NULL);
	if (ret != CL_SUCCESS)
	{
		cout << "获取设备失败!" << endl;
		return;
	}
	cout << endl << "Device :" << dname << endl;


	/*
	cl_uint maxComputeUnits;
	ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxComputeUnits, NULL);
	cout << "Device :maxComputeUnits=" << maxComputeUnits << endl;
	cl_uint maxWorkGroupSize;
	ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(cl_uint), &maxWorkGroupSize, NULL);
	cout << "Device :maxWorkGroupSize=" << maxWorkGroupSize << endl << endl;
	*/

	//创建OpenCL上下文
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	//创建命令队列
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

	//载入内核源码到source_str
	FILE *fp;
	char *source_str;
	size_t source_size;

	//fp = fopen("layer_kernel.cl", "r");
	errno_t error = fopen_s(&fp, "layer_kernel.cl", "r");				//消除警告
	if (!fp) {
		fprintf(stderr, "Failed to load kernel\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	//创建程序
	program = clCreateProgramWithSource(context, 1,
		(const char**)&source_str, (const size_t*)&source_size, &ret);
	//构建程序
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	//创建OpenCL内核
	if (ret != CL_SUCCESS)
	{
		size_t len;
		char buffer[20 * 1024];

		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
	}
	free(source_str);


	cout << "lighten-CNN Net layer params initialization begin ..." << endl;


	/******将各层中参数读取到设备内存中******/

	num_param_layer = 10;									//lighten-CNN 中带参数的层数
	N = (int *)malloc(num_param_layer*sizeof(int));			//N:卷积核数量
	C = (int *)malloc(num_param_layer*sizeof(int));			//C:卷积核通道数
	K = (int *)malloc(num_param_layer*sizeof(int));			//K:卷积核尺寸
	ptr_layers_params_w = (float **)malloc(num_param_layer*sizeof(float *));
	ptr_layers_params_b = (float **)malloc(num_param_layer*sizeof(float *));
	layers_params_w_mem_obj = (cl_mem *)malloc(num_param_layer*sizeof(cl_mem));
	layers_params_b_mem_obj = (cl_mem *)malloc(num_param_layer*sizeof(cl_mem));

	ifstream read_layers_params("layers_params.txt");
	if (!read_layers_params.is_open())
	{
		cout << "layer_params read failure" << endl;
		return;
	}
	string current_layer_name;								//层名称
	string layer_param_type;										//参数类型：w或b								
	//循环读取 9 层卷积层参数
	for (int i = 0; i < num_param_layer-1; i++)
	{
		cout << "        conv " << i << " initialization" << endl;

		//读取层参数相关信息
		read_layers_params >> current_layer_name >> layer_param_type >> N[i] >> C[i] >> K[i] >> K[i];
		//分配当前卷积层 w 参数内存
		ptr_layers_params_w[i] = (float*)malloc(sizeof(float)*N[i] * K[i] * K[i] * C[i]);
		//分配设备缓冲区
		layers_params_w_mem_obj[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
			sizeof(float)*N[i] * K[i] * K[i] * C[i], NULL, &ret);
		//读取当前层 w
		for (int j = 0; j < N[i] * K[i] * K[i] * C[i]; j++)
		{
			read_layers_params >> ptr_layers_params_w[i][j];
		}
		//将主存数据传输到设备缓冲区中
		ret = clEnqueueWriteBuffer(command_queue, layers_params_w_mem_obj[i], CL_TRUE, 0,
			N[i] * K[i] * K[i] * C[i] * sizeof(float), ptr_layers_params_w[i], 0, NULL, NULL);


		//读取参数规格
		read_layers_params >> layer_param_type >> N[i];
		//当前层偏置分配内存
		ptr_layers_params_b[i] = (float*)malloc(sizeof(float)*N[i]);
		//分配缓冲区
		layers_params_b_mem_obj[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
			sizeof(float)*N[i], NULL, &ret);
		//读取当前层 b
		for (int j = 0; j < N[i]; j++)
		{
			read_layers_params >> ptr_layers_params_b[i][j];
		}
		//主存数据传输至设备缓冲区
		ret = clEnqueueWriteBuffer(command_queue, layers_params_b_mem_obj[i], CL_TRUE, 0,
			N[i] * sizeof(float), ptr_layers_params_b[i], 0, NULL, NULL);

		free(ptr_layers_params_w[i]);
		free(ptr_layers_params_b[i]);
		
	}

	//读取全连接层参数
	cout << "        fullconnection " << "initialization" << endl;

	//全连接层用 K = W*H*C
	//读取层参数相关信息
	read_layers_params >> current_layer_name >> layer_param_type >> N[num_param_layer - 1] >> K[num_param_layer - 1];
	//分配当前卷积层 w 参数内存
	ptr_layers_params_w[num_param_layer - 1] = (float*)malloc(sizeof(float)*N[num_param_layer - 1] * K[num_param_layer - 1]);
	//分配设备缓冲区
	layers_params_w_mem_obj[num_param_layer - 1] = clCreateBuffer(context, CL_MEM_READ_ONLY,
		sizeof(float)*N[num_param_layer - 1] * K[num_param_layer - 1], NULL, &ret);
	//读取当前层 w
	for (int j = 0; j < N[num_param_layer - 1] * K[num_param_layer - 1]; j++)
	{
		read_layers_params >> ptr_layers_params_w[num_param_layer - 1][j];
	}
	//将主存数据传输到设备缓冲区中
	ret = clEnqueueWriteBuffer(command_queue, layers_params_w_mem_obj[num_param_layer - 1], CL_TRUE, 0,
		N[num_param_layer - 1] * K[num_param_layer - 1] * sizeof(float), ptr_layers_params_w[num_param_layer - 1], 0, NULL, NULL);

	//读取参数规格
	read_layers_params >> layer_param_type >> N[num_param_layer - 1];
	//当前层偏置分配内存
	ptr_layers_params_b[num_param_layer - 1] = (float*)malloc(sizeof(float)*N[num_param_layer - 1]);
	//分配缓冲区
	layers_params_b_mem_obj[num_param_layer - 1] = clCreateBuffer(context, CL_MEM_READ_ONLY,
		sizeof(float)*N[num_param_layer - 1], NULL, &ret);
	//读取当前层 b
	for (int j = 0; j < N[num_param_layer - 1]; j++)
	{
		read_layers_params >> ptr_layers_params_b[num_param_layer - 1][j];
	}

	read_layers_params.close();

	//主存数据传输至设备缓冲区
	ret = clEnqueueWriteBuffer(command_queue, layers_params_b_mem_obj[num_param_layer - 1], CL_TRUE, 0,
		N[num_param_layer - 1] * sizeof(float), ptr_layers_params_b[num_param_layer - 1], 0, NULL, NULL);

	free(ptr_layers_params_w[num_param_layer - 1]);
	free(ptr_layers_params_b[num_param_layer - 1]);
	free(ptr_layers_params_w);
	free(ptr_layers_params_b);


	/**********分配所有层的输入和输出缓冲区***********/
	num_all_layers = 23;												//lightenCNN 总共有23层
	layers_data = (cl_mem *)malloc((num_all_layers + 1)*sizeof(cl_mem));//数据缓冲区有24个
	layers_data[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, 128 * 128 * sizeof(uchar), NULL, &ret);
	layers_data[1] = clCreateBuffer(context, CL_MEM_READ_ONLY, 96 * 128 * 128 * sizeof(float), NULL, &ret);
	layers_data[2] = clCreateBuffer(context, CL_MEM_READ_ONLY, 48 * 128 * 128 * sizeof(float), NULL, &ret);
	layers_data[3] = clCreateBuffer(context, CL_MEM_READ_ONLY, 48 * 64 * 64 * sizeof(float), NULL, &ret);
	layers_data[4] = clCreateBuffer(context, CL_MEM_READ_ONLY, 96 * 64 * 64 * sizeof(float), NULL, &ret);
	layers_data[5] = clCreateBuffer(context, CL_MEM_READ_ONLY, 48 * 64 * 64 * sizeof(float), NULL, &ret);
	layers_data[6] = clCreateBuffer(context, CL_MEM_READ_ONLY, 192 * 64 * 64 * sizeof(float), NULL, &ret);
	layers_data[7] = clCreateBuffer(context, CL_MEM_READ_ONLY, 96 * 64 * 64 * sizeof(float), NULL, &ret);
	layers_data[8] = clCreateBuffer(context, CL_MEM_READ_ONLY, 96 * 32 * 32 * sizeof(float), NULL, &ret);
	layers_data[9] = clCreateBuffer(context, CL_MEM_READ_ONLY, 192 * 32 * 32 * sizeof(float), NULL, &ret);
	layers_data[10] = clCreateBuffer(context, CL_MEM_READ_ONLY, 96 * 32 * 32 * sizeof(float), NULL, &ret);
	layers_data[11] = clCreateBuffer(context, CL_MEM_READ_ONLY, 384 * 32 * 32 * sizeof(float), NULL, &ret);
	layers_data[12] = clCreateBuffer(context, CL_MEM_READ_ONLY, 192 * 32 * 32 * sizeof(float), NULL, &ret);
	layers_data[13] = clCreateBuffer(context, CL_MEM_READ_ONLY, 192 * 16 * 16 * sizeof(float), NULL, &ret);
	layers_data[14] = clCreateBuffer(context, CL_MEM_READ_ONLY, 384 * 16 * 16 * sizeof(float), NULL, &ret);
	layers_data[15] = clCreateBuffer(context, CL_MEM_READ_ONLY, 192 * 16 * 16 * sizeof(float), NULL, &ret);
	layers_data[16] = clCreateBuffer(context, CL_MEM_READ_ONLY, 256 * 16 * 16 * sizeof(float), NULL, &ret);
	layers_data[17] = clCreateBuffer(context, CL_MEM_READ_ONLY, 128 * 16 * 16 * sizeof(float), NULL, &ret);
	layers_data[18] = clCreateBuffer(context, CL_MEM_READ_ONLY, 256 * 16 * 16 * sizeof(float), NULL, &ret);
	layers_data[19] = clCreateBuffer(context, CL_MEM_READ_ONLY, 128 * 16 * 16 * sizeof(float), NULL, &ret);
	layers_data[20] = clCreateBuffer(context, CL_MEM_READ_ONLY, 256 * 16 * 16 * sizeof(float), NULL, &ret);
	layers_data[21] = clCreateBuffer(context, CL_MEM_READ_ONLY, 128 * 16 * 16 * sizeof(float), NULL, &ret);
	layers_data[22] = clCreateBuffer(context, CL_MEM_READ_ONLY, 128 * 8 * 8 * sizeof(float), NULL, &ret);
	layers_data[23] = clCreateBuffer(context, CL_MEM_READ_ONLY, 512 * sizeof(float), NULL, &ret);


	/**********************************************/
	/**************设置每一层的规格参数************/
	layer_param = (int **)malloc(sizeof(int *)* 23);
	layer_param_mem_obj = (cl_mem *)malloc(sizeof(cl_mem)* 23);

	// W H C K N Stride Padding
	//conv1
	layer_param[0] = (int *)malloc(sizeof(int)* 7);
	layer_param_mem_obj[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)* 7, NULL, &ret);
	layer_param[0][0] = 128;
	layer_param[0][1] = 128;
	layer_param[0][2] = 1;
	layer_param[0][3] = 5;
	layer_param[0][4] = 96;
	layer_param[0][5] = 1;
	layer_param[0][6] = 2;
	ret = clEnqueueWriteBuffer(command_queue, layer_param_mem_obj[0], CL_TRUE, 0, sizeof(int)* 7, layer_param[0], 0, NULL, NULL);
	
	//slice_etlwise1
	layer_param[1] = (int *)malloc(sizeof(int)* 3);
	layer_param_mem_obj[1] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)* 3, NULL, &ret);
	layer_param[1][0] = 128;
	layer_param[1][1] = 128;
	layer_param[1][2] = 96;
	ret = clEnqueueWriteBuffer(command_queue, layer_param_mem_obj[1], CL_TRUE, 0, sizeof(int)* 3, layer_param[1], 0, NULL, NULL);
	
	//pool1
	layer_param[2] = (int *)malloc(sizeof(int)* 4);
	layer_param_mem_obj[2] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)* 4, NULL, &ret);
	layer_param[2][0] = 128;
	layer_param[2][1] = 128;
	layer_param[2][2] = 48;
	layer_param[2][3] = 2;
	ret = clEnqueueWriteBuffer(command_queue, layer_param_mem_obj[2], CL_TRUE, 0, sizeof(int)* 4, layer_param[2], 0, NULL, NULL);
	
	//conv2a
	layer_param[3] = (int *)malloc(sizeof(int)* 7);
	layer_param_mem_obj[3] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)* 7, NULL, &ret);
	layer_param[3][0] = 64;
	layer_param[3][1] = 64;
	layer_param[3][2] = 48;
	layer_param[3][3] = 1;
	layer_param[3][4] = 96;
	layer_param[3][5] = 1;
	layer_param[3][6] = 0;
	ret = clEnqueueWriteBuffer(command_queue, layer_param_mem_obj[3], CL_TRUE, 0, sizeof(int)* 7, layer_param[3], 0, NULL, NULL);
	
	//slice_etlwise2a
	layer_param[4] = (int *)malloc(sizeof(int)* 3);
	layer_param_mem_obj[4] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)* 3, NULL, &ret);
	layer_param[4][0] = 64;
	layer_param[4][1] = 64;
	layer_param[4][2] = 96;
	ret = clEnqueueWriteBuffer(command_queue, layer_param_mem_obj[4], CL_TRUE, 0, sizeof(int)* 3, layer_param[4], 0, NULL, NULL);
	
	//conv2
	layer_param[5] = (int *)malloc(sizeof(int)* 7);
	layer_param_mem_obj[5] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)* 7, NULL, &ret);
	layer_param[5][0] = 64;
	layer_param[5][1] = 64;
	layer_param[5][2] = 48;
	layer_param[5][3] = 3;
	layer_param[5][4] = 192;
	layer_param[5][5] = 1;
	layer_param[5][6] = 1;
	ret = clEnqueueWriteBuffer(command_queue, layer_param_mem_obj[5], CL_TRUE, 0, sizeof(int)* 7, layer_param[5], 0, NULL, NULL);
	
	//slice_etlwise2
	layer_param[6] = (int *)malloc(sizeof(int)* 3);
	layer_param_mem_obj[6] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)* 3, NULL, &ret);
	layer_param[6][0] = 64;
	layer_param[6][1] = 64;
	layer_param[6][2] = 192;
	ret = clEnqueueWriteBuffer(command_queue, layer_param_mem_obj[6], CL_TRUE, 0, sizeof(int)* 3, layer_param[6], 0, NULL, NULL);
	
	//pool2
	layer_param[7] = (int *)malloc(sizeof(int)* 4);
	layer_param_mem_obj[7] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)* 4, NULL, &ret);
	layer_param[7][0] = 64;
	layer_param[7][1] = 64;
	layer_param[7][2] = 96;
	layer_param[7][3] = 2;
	ret = clEnqueueWriteBuffer(command_queue, layer_param_mem_obj[7], CL_TRUE, 0, sizeof(int)* 4, layer_param[7], 0, NULL, NULL);
	
	//conv3a
	layer_param[8] = (int *)malloc(sizeof(int)* 7);
	layer_param_mem_obj[8] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)* 7, NULL, &ret);
	layer_param[8][0] = 32;
	layer_param[8][1] = 32;
	layer_param[8][2] = 96;
	layer_param[8][3] = 1;
	layer_param[8][4] = 192;
	layer_param[8][5] = 1;
	layer_param[8][6] = 0;
	ret = clEnqueueWriteBuffer(command_queue, layer_param_mem_obj[8], CL_TRUE, 0, sizeof(int)* 7, layer_param[8], 0, NULL, NULL);
	
	//slice_etlwise3a
	layer_param[9] = (int *)malloc(sizeof(int)* 3);
	layer_param_mem_obj[9] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)* 3, NULL, &ret);
	layer_param[9][0] = 32;
	layer_param[9][1] = 32;
	layer_param[9][2] = 192;
	ret = clEnqueueWriteBuffer(command_queue, layer_param_mem_obj[9], CL_TRUE, 0, sizeof(int)* 3, layer_param[9], 0, NULL, NULL);
	
	//conv3
	layer_param[10] = (int *)malloc(sizeof(int)* 7);
	layer_param_mem_obj[10] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)* 7, NULL, &ret);
	layer_param[10][0] = 32;
	layer_param[10][1] = 32;
	layer_param[10][2] = 96;
	layer_param[10][3] = 3;
	layer_param[10][4] = 384;
	layer_param[10][5] = 1;
	layer_param[10][6] = 1;
	ret = clEnqueueWriteBuffer(command_queue, layer_param_mem_obj[10], CL_TRUE, 0, sizeof(int)* 7, layer_param[10], 0, NULL, NULL);
	
	//slice_etlwise3
	layer_param[11] = (int *)malloc(sizeof(int)* 3);
	layer_param_mem_obj[11] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)* 3, NULL, &ret);
	layer_param[11][0] = 32;
	layer_param[11][1] = 32;
	layer_param[11][2] = 384;
	ret = clEnqueueWriteBuffer(command_queue, layer_param_mem_obj[11], CL_TRUE, 0, sizeof(int)* 3, layer_param[11], 0, NULL, NULL);
	
	//pool3
	layer_param[12] = (int *)malloc(sizeof(int)* 4);
	layer_param_mem_obj[12] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)* 4, NULL, &ret);
	layer_param[12][0] = 32;
	layer_param[12][1] = 32;
	layer_param[12][2] = 192;
	layer_param[12][3] = 2;
	ret = clEnqueueWriteBuffer(command_queue, layer_param_mem_obj[12], CL_TRUE, 0, sizeof(int)* 4, layer_param[12], 0, NULL, NULL);
	
	//conv4a
	layer_param[13] = (int *)malloc(sizeof(int)* 7);
	layer_param_mem_obj[13] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)* 7, NULL, &ret);
	layer_param[13][0] = 16;
	layer_param[13][1] = 16;
	layer_param[13][2] = 192;
	layer_param[13][3] = 1;
	layer_param[13][4] = 384;
	layer_param[13][5] = 1;
	layer_param[13][6] = 0;
	ret = clEnqueueWriteBuffer(command_queue, layer_param_mem_obj[13], CL_TRUE, 0, sizeof(int)* 7, layer_param[13], 0, NULL, NULL);
	
	//slice_etlwise4a
	layer_param[14] = (int *)malloc(sizeof(int)* 3);
	layer_param_mem_obj[14] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)* 3, NULL, &ret);
	layer_param[14][0] = 16;
	layer_param[14][1] = 16;
	layer_param[14][2] = 384;
	ret = clEnqueueWriteBuffer(command_queue, layer_param_mem_obj[14], CL_TRUE, 0, sizeof(int)* 3, layer_param[14], 0, NULL, NULL);
	
	//conv4
	layer_param[15] = (int *)malloc(sizeof(int)* 7);
	layer_param_mem_obj[15] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)* 7, NULL, &ret);
	layer_param[15][0] = 16;
	layer_param[15][1] = 16;
	layer_param[15][2] = 192;
	layer_param[15][3] = 3;
	layer_param[15][4] = 256;
	layer_param[15][5] = 1;
	layer_param[15][6] = 1;
	ret = clEnqueueWriteBuffer(command_queue, layer_param_mem_obj[15], CL_TRUE, 0, sizeof(int)* 7, layer_param[15], 0, NULL, NULL);
	
	//slice_etlwise4
	layer_param[16] = (int *)malloc(sizeof(int)* 3);
	layer_param_mem_obj[16] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)* 3, NULL, &ret);
	layer_param[16][0] = 16;
	layer_param[16][1] = 16;
	layer_param[16][2] = 256;
	ret = clEnqueueWriteBuffer(command_queue, layer_param_mem_obj[16], CL_TRUE, 0, sizeof(int)* 3, layer_param[16], 0, NULL, NULL);
	
	//conv5a
	layer_param[17] = (int *)malloc(sizeof(int)* 7);
	layer_param_mem_obj[17] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)* 7, NULL, &ret);
	layer_param[17][0] = 16;
	layer_param[17][1] = 16;
	layer_param[17][2] = 128;
	layer_param[17][3] = 1;
	layer_param[17][4] = 256;
	layer_param[17][5] = 1;
	layer_param[17][6] = 0;
	ret = clEnqueueWriteBuffer(command_queue, layer_param_mem_obj[17], CL_TRUE, 0, sizeof(int)* 7, layer_param[17], 0, NULL, NULL);
	
	//slice_etlwise5a
	layer_param[18] = (int *)malloc(sizeof(int)* 3);
	layer_param_mem_obj[18] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)* 3, NULL, &ret);
	layer_param[18][0] = 16;
	layer_param[18][1] = 16;
	layer_param[18][2] = 256;
	ret = clEnqueueWriteBuffer(command_queue, layer_param_mem_obj[18], CL_TRUE, 0, sizeof(int)* 3, layer_param[18], 0, NULL, NULL);
	
	//conv5
	layer_param[19] = (int *)malloc(sizeof(int)* 7);
	layer_param_mem_obj[19] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)* 7, NULL, &ret);
	layer_param[19][0] = 16;
	layer_param[19][1] = 16;
	layer_param[19][2] = 128;
	layer_param[19][3] = 3;
	layer_param[19][4] = 256;
	layer_param[19][5] = 1;
	layer_param[19][6] = 1;
	ret = clEnqueueWriteBuffer(command_queue, layer_param_mem_obj[19], CL_TRUE, 0, sizeof(int)* 7, layer_param[19], 0, NULL, NULL);
	
	//slice_etlwise5
	layer_param[20] = (int *)malloc(sizeof(int)* 3);
	layer_param_mem_obj[20] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)* 3, NULL, &ret);
	layer_param[20][0] = 16;
	layer_param[20][1] = 16;
	layer_param[20][2] = 256;
	ret = clEnqueueWriteBuffer(command_queue, layer_param_mem_obj[20], CL_TRUE, 0, sizeof(int)* 3, layer_param[20], 0, NULL, NULL);
	
	//pool4
	layer_param[21] = (int *)malloc(sizeof(int)* 4);
	layer_param_mem_obj[21] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)* 4, NULL, &ret);
	layer_param[21][0] = 16;
	layer_param[21][1] = 16;
	layer_param[21][2] = 128;
	layer_param[21][3] = 2;
	ret = clEnqueueWriteBuffer(command_queue, layer_param_mem_obj[21], CL_TRUE, 0, sizeof(int)* 4, layer_param[21], 0, NULL, NULL);
	
	//fc1
	layer_param[22] = (int *)malloc(sizeof(int)* 5);
	layer_param_mem_obj[22] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)* 5, NULL, &ret);
	layer_param[22][0] = 8;
	layer_param[22][1] = 8;
	layer_param[22][2] = 128;
	layer_param[22][4] = 512;
	ret = clEnqueueWriteBuffer(command_queue, layer_param_mem_obj[22], CL_TRUE, 0, sizeof(int)* 5, layer_param[22], 0, NULL, NULL);
	
	/***********************************************************/


	//网络初始化完毕
	cout << "lighten-CNN Net layer_params initialization end" << endl;
}


//第一层的卷积层
//W, H, C, K, N;
void Deepface::convolution(int index, uchar *input, bool debug)
{
	index_current_layer = 0;			//初始化当前层

	//创建内存缓冲对象，在设备上为每个向量
	cl_mem pi_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
		layer_param[index_current_layer][2] * (layer_param[index_current_layer][0] + layer_param[index_current_layer][6] * 2) * (layer_param[index_current_layer][1] + layer_param[index_current_layer][6] * 2) * sizeof(float), NULL, &ret);
	cl_mem ti_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
		((layer_param[index_current_layer][0] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1)*((layer_param[index_current_layer][1] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1)*(layer_param[index_current_layer][2] * layer_param[index_current_layer][3] * layer_param[index_current_layer][3])*sizeof(float), NULL, &ret);

	//拷贝卷积核和图片输入到对应的内存缓冲
	ret = clEnqueueWriteBuffer(command_queue, layers_data[index_current_layer], CL_TRUE, 0,
		layer_param[index_current_layer][0] * layer_param[index_current_layer][1] * sizeof(uchar), input, 0, NULL, NULL);

	/*********************卷积输入Padding*****************/
	//创建OpenCL内核
	cl_kernel kernel_p = clCreateKernel(program, "padding_char", &ret);

	//设置内核参数
	ret = clSetKernelArg(kernel_p, 0, sizeof(cl_mem), (void*)&layers_data[index_current_layer]);
	ret = clSetKernelArg(kernel_p, 1, sizeof(cl_mem), (void*)&pi_mem_obj);
	ret = clSetKernelArg(kernel_p, 2, sizeof(cl_mem), (void*)&layer_param_mem_obj[index_current_layer]);

	//执行内核
	size_t global_work_size3_p[3] = { layer_param[index_current_layer][0], layer_param[index_current_layer][1], layer_param[index_current_layer][2] };			//工作节点分配
	ret = clEnqueueNDRangeKernel(command_queue, kernel_p, 3, NULL,
		global_work_size3_p, NULL, 0, NULL, NULL);

	//读取内存缓冲C到本地变量C
	/*
	float * padding_image = (float*)malloc(sizeof(float)*(layer_param[index_current_layer][0] + 2 * layer_param[index_current_layer][6])*(layer_param[index_current_layer][1] + 2 * layer_param[index_current_layer][6])*layer_param[index_current_layer][2]);
	ret = clEnqueueReadBuffer(command_queue, pi_mem_obj, CL_TRUE, 0,
		(layer_param[index_current_layer][0] + 2 * layer_param[index_current_layer][6])*(layer_param[index_current_layer][1] + 2 * layer_param[index_current_layer][6])*layer_param[index_current_layer][2] * sizeof(float), padding_image, 0, NULL, NULL);

	if (debug)
	{
		ofstream conv2_out("OpenCL_padding_conv2_out.txt");
		for (int i = 0; i < 48*66*66; i++)
		{
			conv2_out << " " << padding_image[i];
		}
		conv2_out.close();
	}
	
	//free(padding_image);
	*/

	/*******************滑动窗口转换*********************/
	//创建OpenCL内核
	cl_kernel kernel_i = clCreateKernel(program, "transform_input", &ret);

	//设置内核参数
	ret = clSetKernelArg(kernel_i, 0, sizeof(cl_mem), (void*)&pi_mem_obj);
	ret = clSetKernelArg(kernel_i, 1, sizeof(cl_mem), (void*)&ti_mem_obj);
	ret = clSetKernelArg(kernel_i, 2, sizeof(cl_mem), (void*)&layer_param_mem_obj[index_current_layer]);

	//执行内核
	size_t global_work_size3[3] = { (layer_param[index_current_layer][0] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1, (layer_param[index_current_layer][1] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1, layer_param[index_current_layer][2] };			//工作节点分配
	ret = clEnqueueNDRangeKernel(command_queue, kernel_i, 3, NULL,
		global_work_size3, NULL, 0, NULL, NULL);

	//读取内存缓冲C到本地变量C
	/*
	float *transform_input = (float*)malloc(sizeof(float)*
		((layer_param[index_current_layer][0] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1)*((layer_param[index_current_layer][1] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1)*(layer_param[index_current_layer][2] * layer_param[index_current_layer][3] * layer_param[index_current_layer][3]));
	ret = clEnqueueReadBuffer(command_queue, ti_mem_obj, CL_TRUE, 0,
		((layer_param[index_current_layer][0] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1)*((layer_param[index_current_layer][1] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1)*(layer_param[index_current_layer][2] * layer_param[index_current_layer][3] * layer_param[index_current_layer][3])*sizeof(float), transform_input, 0, NULL, NULL);
	
	if (debug)
	{
		ofstream conv2_out("OpenCL_transform_conv2_out.txt");
		for (int i = 0; i < 64 * 64 * 48*9; i++)
		{
			conv2_out << " " << transform_input[i];
		}
		conv2_out.close();
	}

	//free(transform_input);
	*/

	/****************矩阵乘法****************/

	clock_t begin, end;
	begin = clock();

	//cl_kernel kernel_m = clCreateKernel(program, "matrix_multiply", &ret);
	cl_kernel kernel_m = clCreateKernel(program, "matrix_multiply_singel", &ret);

	//设置内核参数
	ret = clSetKernelArg(kernel_m, 0, sizeof(cl_mem), (void*)&ti_mem_obj);
	ret = clSetKernelArg(kernel_m, 1, sizeof(cl_mem), (void*)&layers_params_w_mem_obj[index]);
	ret = clSetKernelArg(kernel_m, 2, sizeof(cl_mem), (void*)&layers_params_b_mem_obj[index]);
	ret = clSetKernelArg(kernel_m, 3, sizeof(cl_mem), (void*)&layers_data[index_current_layer + 1]);
	ret = clSetKernelArg(kernel_m, 4, sizeof(cl_mem), (void*)&layer_param_mem_obj[index_current_layer]);

	//执行内核
	size_t global_item_size[2] = { layer_param[index_current_layer][4], ((layer_param[index_current_layer][1] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1)*((layer_param[index_current_layer][0] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1) };			//工作节点分配
	ret = clEnqueueNDRangeKernel(command_queue, kernel_m, 2, NULL,
		global_item_size, NULL, 0, NULL, NULL);

	//清理资源
	ret = clReleaseKernel(kernel_i);
	ret = clReleaseKernel(kernel_m);
	ret = clReleaseKernel(kernel_p);
	ret = clReleaseMemObject(pi_mem_obj);
	ret = clReleaseMemObject(ti_mem_obj);

	end = clock();
	double time = (double)(end - begin) / CLOCKS_PER_SEC;
	cout << endl << "matrix multiply cost " << time << endl;
	cout << "matrix size:" << layer_param[index_current_layer][4] << "*" << ((layer_param[index_current_layer][1] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1)*((layer_param[index_current_layer][0] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1) << endl;
	cout << "add size:" << layer_param[index_current_layer][2] * layer_param[index_current_layer][3] * layer_param[index_current_layer][3] << "=" << layer_param[index_current_layer][2] << "*" << layer_param[index_current_layer][3] << "*" << layer_param[index_current_layer][3] << endl;

}

//W, H, C, K, N;
void Deepface::convolution(int index, bool debug)
{
	//切换至当前层
	index_current_layer++;

	//创建内存缓冲对象，在设备上为每个向量
	cl_mem pi_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
		layer_param[index_current_layer][2] * (layer_param[index_current_layer][0] + layer_param[index_current_layer][6] * 2) * (layer_param[index_current_layer][1] + layer_param[index_current_layer][6] * 2) * sizeof(float), NULL, &ret);
	cl_mem ti_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
		((layer_param[index_current_layer][0] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1)*((layer_param[index_current_layer][1] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1)*(layer_param[index_current_layer][2] * layer_param[index_current_layer][3] * layer_param[index_current_layer][3])*sizeof(float), NULL, &ret);


	/*********************卷积输入Padding*****************/
	//创建OpenCL内核
	cl_kernel kernel_p = clCreateKernel(program, "padding", &ret);

	//设置内核参数
	ret = clSetKernelArg(kernel_p, 0, sizeof(cl_mem), (void*)&layers_data[index_current_layer]);
	ret = clSetKernelArg(kernel_p, 1, sizeof(cl_mem), (void*)&pi_mem_obj);
	ret = clSetKernelArg(kernel_p, 2, sizeof(cl_mem), (void*)&layer_param_mem_obj[index_current_layer]);

	//执行内核
	size_t global_work_size3_p[3] = { layer_param[index_current_layer][0], layer_param[index_current_layer][1], layer_param[index_current_layer][2] };			//工作节点分配
	ret = clEnqueueNDRangeKernel(command_queue, kernel_p, 3, NULL,
		global_work_size3_p, NULL, 0, NULL, NULL);


	/*******************滑动窗口转换*********************/
	//创建OpenCL内核
	cl_kernel kernel_i = clCreateKernel(program, "transform_input", &ret);

	//设置内核参数
	ret = clSetKernelArg(kernel_i, 0, sizeof(cl_mem), (void*)&pi_mem_obj);
	ret = clSetKernelArg(kernel_i, 1, sizeof(cl_mem), (void*)&ti_mem_obj);
	ret = clSetKernelArg(kernel_i, 2, sizeof(cl_mem), (void*)&layer_param_mem_obj[index_current_layer]);

	//执行内核
	size_t global_work_size3[3] = { (layer_param[index_current_layer][0] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1, (layer_param[index_current_layer][1] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1, layer_param[index_current_layer][2] };			//工作节点分配
	ret = clEnqueueNDRangeKernel(command_queue, kernel_i, 3, NULL,
		global_work_size3, NULL, 0, NULL, NULL);


	/****************矩阵乘法****************/
	cl_kernel kernel_m = clCreateKernel(program, "matrix_multiply", &ret);

	//设置内核参数
	ret = clSetKernelArg(kernel_m, 0, sizeof(cl_mem), (void*)&ti_mem_obj);
	ret = clSetKernelArg(kernel_m, 1, sizeof(cl_mem), (void*)&layers_params_w_mem_obj[index]);
	ret = clSetKernelArg(kernel_m, 2, sizeof(cl_mem), (void*)&layers_params_b_mem_obj[index]);
	ret = clSetKernelArg(kernel_m, 3, sizeof(cl_mem), (void*)&layers_data[index_current_layer + 1]);
	ret = clSetKernelArg(kernel_m, 4, sizeof(cl_mem), (void*)&layer_param_mem_obj[index_current_layer]);

	//执行内核
	size_t global_item_size[2] = { layer_param[index_current_layer][4], ((layer_param[index_current_layer][1] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1)*((layer_param[index_current_layer][0] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1) };			//工作节点分配
	size_t global_group_size[2] = { 8, 8 };
	ret = clEnqueueNDRangeKernel(command_queue, kernel_m, 2, NULL,
		global_item_size, global_group_size, 0, NULL, NULL);


	//清理资源
	ret = clReleaseKernel(kernel_i);
	ret = clReleaseKernel(kernel_m);
	ret = clReleaseKernel(kernel_p);
	ret = clReleaseMemObject(pi_mem_obj);
	ret = clReleaseMemObject(ti_mem_obj);
}

//W,H,C,K,N
void Deepface::slice_etlwise()
{
	//切换至当前层
	index_current_layer++;

	//创建OpenCL内核
	cl_kernel kernel_m = clCreateKernel(program, "slice_etlwise", &ret);

	//设置内核参数
	ret = clSetKernelArg(kernel_m, 0, sizeof(cl_mem), (void*)&layers_data[index_current_layer]);
	ret = clSetKernelArg(kernel_m, 1, sizeof(cl_mem), (void*)&layers_data[index_current_layer + 1]);
	ret = clSetKernelArg(kernel_m, 2, sizeof(cl_mem), (void*)&layer_param_mem_obj[index_current_layer]);

	//执行内核
	size_t global_global_size[2] = { layer_param[index_current_layer][2] / 2, layer_param[index_current_layer][0] * layer_param[index_current_layer][1] };			//工作节点分配
	ret = clEnqueueNDRangeKernel(command_queue, kernel_m, 2, NULL,
		global_global_size, NULL, 0, NULL, NULL);

	ret = clReleaseKernel(kernel_m);
}


//W, H, C, K
void Deepface::pooling()
{
	//切换至当前层
	index_current_layer++;

	//创建OpenCL内核
	cl_kernel kernel = clCreateKernel(program, "pool", &ret);

	//设置内核参数
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&layers_data[index_current_layer]);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&layers_data[index_current_layer + 1]);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&layer_param_mem_obj[index_current_layer]);

	//执行内核
	size_t global_work_size[3] = { (layer_param[index_current_layer][0] / layer_param[index_current_layer][3]), (layer_param[index_current_layer][1] / layer_param[index_current_layer][3]), layer_param[index_current_layer][2] };			//工作节点分配
	
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 3, NULL,
		global_work_size, NULL, 0, NULL, NULL);

	ret = clReleaseKernel(kernel);
}


//W, H, C, N;
void Deepface::full_connection(int index, float *full_connection_result)
{
	//切换至当前层
	index_current_layer++;

	//创建内存缓冲对象，在设备上为每个向量
	//计算全连接层的临时缓冲
	cl_mem t_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		layer_param[index_current_layer][4] * layer_param[index_current_layer][0] * layer_param[index_current_layer][1] * layer_param[index_current_layer][2] * sizeof(float), NULL, &ret);


	//创建OpenCL内核
	cl_kernel kernel_m = clCreateKernel(program, "full_connection", &ret);

	//设置内核参数
	ret = clSetKernelArg(kernel_m, 0, sizeof(cl_mem), (void*)&layers_data[index_current_layer]);
	ret = clSetKernelArg(kernel_m, 1, sizeof(cl_mem), (void*)&layers_params_w_mem_obj[index]);
	ret = clSetKernelArg(kernel_m, 2, sizeof(cl_mem), (void*)&t_mem_obj);
	ret = clSetKernelArg(kernel_m, 3, sizeof(cl_mem), (void*)&layers_params_b_mem_obj[index]);
	ret = clSetKernelArg(kernel_m, 4, sizeof(cl_mem), (void*)&layers_data[index_current_layer + 1]);
	ret = clSetKernelArg(kernel_m, 5, sizeof(cl_mem), (void*)&layer_param_mem_obj[index_current_layer]);

	//执行内核
	size_t global_global_size[2] = { layer_param[index_current_layer][4], layer_param[index_current_layer][0] * layer_param[index_current_layer][1] * layer_param[index_current_layer][2] };			//工作节点分配
	ret = clEnqueueNDRangeKernel(command_queue, kernel_m, 2, NULL,
		global_global_size, NULL, 0, NULL, NULL);

	//读取内存缓冲C到本地变量C
	ret = clEnqueueReadBuffer(command_queue, layers_data[index_current_layer + 1], CL_TRUE, 0,
		layer_param[index_current_layer][4] * sizeof(float), full_connection_result, 0, NULL, NULL);

	ret = clReleaseKernel(kernel_m);
	ret = clReleaseMemObject(t_mem_obj);
}
