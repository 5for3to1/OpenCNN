#include "Deepface.h"


//����ǰ�����
void Deepface::forward(string output_feature_name)
{
	int index_params_layer;							//�������Ĳ����

	//�������ͬ��
	ret = clFinish(command_queue);

	//��ʼ��ʱ
	clock_t start, end;								//ʱ�Ӽ�ʱ
	start = clock();

	//��ʱ
	clock_t begin_count, end_count;
	double time_count;

	float * full_connection_result = (float*)malloc(sizeof(float)* 512);			//���յ�512ά��������

	//ѭ�� iters ��
	int iters = 1;
	for (int i = 0; i < iters; i++)
	{

		/***************�����: conv1*************/
		//��ʼ���������Ĳ��
		index_params_layer = 0;

		//��ʱ��ʼ
		begin_count = clock();

		//����Mat���ͼƬ
		Mat src_img = imread("data_jjw/gray_128_128.jpg", 0);			//����һ�ŻҶ�ͼ
		if (src_img.empty())
		{
			cout << "image read failure" << endl;
			return;
		}

		convolution(index_params_layer, src_img.data, 0);					//��һ�ξ��

		cout << endl << "------conv1------" << endl;

		end_count = clock();
		time_count = (double)(end_count - begin_count) / CLOCKS_PER_SEC;
		cout << "conv cost " << time_count << " s" << endl << endl;


		/***************slice1��etlwise1��**************/

		slice_etlwise();

		//���slice��etlwise�������
		cout << "------slice11 etlwise11------" << endl;


		/***************�ػ���: pool1************/

		pooling();

		cout << endl << "------pool1------" << endl;


		/*********************conv2a*********************/
		//�л������
		index_params_layer++;

		begin_count = clock();

		convolution(index_params_layer, 0);

		cout << endl << "------conv2a------" << endl;

		end_count = clock();
		time_count = (double)(end_count - begin_count) / CLOCKS_PER_SEC;
		cout << "conv cost " << time_count << " s" << endl << endl;


		/*******************slice2a��etlwise2a****************/

		slice_etlwise();

		cout << "------slice2a etlwise2a------" << endl;


		/*******************conv2*********************/
		//�л������
		index_params_layer++;

		begin_count = clock();

		convolution(index_params_layer, 0);

		cout << endl << "------conv2------" << endl;

		end_count = clock();
		time_count = (double)(end_count - begin_count) / CLOCKS_PER_SEC;
		cout << "conv cost " << time_count << " s" << endl << endl;


		/*******************slice2��etlwise2****************/

		slice_etlwise();

		cout << "------slice2 etlwise2------" << endl;


		/****************pool2************/

		pooling();

		cout << endl << "------pool2------" << endl;


		/*********************conv3a*********************/
		//�л������
		index_params_layer++;

		begin_count = clock();

		convolution(index_params_layer, 0);

		cout << endl << "------conv3a------" << endl;

		end_count = clock();
		time_count = (double)(end_count - begin_count) / CLOCKS_PER_SEC;
		cout << "conv cost " << time_count << " s" << endl << endl;


		/*******************slice3a��etlwise3a****************/

		slice_etlwise();

		cout << "------slice3a etlwise3a------" << endl;


		/*******************conv3*********************/

		//�л������
		index_params_layer++;

		begin_count = clock();

		convolution(index_params_layer, 0);

		cout << endl << "------conv3------" << endl;

		end_count = clock();
		time_count = (double)(end_count - begin_count) / CLOCKS_PER_SEC;
		cout << "conv cost " << time_count << " s" << endl << endl;


		/*******************slice3��etlwise3****************/

		slice_etlwise();

		cout << "------slice3 etlwise3------" << endl;


		/***************pool3************/

		pooling();

		cout << endl << "------pool3------" << endl;



		/*********************conv4a*********************/
		//�л������
		index_params_layer++;

		begin_count = clock();

		convolution(index_params_layer, 0);

		cout << endl << "------conv4a------" << endl;

		end_count = clock();
		time_count = (double)(end_count - begin_count) / CLOCKS_PER_SEC;
		cout << "conv cost " << time_count << " s" << endl << endl;


		/*******************slice4a��etlwise4a****************/

		slice_etlwise();

		cout << "------slice4a etlwise4a------" << endl;


		/*******************conv4*********************/
		//�л������
		index_params_layer++;

		begin_count = clock();

		convolution(index_params_layer, 0);

		cout << endl << "------conv4------" << endl;

		end_count = clock();
		time_count = (double)(end_count - begin_count) / CLOCKS_PER_SEC;
		cout << "conv cost " << time_count << " s" << endl << endl;

		/*******************slice4��etlwise4****************/

		slice_etlwise();

		cout << "------slice4 etlwise4------" << endl;


		/*********************conv5a*********************/
		//�л������
		index_params_layer++;

		begin_count = clock();

		convolution(index_params_layer, 0);

		cout << endl << "------conv5a------" << endl;

		end_count = clock();
		time_count = (double)(end_count - begin_count) / CLOCKS_PER_SEC;
		cout << "conv cost " << time_count << " s" << endl << endl;

		/*******************slice5a��etlwise5a****************/

		slice_etlwise();

		cout << "------slice5a etlwise5a------" << endl;


		/*******************conv5*********************/
		//�л������
		index_params_layer++;

		begin_count = clock();

		convolution(index_params_layer, 0);

		cout << endl << "------conv5------" << endl;

		end_count = clock();
		time_count = (double)(end_count - begin_count) / CLOCKS_PER_SEC;
		cout << "conv cost " << time_count << " s" << endl << endl;

		/*******************slice5��etlwise5****************/

		slice_etlwise();

		cout << "------slice5 etlwise5------" << endl;


		/***************pool4************/

		pooling();

		//����ػ�����
		cout << endl << "------pool4------" << endl;


		/****************fc1***************/
		//�л���ȫ���Ӳ�
		index_params_layer++;

		//ȫ���Ӳ��ʱ
		clock_t fc_begin, fc_end;
		fc_begin = clock();

		full_connection(index_params_layer, full_connection_result);

		fc_end = clock();

		//���ȫ���Ӳ������
		cout << endl << "------fc1------" << endl;

		double fc_time = (double)(fc_end - fc_begin) / CLOCKS_PER_SEC;
		cout << "fc layer cost " << fc_time << " s" << endl;

		//��txt��д512ά����
		ofstream out("data_jjw/" + output_feature_name + ".txt");
		for (int j = 0; j < 512; j++)
		{
			out << setiosflags(ios::fixed) << setprecision(7) << full_connection_result[j] << endl;
		}
		out.close();

	}

	//�������ͬ��
	ret = clFinish(command_queue);

	//�������н���
	free(full_connection_result);
	cout << endl << "nets have executed " << index_current_layer + 1 << " layers" << endl;
	cout << endl << "lighten-CNN finish!" << endl;

	end = clock();
	double time = (double)(end - start) / CLOCKS_PER_SEC;
	cout << "lighten-CNN����ʱ�䣺" << time / iters << "s" << endl;

}

Deepface::~Deepface()
{
	//�ͷŲ���
	free(N);
	free(C);
	free(K);
	//�ͷ��豸�˲���������
	for (int i = 0; i < num_param_layer; i++)
	{
		ret = clReleaseMemObject(layers_params_w_mem_obj[i]);
		ret = clReleaseMemObject(layers_params_b_mem_obj[i]);
	}
	free(layers_params_w_mem_obj);
	free(layers_params_b_mem_obj);

	//�ͷ����в����������Լ�����
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

	//��ȡƽ̨���豸
	ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
	platform_id = (cl_platform_id*)malloc(ret_num_platforms*sizeof(cl_platform_id));

	ret = clGetPlatformIDs(ret_num_platforms, platform_id, &ret_num_platforms);

	char dname[512];
	clGetPlatformInfo(platform_id[1], CL_PLATFORM_NAME, 512, dname, NULL);
	cout << "CL_PLATFORM_NAME:" << dname << endl;

	char vendor[512];//��Ӧ��
	char version[512];//OpenCL�汾
	clGetPlatformInfo(platform_id[1], CL_PLATFORM_VENDOR, 512, vendor, NULL);
	clGetPlatformInfo(platform_id[1], CL_PLATFORM_VERSION, 512, version, NULL);

	cout << "CL_PLATFORM_VENDOR:" << vendor << endl;
	cout << "CL_PLATFORM_VERSION:" << version << endl;


	ret = clGetDeviceIDs(platform_id[1], CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

	ret = clGetDeviceInfo(device_id, CL_DEVICE_NAME, 512, dname, NULL);
	if (ret != CL_SUCCESS)
	{
		cout << "��ȡ�豸ʧ��!" << endl;
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

	//����OpenCL������
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	//�����������
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

	//�����ں�Դ�뵽source_str
	FILE *fp;
	char *source_str;
	size_t source_size;

	//fp = fopen("layer_kernel.cl", "r");
	errno_t error = fopen_s(&fp, "layer_kernel.cl", "r");				//��������
	if (!fp) {
		fprintf(stderr, "Failed to load kernel\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	//��������
	program = clCreateProgramWithSource(context, 1,
		(const char**)&source_str, (const size_t*)&source_size, &ret);
	//��������
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	//����OpenCL�ں�
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


	/******�������в�����ȡ���豸�ڴ���******/

	num_param_layer = 10;									//lighten-CNN �д������Ĳ���
	N = (int *)malloc(num_param_layer*sizeof(int));			//N:���������
	C = (int *)malloc(num_param_layer*sizeof(int));			//C:�����ͨ����
	K = (int *)malloc(num_param_layer*sizeof(int));			//K:����˳ߴ�
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
	string current_layer_name;								//������
	string layer_param_type;										//�������ͣ�w��b								
	//ѭ����ȡ 9 ���������
	for (int i = 0; i < num_param_layer-1; i++)
	{
		cout << "        conv " << i << " initialization" << endl;

		//��ȡ����������Ϣ
		read_layers_params >> current_layer_name >> layer_param_type >> N[i] >> C[i] >> K[i] >> K[i];
		//���䵱ǰ����� w �����ڴ�
		ptr_layers_params_w[i] = (float*)malloc(sizeof(float)*N[i] * K[i] * K[i] * C[i]);
		//�����豸������
		layers_params_w_mem_obj[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
			sizeof(float)*N[i] * K[i] * K[i] * C[i], NULL, &ret);
		//��ȡ��ǰ�� w
		for (int j = 0; j < N[i] * K[i] * K[i] * C[i]; j++)
		{
			read_layers_params >> ptr_layers_params_w[i][j];
		}
		//���������ݴ��䵽�豸��������
		ret = clEnqueueWriteBuffer(command_queue, layers_params_w_mem_obj[i], CL_TRUE, 0,
			N[i] * K[i] * K[i] * C[i] * sizeof(float), ptr_layers_params_w[i], 0, NULL, NULL);


		//��ȡ�������
		read_layers_params >> layer_param_type >> N[i];
		//��ǰ��ƫ�÷����ڴ�
		ptr_layers_params_b[i] = (float*)malloc(sizeof(float)*N[i]);
		//���仺����
		layers_params_b_mem_obj[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
			sizeof(float)*N[i], NULL, &ret);
		//��ȡ��ǰ�� b
		for (int j = 0; j < N[i]; j++)
		{
			read_layers_params >> ptr_layers_params_b[i][j];
		}
		//�������ݴ������豸������
		ret = clEnqueueWriteBuffer(command_queue, layers_params_b_mem_obj[i], CL_TRUE, 0,
			N[i] * sizeof(float), ptr_layers_params_b[i], 0, NULL, NULL);

		free(ptr_layers_params_w[i]);
		free(ptr_layers_params_b[i]);
		
	}

	//��ȡȫ���Ӳ����
	cout << "        fullconnection " << "initialization" << endl;

	//ȫ���Ӳ��� K = W*H*C
	//��ȡ����������Ϣ
	read_layers_params >> current_layer_name >> layer_param_type >> N[num_param_layer - 1] >> K[num_param_layer - 1];
	//���䵱ǰ����� w �����ڴ�
	ptr_layers_params_w[num_param_layer - 1] = (float*)malloc(sizeof(float)*N[num_param_layer - 1] * K[num_param_layer - 1]);
	//�����豸������
	layers_params_w_mem_obj[num_param_layer - 1] = clCreateBuffer(context, CL_MEM_READ_ONLY,
		sizeof(float)*N[num_param_layer - 1] * K[num_param_layer - 1], NULL, &ret);
	//��ȡ��ǰ�� w
	for (int j = 0; j < N[num_param_layer - 1] * K[num_param_layer - 1]; j++)
	{
		read_layers_params >> ptr_layers_params_w[num_param_layer - 1][j];
	}
	//���������ݴ��䵽�豸��������
	ret = clEnqueueWriteBuffer(command_queue, layers_params_w_mem_obj[num_param_layer - 1], CL_TRUE, 0,
		N[num_param_layer - 1] * K[num_param_layer - 1] * sizeof(float), ptr_layers_params_w[num_param_layer - 1], 0, NULL, NULL);

	//��ȡ�������
	read_layers_params >> layer_param_type >> N[num_param_layer - 1];
	//��ǰ��ƫ�÷����ڴ�
	ptr_layers_params_b[num_param_layer - 1] = (float*)malloc(sizeof(float)*N[num_param_layer - 1]);
	//���仺����
	layers_params_b_mem_obj[num_param_layer - 1] = clCreateBuffer(context, CL_MEM_READ_ONLY,
		sizeof(float)*N[num_param_layer - 1], NULL, &ret);
	//��ȡ��ǰ�� b
	for (int j = 0; j < N[num_param_layer - 1]; j++)
	{
		read_layers_params >> ptr_layers_params_b[num_param_layer - 1][j];
	}

	read_layers_params.close();

	//�������ݴ������豸������
	ret = clEnqueueWriteBuffer(command_queue, layers_params_b_mem_obj[num_param_layer - 1], CL_TRUE, 0,
		N[num_param_layer - 1] * sizeof(float), ptr_layers_params_b[num_param_layer - 1], 0, NULL, NULL);

	free(ptr_layers_params_w[num_param_layer - 1]);
	free(ptr_layers_params_b[num_param_layer - 1]);
	free(ptr_layers_params_w);
	free(ptr_layers_params_b);


	/**********�������в����������������***********/
	num_all_layers = 23;												//lightenCNN �ܹ���23��
	layers_data = (cl_mem *)malloc((num_all_layers + 1)*sizeof(cl_mem));//���ݻ�������24��
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
	/**************����ÿһ��Ĺ�����************/
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


	//�����ʼ�����
	cout << "lighten-CNN Net layer_params initialization end" << endl;
}


//��һ��ľ����
//W, H, C, K, N;
void Deepface::convolution(int index, uchar *input, bool debug)
{
	index_current_layer = 0;			//��ʼ����ǰ��

	//�����ڴ滺��������豸��Ϊÿ������
	cl_mem pi_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
		layer_param[index_current_layer][2] * (layer_param[index_current_layer][0] + layer_param[index_current_layer][6] * 2) * (layer_param[index_current_layer][1] + layer_param[index_current_layer][6] * 2) * sizeof(float), NULL, &ret);
	cl_mem ti_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
		((layer_param[index_current_layer][0] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1)*((layer_param[index_current_layer][1] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1)*(layer_param[index_current_layer][2] * layer_param[index_current_layer][3] * layer_param[index_current_layer][3])*sizeof(float), NULL, &ret);

	//��������˺�ͼƬ���뵽��Ӧ���ڴ滺��
	ret = clEnqueueWriteBuffer(command_queue, layers_data[index_current_layer], CL_TRUE, 0,
		layer_param[index_current_layer][0] * layer_param[index_current_layer][1] * sizeof(uchar), input, 0, NULL, NULL);

	/*********************�������Padding*****************/
	//����OpenCL�ں�
	cl_kernel kernel_p = clCreateKernel(program, "padding_char", &ret);

	//�����ں˲���
	ret = clSetKernelArg(kernel_p, 0, sizeof(cl_mem), (void*)&layers_data[index_current_layer]);
	ret = clSetKernelArg(kernel_p, 1, sizeof(cl_mem), (void*)&pi_mem_obj);
	ret = clSetKernelArg(kernel_p, 2, sizeof(cl_mem), (void*)&layer_param_mem_obj[index_current_layer]);

	//ִ���ں�
	size_t global_work_size3_p[3] = { layer_param[index_current_layer][0], layer_param[index_current_layer][1], layer_param[index_current_layer][2] };			//�����ڵ����
	ret = clEnqueueNDRangeKernel(command_queue, kernel_p, 3, NULL,
		global_work_size3_p, NULL, 0, NULL, NULL);

	//��ȡ�ڴ滺��C�����ر���C
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

	/*******************��������ת��*********************/
	//����OpenCL�ں�
	cl_kernel kernel_i = clCreateKernel(program, "transform_input", &ret);

	//�����ں˲���
	ret = clSetKernelArg(kernel_i, 0, sizeof(cl_mem), (void*)&pi_mem_obj);
	ret = clSetKernelArg(kernel_i, 1, sizeof(cl_mem), (void*)&ti_mem_obj);
	ret = clSetKernelArg(kernel_i, 2, sizeof(cl_mem), (void*)&layer_param_mem_obj[index_current_layer]);

	//ִ���ں�
	size_t global_work_size3[3] = { (layer_param[index_current_layer][0] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1, (layer_param[index_current_layer][1] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1, layer_param[index_current_layer][2] };			//�����ڵ����
	ret = clEnqueueNDRangeKernel(command_queue, kernel_i, 3, NULL,
		global_work_size3, NULL, 0, NULL, NULL);

	//��ȡ�ڴ滺��C�����ر���C
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

	/****************����˷�****************/

	clock_t begin, end;
	begin = clock();

	//cl_kernel kernel_m = clCreateKernel(program, "matrix_multiply", &ret);
	cl_kernel kernel_m = clCreateKernel(program, "matrix_multiply_singel", &ret);

	//�����ں˲���
	ret = clSetKernelArg(kernel_m, 0, sizeof(cl_mem), (void*)&ti_mem_obj);
	ret = clSetKernelArg(kernel_m, 1, sizeof(cl_mem), (void*)&layers_params_w_mem_obj[index]);
	ret = clSetKernelArg(kernel_m, 2, sizeof(cl_mem), (void*)&layers_params_b_mem_obj[index]);
	ret = clSetKernelArg(kernel_m, 3, sizeof(cl_mem), (void*)&layers_data[index_current_layer + 1]);
	ret = clSetKernelArg(kernel_m, 4, sizeof(cl_mem), (void*)&layer_param_mem_obj[index_current_layer]);

	//ִ���ں�
	size_t global_item_size[2] = { layer_param[index_current_layer][4], ((layer_param[index_current_layer][1] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1)*((layer_param[index_current_layer][0] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1) };			//�����ڵ����
	ret = clEnqueueNDRangeKernel(command_queue, kernel_m, 2, NULL,
		global_item_size, NULL, 0, NULL, NULL);

	//������Դ
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
	//�л�����ǰ��
	index_current_layer++;

	//�����ڴ滺��������豸��Ϊÿ������
	cl_mem pi_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
		layer_param[index_current_layer][2] * (layer_param[index_current_layer][0] + layer_param[index_current_layer][6] * 2) * (layer_param[index_current_layer][1] + layer_param[index_current_layer][6] * 2) * sizeof(float), NULL, &ret);
	cl_mem ti_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
		((layer_param[index_current_layer][0] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1)*((layer_param[index_current_layer][1] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1)*(layer_param[index_current_layer][2] * layer_param[index_current_layer][3] * layer_param[index_current_layer][3])*sizeof(float), NULL, &ret);


	/*********************�������Padding*****************/
	//����OpenCL�ں�
	cl_kernel kernel_p = clCreateKernel(program, "padding", &ret);

	//�����ں˲���
	ret = clSetKernelArg(kernel_p, 0, sizeof(cl_mem), (void*)&layers_data[index_current_layer]);
	ret = clSetKernelArg(kernel_p, 1, sizeof(cl_mem), (void*)&pi_mem_obj);
	ret = clSetKernelArg(kernel_p, 2, sizeof(cl_mem), (void*)&layer_param_mem_obj[index_current_layer]);

	//ִ���ں�
	size_t global_work_size3_p[3] = { layer_param[index_current_layer][0], layer_param[index_current_layer][1], layer_param[index_current_layer][2] };			//�����ڵ����
	ret = clEnqueueNDRangeKernel(command_queue, kernel_p, 3, NULL,
		global_work_size3_p, NULL, 0, NULL, NULL);


	/*******************��������ת��*********************/
	//����OpenCL�ں�
	cl_kernel kernel_i = clCreateKernel(program, "transform_input", &ret);

	//�����ں˲���
	ret = clSetKernelArg(kernel_i, 0, sizeof(cl_mem), (void*)&pi_mem_obj);
	ret = clSetKernelArg(kernel_i, 1, sizeof(cl_mem), (void*)&ti_mem_obj);
	ret = clSetKernelArg(kernel_i, 2, sizeof(cl_mem), (void*)&layer_param_mem_obj[index_current_layer]);

	//ִ���ں�
	size_t global_work_size3[3] = { (layer_param[index_current_layer][0] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1, (layer_param[index_current_layer][1] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1, layer_param[index_current_layer][2] };			//�����ڵ����
	ret = clEnqueueNDRangeKernel(command_queue, kernel_i, 3, NULL,
		global_work_size3, NULL, 0, NULL, NULL);


	/****************����˷�****************/
	cl_kernel kernel_m = clCreateKernel(program, "matrix_multiply", &ret);

	//�����ں˲���
	ret = clSetKernelArg(kernel_m, 0, sizeof(cl_mem), (void*)&ti_mem_obj);
	ret = clSetKernelArg(kernel_m, 1, sizeof(cl_mem), (void*)&layers_params_w_mem_obj[index]);
	ret = clSetKernelArg(kernel_m, 2, sizeof(cl_mem), (void*)&layers_params_b_mem_obj[index]);
	ret = clSetKernelArg(kernel_m, 3, sizeof(cl_mem), (void*)&layers_data[index_current_layer + 1]);
	ret = clSetKernelArg(kernel_m, 4, sizeof(cl_mem), (void*)&layer_param_mem_obj[index_current_layer]);

	//ִ���ں�
	size_t global_item_size[2] = { layer_param[index_current_layer][4], ((layer_param[index_current_layer][1] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1)*((layer_param[index_current_layer][0] + layer_param[index_current_layer][6] * 2 - layer_param[index_current_layer][3]) / layer_param[index_current_layer][5] + 1) };			//�����ڵ����
	size_t global_group_size[2] = { 8, 8 };
	ret = clEnqueueNDRangeKernel(command_queue, kernel_m, 2, NULL,
		global_item_size, global_group_size, 0, NULL, NULL);


	//������Դ
	ret = clReleaseKernel(kernel_i);
	ret = clReleaseKernel(kernel_m);
	ret = clReleaseKernel(kernel_p);
	ret = clReleaseMemObject(pi_mem_obj);
	ret = clReleaseMemObject(ti_mem_obj);
}

//W,H,C,K,N
void Deepface::slice_etlwise()
{
	//�л�����ǰ��
	index_current_layer++;

	//����OpenCL�ں�
	cl_kernel kernel_m = clCreateKernel(program, "slice_etlwise", &ret);

	//�����ں˲���
	ret = clSetKernelArg(kernel_m, 0, sizeof(cl_mem), (void*)&layers_data[index_current_layer]);
	ret = clSetKernelArg(kernel_m, 1, sizeof(cl_mem), (void*)&layers_data[index_current_layer + 1]);
	ret = clSetKernelArg(kernel_m, 2, sizeof(cl_mem), (void*)&layer_param_mem_obj[index_current_layer]);

	//ִ���ں�
	size_t global_global_size[2] = { layer_param[index_current_layer][2] / 2, layer_param[index_current_layer][0] * layer_param[index_current_layer][1] };			//�����ڵ����
	ret = clEnqueueNDRangeKernel(command_queue, kernel_m, 2, NULL,
		global_global_size, NULL, 0, NULL, NULL);

	ret = clReleaseKernel(kernel_m);
}


//W, H, C, K
void Deepface::pooling()
{
	//�л�����ǰ��
	index_current_layer++;

	//����OpenCL�ں�
	cl_kernel kernel = clCreateKernel(program, "pool", &ret);

	//�����ں˲���
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&layers_data[index_current_layer]);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&layers_data[index_current_layer + 1]);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&layer_param_mem_obj[index_current_layer]);

	//ִ���ں�
	size_t global_work_size[3] = { (layer_param[index_current_layer][0] / layer_param[index_current_layer][3]), (layer_param[index_current_layer][1] / layer_param[index_current_layer][3]), layer_param[index_current_layer][2] };			//�����ڵ����
	
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 3, NULL,
		global_work_size, NULL, 0, NULL, NULL);

	ret = clReleaseKernel(kernel);
}


//W, H, C, N;
void Deepface::full_connection(int index, float *full_connection_result)
{
	//�л�����ǰ��
	index_current_layer++;

	//�����ڴ滺��������豸��Ϊÿ������
	//����ȫ���Ӳ����ʱ����
	cl_mem t_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		layer_param[index_current_layer][4] * layer_param[index_current_layer][0] * layer_param[index_current_layer][1] * layer_param[index_current_layer][2] * sizeof(float), NULL, &ret);


	//����OpenCL�ں�
	cl_kernel kernel_m = clCreateKernel(program, "full_connection", &ret);

	//�����ں˲���
	ret = clSetKernelArg(kernel_m, 0, sizeof(cl_mem), (void*)&layers_data[index_current_layer]);
	ret = clSetKernelArg(kernel_m, 1, sizeof(cl_mem), (void*)&layers_params_w_mem_obj[index]);
	ret = clSetKernelArg(kernel_m, 2, sizeof(cl_mem), (void*)&t_mem_obj);
	ret = clSetKernelArg(kernel_m, 3, sizeof(cl_mem), (void*)&layers_params_b_mem_obj[index]);
	ret = clSetKernelArg(kernel_m, 4, sizeof(cl_mem), (void*)&layers_data[index_current_layer + 1]);
	ret = clSetKernelArg(kernel_m, 5, sizeof(cl_mem), (void*)&layer_param_mem_obj[index_current_layer]);

	//ִ���ں�
	size_t global_global_size[2] = { layer_param[index_current_layer][4], layer_param[index_current_layer][0] * layer_param[index_current_layer][1] * layer_param[index_current_layer][2] };			//�����ڵ����
	ret = clEnqueueNDRangeKernel(command_queue, kernel_m, 2, NULL,
		global_global_size, NULL, 0, NULL, NULL);

	//��ȡ�ڴ滺��C�����ر���C
	ret = clEnqueueReadBuffer(command_queue, layers_data[index_current_layer + 1], CL_TRUE, 0,
		layer_param[index_current_layer][4] * sizeof(float), full_connection_result, 0, NULL, NULL);

	ret = clReleaseKernel(kernel_m);
	ret = clReleaseMemObject(t_mem_obj);
}
