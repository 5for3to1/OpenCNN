/************卷积层内核*************/
//param: W H C K N

//卷积层输入Padding
/**/
__kernel
void padding_char(__global uchar * image,__global float * padding_image,__global int * param)
{
	int gx = get_global_id(0);		//	w
	int gy = get_global_id(1);		//  h
	int gz = get_global_id(2);		//	c

	padding_image[gz*(param[0] + param[6] * 2)*(param[1] + param[6] * 2) + (gy + param[6])*(param[0] + param[6] * 2) + gx + param[6]] =
		(float)image[(gz*param[0] * param[1] + gy*param[0] + gx)];

	barrier(CLK_GLOBAL_MEM_FENCE);
	
	if (gx == 0 && gy == 0)
	{
		for (int i = 0; i < param[6]; i++)
		{
			for (int j = 0; j < param[0] + 2 * param[6]; j++)
			{
				padding_image[gz*(param[0] + param[6] * 2)*(param[1] + param[6] * 2) + i*(param[0] + param[6] * 2) + j] = 0;
				padding_image[gz*(param[0] + param[6] * 2)*(param[1] + param[6] * 2) + (i+param[6]+param[1])*(param[0] + param[6] * 2) + j] = 0;
			}
			for (int k = 0; k < param[1]; k++)
			{
				padding_image[gz*(param[0] + param[6] * 2)*(param[1] + param[6] * 2) + (k + param[6])*(param[0] + param[6] * 2) + i] = 0;
				padding_image[gz*(param[0] + param[6] * 2)*(param[1] + param[6] * 2) + (k + param[6])*(param[0] + param[6] * 2) + i + param[6] + param[0]] = 0;
			}
		}
	}
}

/*********/
__kernel
void padding(__global float * image, __global float * padding_image, __global int * param)
{
	int gx = get_global_id(0);		//	w
	int gy = get_global_id(1);		//  h
	int gz = get_global_id(2);		//	c
	/**/
	padding_image[gz*(param[0] + param[6] * 2)*(param[1] + param[6] * 2) + (gy + param[6])*(param[0] + param[6] * 2) + gx + param[6]] =
		image[gz*param[0] * param[1] + gy*param[0] + gx];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (gx == 0 && gy == 0)
	{
		for (int i = 0; i < param[6]; i++)
		{
			for (int j = 0; j < param[0] + 2 * param[6]; j++)
			{
				padding_image[gz*(param[0] + param[6] * 2)*(param[1] + param[6] * 2) + i*(param[0] + param[6] * 2) + j] = 0;
				padding_image[gz*(param[0] + param[6] * 2)*(param[1] + param[6] * 2) + (i + param[6] + param[1])*(param[0] + param[6] * 2) + j] = 0;
			}
			for (int k = 0; k < param[1]; k++)
			{
				padding_image[gz*(param[0] + param[6] * 2)*(param[1] + param[6] * 2) + (k + param[6])*(param[0] + param[6] * 2) + i] = 0;
				padding_image[gz*(param[0] + param[6] * 2)*(param[1] + param[6] * 2) + (k + param[6])*(param[0] + param[6] * 2) + i + param[6] + param[0]] = 0;
			}
		}
	}
}

//将一个三维矩阵经过滑动窗口后 转化为一个二维矩阵
__kernel
void transform_input(__global float * image, __global float * transform_image, __global int *param)
{
	//gx:W-K+1, gy:H-K+1, gz:C;
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int gz = get_global_id(2);
	for (int i = 0; i < param[3]; i++)
	{
		for (int j = 0; j < param[3]; j++)
		{
			transform_image[(((gy*((param[0] + param[6] * 2 - param[3]) / param[5] + 1) + gx)*param[2] + gz)*param[3] + i)*param[3] + j]
				= image[gz*(param[0] + param[6] * 2)*(param[1] + param[6] * 2) + (gy*param[5] + i)*(param[0] + param[6] * 2) + gx*param[5] + j];
		}
	}
}

//#define BLOCK_SIZE 8
//N*(C*S*S)的矩阵 * [（W-S+1）*（H-S+1）]*(C*S*S)的矩阵
__kernel
void matrix_multiply(__global float * transform_image, __global float *transform_kernel, __global float *bias, __global float *featureMap, __global int *param)
{
	__local float i_local[8][8];
	__local float k_local[8][8];
	float running_sum = bias[get_global_id(0)];

	int block_x = get_group_id(0);
	int block_y = get_group_id(1);
	int thread_x = get_local_id(0);
	int thread_y = get_local_id(1);

	int k_start = block_x*(param[2] * param[3] * param[3]) * 8;			//卷积核矩阵的第 block_x 行的第一个block的第一个行的第一个元素
	int k_end = k_start + param[2] * param[3] * param[3] - 1;			//卷积核矩阵的第 block_y 行的最后一个block的第一行的最后一个元素
	int i_start = block_y*(param[2] * param[3] * param[3]) * 8;			//输入矩阵的第 block_y 行的 字一个block的第一行的第一个元素

	for (int a = k_start, b = i_start; a <= k_end; a += 8, b += 8)
	{
		k_local[thread_x][thread_y] = transform_kernel[a + thread_x*(param[2] * param[3] * param[3]) + thread_y];
		i_local[thread_x][thread_y] = transform_image[b + thread_x*(param[2] * param[3] * param[3]) + thread_y];
		barrier(CLK_LOCAL_MEM_FENCE);

		for (int k = 0; k < 8; k++)
		{
			running_sum += k_local[thread_x][k] * i_local[thread_y][k];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	//barrier(CLK_GLOBAL_MEM_FENCE);
	featureMap[get_global_id(0)*get_global_size(1) + get_global_id(1)] = running_sum;
}

//N*(C*S*S)的矩阵 * [（W-S+1）*（H-S+1）]*(C*S*S)的矩阵
__kernel
void matrix_multiply_f(__global float * transform_image, __global float *transform_kernel, __global float *bias, __global float *featureMap, __global int *param)
{
	__local float i_local[8][8];
	__local float k_local[8][8];
	float running_sum = bias[get_global_id(0)];

	int block_x = get_group_id(0);
	int block_y = get_group_id(1);
	int thread_x = get_local_id(0);
	int thread_y = get_local_id(1);

	int k_start = block_x*(param[2] * param[3] * param[3]) * 8;	//卷积核矩阵的第 block_x 行的第一个block的第一个行的第一个元素
	int k_end = k_start + param[2] * param[3] * param[3] - 1;				//卷积核矩阵的第 block_y 行的最后一个block的第一行的最后一个元素
	int i_start = block_y*(param[2] * param[3] * param[3]) * 8;	//输入矩阵的第 block_y 行的 字一个block的第一行的第一个元素

	for (int a = k_start, b = i_start; a <= k_end; a += 8, b += 8)
	{
		k_local[thread_x][thread_y] = transform_kernel[a + thread_x*(param[2] * param[3] * param[3]) + thread_y];
		i_local[thread_x][thread_y] = transform_image[b + thread_x*(param[2] * param[3] * param[3]) + thread_y];
		barrier(CLK_LOCAL_MEM_FENCE);

		for (int k = 0; k < 8; k++)
		{
			running_sum += k_local[thread_x][k] * i_local[thread_y][k];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	featureMap[get_global_id(0)*get_global_size(1) + get_global_id(1)] = running_sum;
}

__kernel
void matrix_multiply_singel(__global float * transform_image, __global float *transform_kernel, __global float *bias, __global float *featureMap, __global int *param)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	
	float running_sum = bias[gx];
	for (int i = 0; i < param[2] * param[3] * param[3]; i++)
	{
		running_sum += transform_kernel[gx*param[2] * param[3] * param[3] + i] * transform_image[gy*param[2] * param[3] * param[3] + i];
	}
	barrier(CLK_GLOBAL_MEM_FENCE);

	featureMap[gx*((param[1] + param[6] * 2 - param[3]) / param[5] + 1)*((param[0] + param[6] * 2 - param[3]) / param[5] + 1) + gy] = running_sum;
	
	/*
	int gz = get_global_id(2);

	for (int i = 0; i < param[2] * param[3] * param[3] / 16; i++)
	{
		tem[(gy*param[4] + gx) * 16 + gz] += transform_kernel[gx*param[2] * param[3] * param[3] + i * 16 + gz] * transform_image[gy*param[2] * param[3] * param[3] + i * 16 + gz];
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	for (int len = 8; len >= 1; len /= 2)
	{
		if (gz < len)
		{
			tem[(gy*param[4] + gx) * 16 + gz] += tem[(gy*param[4] + gx) * 16 + gz + len];
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	if (gz == 0)
	{
		featureMap[gx*((param[1] + param[6] * 2 - param[3]) / param[5] + 1)*((param[0] + param[6] * 2 - param[3]) / param[5] + 1) + gy] = bias[gx] + tem[(gy*param[4] + gx) * 16 + gz];
	}
	*/
}

/***************池化层内核***************/
// W H C K
__kernel
void pool(__global float *input, __global float *result, __global int *param)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int gz = get_global_id(2);
	
	float maxvalue = input[gx * param[3] + gy * param[3] * param[0] + gz * param[0] * param[1]];
	if (maxvalue < input[gx * param[3] + 1 + gy * param[3] * param[0] + gz * param[0] * param[1]])
	{
		maxvalue = input[gx * param[3] + 1 + gy * param[3] * param[0] + gz * param[0] * param[1]];
	}
	if (maxvalue < input[gx * param[3] + (gy * param[3] + 1 ) * param[0] + gz * param[0] * param[1]])
	{
		maxvalue = input[gx * param[3] + (gy * param[3] + 1) * param[0] + gz * param[0] * param[1]];
	}
	if (maxvalue < input[gx * param[3] + 1 + (gy * param[3] + 1) * param[0] + gz * param[0] * param[1]])
	{
		maxvalue = input[gx * param[3] + 1 + (gy * param[3] + 1) * param[0] + gz * param[0] * param[1]];
	}

	result[gx + gy * (param[0] / param[3]) + gz * (param[0] / param[3]) * (param[1] / param[3])] = maxvalue;
	
}

/*******************全连接层内核*******************/
//N*(C*S*S)的矩阵 * [（W-S+1）*（H-S+1）]*(C*S*S)的矩阵    
//W H C N
__kernel
void full_connection(__global float * full_connection_input, __global float *full_connection_kernel, __global float * full_connection_tem, __global float * full_connection_bias, __global float *result, __global int *param)
{

	int gx = get_global_id(0);
	int gy = get_global_id(1);

	full_connection_tem[gx* param[2] * param[1] * param[0] + gy] = full_connection_kernel[gx* param[2] * param[1] * param[0] + gy] * full_connection_input[gy];
	barrier(CLK_GLOBAL_MEM_FENCE);

	//循环
	/*
	if (gy == 0)
	{
		result[gx] = full_connection_bias[gx];
		barrier(CLK_GLOBAL_MEM_FENCE);

		#pragma unroll 4
		for (int i = 0; i < param[0] * param[1] * param[2]; i++)
		{
			result[gx] += full_connection_tem[gx*param[0] * param[1] * param[2] + i];
		}
	}*/
	
	//迭代
	/**/
	//#pragma unroll 
	int len;
	for (len = param[0] * param[1] * param[2] / 2; len >= 64*128; len /= 2)
	{
		if (gy < len)
		{
			full_connection_tem[gx*param[0] * param[1] * param[2] + gy] += full_connection_tem[gx*param[0] * param[1] * param[2] + gy + len];
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	if (gy == 0)
	{
		result[gx] = full_connection_bias[gx];
		barrier(CLK_GLOBAL_MEM_FENCE);
		
		for (int j = 0; j < len*2; j++)
		{
			result[gx] += full_connection_tem[gx*param[0] * param[1] * param[2] + j];
		}
	}
	

	//迭代2
	/*
	int len;
	int base, offset;
	for (len = 2; len <= param[0] * param[1] * param[2]; len *= 2)
	{
		base = gy / len;
		offset = len / 2;
		full_connection_tem[base] += full_connection_tem[base + offset];

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	if (gy == 0)
	{
		result[gx] = full_connection_bias[gx] + full_connection_tem[gx*param[0] * param[1] * param[2]];
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	*/

}

/*********************************slice和etlwise层内核函数***********************************/
__kernel
void slice_etlwise(__global float * input, __global float * output, __global int * param)
{
	int gx = get_global_id(0);		//通道维度
	int gy = get_global_id(1);

	//output[gx*param[0] * param[1] + gy] =
	//	input[gx*param[0] * param[1] + gy] > input[(gx + param[2] / 2)*param[0] * param[1] + gy] ? input[gx*param[0] * param[1] + gy] : input[(gx + param[2] / 2)*param[0] * param[1] + gy];
	if (input[gx*param[0] * param[1] + gy] > input[(gx + param[2] / 2)*param[0] * param[1] + gy])
	{
		output[gx*param[0] * param[1] + gy] = input[gx*param[0] * param[1] + gy];
	}
	else
	{
		output[gx*param[0] * param[1] + gy] = input[(gx + param[2] / 2)*param[0] * param[1] + gy];
	}
}