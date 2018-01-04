#include"Deepface.h"

void thread01()
{
	Deepface lighten_CNN;
	lighten_CNN.forward("feature_512_1");
}

void thread02()
{
	Deepface lighten_CNN;
	lighten_CNN.forward("feature_512_2");
}

void main()
{
	Deepface lighten_CNN;
	lighten_CNN.forward("feature_512");

	//实例化lighten-CNN
	//Deepface lighten_CNN_1;
	//Deepface lighten_CNN_2;

	//clock_t begin, end;
	//begin = clock();

	//thread task01(thread01);
	//thread task02(thread02);
	//task01.join();
	//task02.join();
	//task01.detach();
	//task02.detach();

	//end = clock();
	//double time = (double)(end - begin) / CLOCKS_PER_SEC;
	//cout << "lighten-CNN运行时间：" << time << "s" << endl;

	//system("pause");
}