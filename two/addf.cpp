#include "stdafx.h"
#include "dps.h"

//CS类型转CHAR
char* StrToChar(CString str)
{
	int n = str.GetLength();
	int len = WideCharToMultiByte(CP_ACP, 0, str, str.GetLength(), NULL, 0, NULL, NULL);
	char * src = new char[len + 1];   //以字节为单位
	WideCharToMultiByte(CP_ACP, 0, str, str.GetLength() + 1, src, len + 1, NULL, NULL);
	src[len] = '\0';               //多字节字符以'\0'结束
	return src;
}
/*********************************************************************
*                           椒盐化噪声函数                           *
*********************************************************************/
Mat addSaltNoise(const Mat srcImage, int n)
{
	Mat dstImage = srcImage.clone();
	srand(time(NULL));
	for (int k = 0; k < n; k++)
	{
		//随机取值行列  
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//图像通道判定  
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 255;       //盐噪声  
		}
		else
		{
			dstImage.at<Vec3b>(i, j)[0] = 255;
			dstImage.at<Vec3b>(i, j)[1] = 255;
			dstImage.at<Vec3b>(i, j)[2] = 255;
		}
	}
	for (int k = 0; k < n; k++)
	{
		//随机取值行列  
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//图像通道判定  
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 0;     //椒噪声  
		}
		else
		{
			dstImage.at<Vec3b>(i, j)[0] = 0;
			dstImage.at<Vec3b>(i, j)[1] = 0;
			dstImage.at<Vec3b>(i, j)[2] = 0;
		}
	}
	return dstImage;
};



/*********************************************************************
*                          直方图函数                          *
*********************************************************************/

Mat Hisg(const Mat Mt)
{
	Mat M = Mt.clone();
	    int Max = 0;
		Mat Mhis(300, 256 * 2, CV_8UC1);
		int amount[256 * 2] = { 0 };
		int p = 0;
		for (int i = 0;i < Mhis.rows;i++)        //初始化为白色
		{
			for (int j = 0;j < Mhis.cols;j++)
			{
				Mhis.at<uchar>(i, j) = 255;
			}
		}
		for (int i = 0;i < M.rows;i++)        //检测不同像素的个数
		{
			for (int j = 0;j < M.cols;j++)
			{
				p = M.at<uchar>(i, j);
				amount[2 * p]++;
			}
		}
		for (p = 0;p < 256;p++)
		{
			if (amount[2 * p] > Max)
				Max = amount[2 * (p)];
		}
		for (p = 0;p < Mhis.cols;p++)    //描绘直方图
		{
			amount[p] = amount[p] / ((Max/300)+1);
			//if (amount[p] > 299)
			//	amount[p] = 299;
			for (int i = Mhis.rows - amount[p] - 1;i < Mhis.rows;i++)      //描绘直方图
			{
				Mhis.at<uchar>(i, p) = 0;
			}

		}

	//else
	//{
	//	Mat Mhis(300, 256 * 2, CV_8UC1);
	//	int amount[256 * 2] = { 0 };
	//	int p = 0;
	//	for (int i = 0;i < Mhis.rows;i++)        //初始化为白色
	//	{
	//		for (int j = 0;j < Mhis.cols;j++)
	//		{
	//			Mhis.at<uchar>(i, j) = 255;
	//		}
	//	}
	//	for (int i = 0;i < M.rows;i++)        //检测不同像素的个数
	//	{
	//		for (int j = 0;j < M.cols;j++)
	//		{
	//			p = M.at<uchar>(i, j);
	//			amount[2 * p]++;
	//		}
	//	}
	//	for (p = 0;p < Mhis.cols;p++)    //描绘直方图
	//	{
	//		amount[p] = amount[p] / 10;
	//		if (amount[p] > 299)
	//			amount[p] = 299;
	//		for (int i = Mhis.rows - amount[p] - 1;i < Mhis.rows;i++)      //描绘直方图
	//		{
	//			Mhis.at<uchar>(i, p) = 0;
	//		}

	//	}
	//}

	return Mhis;
}

/*********************************************************************
*                         均衡化/归一化函数                          *
*********************************************************************/

Mat Eqlz(const Mat M,int Range[])
{
	Mat Me = M.clone();
	int amount[256] = { 0 };
	int p = 0;
	int sum[256] = { 0 };

		for (int i = 0;i<Me.rows;i++)        //检测不同像素的个数
		{
			for (int j = 0;j<Me.cols;j++)
			{
				p = Me.at<uchar>(i, j);
					amount[p]++;
			}
		}
		sum[0] = amount[0];
		for (int k = 1;k < 256;k++)
		{
			sum[k] = amount[k]+sum[k-1];
		}

		for (int i = 0;i<Me.rows;i++)        //检测不同像素的个数
		{
			for (int j = 0;j<Me.cols;j++)
			{
				p = Me.at<uchar>(i, j);
				Me.at<uchar>(i, j) = Range[0]+(Range[1]-Range[0])* sum[p] / (Me.rows*Me.cols);
			}
		}





	return Me;
}

/*********************************************************************
*                          灰度分层函数                           *
*********************************************************************/

Mat Glvls(const Mat M,int Rang[],int Mark)
{
	Mat Me = M.clone();
	int amount[256] = { 0 };
	int p = 0;
	if (Mark==1)
	{

		for (int i = 0;i < Me.rows;i++)        //检测不同像素的个数
		{
			for (int j = 0;j < Me.cols;j++)
			{
				p = Me.at<uchar>(i, j);
				if (p >= Rang[0] && p <= Rang[1])
					Me.at<uchar>(i, j) = 200;



			}
		}
	}
	else
	{

		for (int i = 0;i < Me.rows;i++)        //检测不同像素的个数
		{
			for (int j = 0;j < Me.cols;j++)
			{
				p = Me.at<uchar>(i, j);
				if (p >= Rang[0] && p <= Rang[1])
					Me.at<uchar>(i, j) = 155;
				else
					Me.at<uchar>(i, j) = 30;



			}
		}
	}



	return Me;
}

/*********************************************************************
*                          伽玛变换函数                           *
*********************************************************************/

Mat Gamma_ma(const Mat M,double y)
{
	Mat Me = M.clone();
	int p = 0;
	double c=255./ pow(255, y);

	for (int i = 0;i<Me.rows;i++)        //检测不同像素的个数
	{
		for (int j = 0;j<Me.cols;j++)
		{
			p = Me.at<uchar>(i, j);
			Me.at<uchar>(i, j) = (uchar)(c* pow(p, y));


		}
	}


	return Me;
}

/*********************************************************************
*                         DFT与IDFT                     *
*********************************************************************/
void Dft_IDft(const Mat M)
{
	Mat inputim = M.clone();
	int w = getOptimalDFTSize(inputim.cols);
	int h = getOptimalDFTSize(inputim.rows);//获取最佳尺寸，快速傅立叶变换要求尺寸为2的n次方  
	Mat padded;
	copyMakeBorder(inputim, padded, 0, h - inputim.rows, 0, w - inputim.cols, BORDER_CONSTANT, Scalar::all(0));//填充图像保存到padded中  
	Mat plane[] = { Mat_<float>(padded),Mat::zeros(padded.size(),CV_32F) };//创建通道  
	Mat complexIm;
	merge(plane, 2, complexIm);//合并通道  
	dft(complexIm, complexIm);//进行傅立叶变换，结果保存在自身  
	split(complexIm, plane);//分离通道  
	magnitude(plane[0], plane[1], plane[0]);//获取幅度图像，0通道为实数通道，1为虚数，因为二维傅立叶变换结果是复数  
	int cx = padded.cols / 2;int cy = padded.rows / 2;//一下的操作是移动图像，左上与右下交换位置，右上与左下交换位置  
	Mat temp;
	Mat part1(plane[0], Rect(0, 0, cx, cy));
	Mat part2(plane[0], Rect(cx, 0, cx, cy));
	Mat part3(plane[0], Rect(0, cy, cx, cy));
	Mat part4(plane[0], Rect(cx, cy, cx, cy));


	part1.copyTo(temp);
	part4.copyTo(part1);
	temp.copyTo(part4);

	part2.copyTo(temp);
	part3.copyTo(part2);
	temp.copyTo(part3);
	//******************************************************************* 
	Mat &mag = plane[0];


	//Mat _complexim(complexIm,Rect(padded.cols/4,padded.rows/4,padded.cols/2,padded.rows/2));  
	//opyMakeBorder(_complexim,_complexim,padded.rows/4,padded.rows/4,padded.cols/4,padded.cols/4,BORDER_CONSTANT,Scalar::all(0.75));  
	Mat _complexim;
	complexIm.copyTo(_complexim);//把变换结果复制一份，进行逆变换，也就是恢复原图  
	Mat iDft[] = { Mat::zeros(plane[0].size(),CV_32F),Mat::zeros(plane[0].size(),CV_32F) };//创建两个通道，类型为float，大小为填充后的尺寸  
	idft(_complexim, _complexim);//傅立叶逆变换  
	split(_complexim, iDft);//结果貌似也是复数  
	//magnitude(iDft[0], iDft[1], iDft[0]);//分离通道，主要获取0通道  
	normalize(iDft[0], iDft[0], 1, 0, CV_MINMAX);//归一化处理，float类型的显示范围为0-1,大于1为白色，小于0为黑色  
	imshow("idft", iDft[0]);//显示逆变换  
							//*******************************************************************  
	plane[0] += Scalar::all(1);//傅立叶变换后的图片不好分析，进行对数处理，结果比较好看  
	log(plane[0], plane[0]);
	normalize(plane[0], plane[0], 1, 0, CV_MINMAX);

	imshow("dft", plane[0]);
}

/*********************************************************************
*                          理想低通滤波器                     *
*********************************************************************/

Mat ideal_Low_Pass_Filter(Mat src,int D0) 
{
	Mat img=src.clone();

	//调整图像加速傅里叶变换  
	int M = getOptimalDFTSize(img.rows);
	int N = getOptimalDFTSize(img.cols);
	Mat padded;
	copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));
	//记录傅里叶变换的实部和虚部  
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexImg;
	merge(planes, 2, complexImg);
	//进行傅里叶变换  
	dft(complexImg, complexImg);
	//获取图像  
	Mat &mag = complexImg;
	mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));//这里为什么&上-2具体查看opencv文档  
														//其实是为了把行和列变成偶数 -2的二进制是11111111.......10 最后一位是0  
														//获取中心点坐标  
	int cx = padded.cols / 2;int cy = padded.rows / 2;//一下的操作是移动图像，左上与右下交换位置，右上与左下交换位置  
	//调整频域  
	Mat tmp;
	Mat q0(mag, Rect(0, 0, cx, cy));
	Mat q1(mag, Rect(cx, 0, cx, cy));
	Mat q2(mag, Rect(0, cy, cx, cy));
	Mat q3(mag, Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	//Do为自己设定的阀值具体看公式  
	//double D0 = 60;
	//处理按公式保留中心部分  
	for (int y = 0; y < mag.rows; y++) {
		double* data = mag.ptr<double>(y);
		for (int x = 0; x < mag.cols; x++) {
			double d = sqrt(pow((y - cy), 2) + pow((x - cx), 2));
			if (d <= D0) {

			}
			else {
				data[x] = 0;
			}
		}
	}

	split(mag, planes);//分离通道  
	magnitude(planes[0], planes[1], planes[0]);//获取幅度图像，0通道为实数通道，1为虚数，因为二维傅立叶变换结果是复数  

	planes[0] += Scalar::all(1);//傅立叶变换后的图片不好分析，进行对数处理，结果比较好看  
	log(planes[0], planes[0]);
	normalize(planes[0], planes[0], 1, 0, CV_MINMAX);
	imshow("滤波器", planes[0]);





	//imshow("滤波器", M_Fil);
	//再调整频域  
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	//逆变换  
	Mat invDFT, invDFTcvt;
	idft(mag, invDFT, DFT_SCALE | DFT_REAL_OUTPUT); // Applying IDFT  
	invDFT.convertTo(invDFTcvt, CV_8U);
	return invDFTcvt;
}

/*********************************************************************
*                         巴特沃斯低通滤波器                     *
*********************************************************************/
Mat Butterworth_Low_Paass_Filter(Mat src,int D0,int n)
{
	/*int n = 1;*///表示巴特沃斯滤波器的次数  
			  //H = 1 / (1+(D/D0)^2n)  
	Mat img=src.clone();

	int M = getOptimalDFTSize(img.rows);
	int N = getOptimalDFTSize(img.cols);
	Mat padded;
	copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexImg;
	merge(planes, 2, complexImg);

	dft(complexImg, complexImg);

	Mat mag = complexImg;
	mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));

	int cx = mag.cols / 2;
	int cy = mag.rows / 2;

	Mat tmp;
	Mat q0(mag, Rect(0, 0, cx, cy));
	Mat q1(mag, Rect(cx, 0, cx, cy));
	Mat q2(mag, Rect(0, cy, cx, cy));
	Mat q3(mag, Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	//double D0 = 100;

	for (int y = 0; y < mag.rows; y++) {
		float* data = mag.ptr<float>(y);
		for (int x = 0; x < mag.cols; x++) {
			double d = sqrt(pow((y - cy), 2) + pow((x - cx), 2));
			double s = (double)d / D0;
			double h = (double)1 / (1 + pow(s, 2 * n));

			//if (h <= 0.01)
			//{
			//	data[2*x] = 0;
			//	data[2*x+1] = 0;
			//}
			//else {
		        data[2*x] = data[2 * x]* h;
				data[2 * x + 1] = data[2 * x+1]*h;
			//}
		}
	}

	split(mag, planes);//分离通道  
	magnitude(planes[0], planes[1], planes[0]);//获取幅度图像，0通道为实数通道，1为虚数，因为二维傅立叶变换结果是复数  

	planes[0] += Scalar::all(1);//傅立叶变换后的图片不好分析，进行对数处理，结果比较好看  
	log(planes[0], planes[0]);
	normalize(planes[0], planes[0], 1, 0, CV_MINMAX);
	imshow("滤波器", planes[0]);

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	//逆变换  
	Mat invDFT, invDFTcvt;
	idft(complexImg, invDFT, DFT_SCALE | DFT_REAL_OUTPUT); // Applying IDFT  
	invDFT.convertTo(invDFTcvt, CV_8U);
	return invDFTcvt;
}

/*********************************************************************
*                         高斯低通滤波器                     *
*********************************************************************/
Mat Gauss_Low_Paass_Filter(Mat src, int sigma)
{
	/*int n = 1;*///表示巴特沃斯滤波器的次数  
				  //H = 1 / (1+(D/D0)^2n)  
	Mat img = src.clone();

	int M = getOptimalDFTSize(img.rows);
	int N = getOptimalDFTSize(img.cols);
	Mat padded;
	copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexImg;
	merge(planes, 2, complexImg);

	dft(complexImg, complexImg);

	Mat mag = complexImg;
	mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));

	int cx = mag.cols / 2;
	int cy = mag.rows / 2;

	Mat tmp;
	Mat q0(mag, Rect(0, 0, cx, cy));
	Mat q1(mag, Rect(cx, 0, cx, cy));
	Mat q2(mag, Rect(0, cy, cx, cy));
	Mat q3(mag, Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);


	//static const double pi = 3.1415926;
	//int xcenter = padded.cols/2;
	//int ycenter = padded.rows / 2;
	//double **gauss = new double *[padded.cols];
	//for (int i = 0;i<padded.cols;i++)
	//	gauss[i] = new double[padded.rows];//为每行分配空间（每行中有col个元素） 
	//double x2, y2;
	//double org = (double)1 / 2 * pi * sigma;
	//for (int i = 0; i < padded.cols; i++)
	//{
	//	x2 = pow(i - xcenter, 2);
	//	for (int j = 0; j < padded.rows; j++)
	//	{
	//		y2 = pow(j - ycenter, 2);
	//		double g = exp(-(x2 + y2) / (2 * sigma * sigma));
	//		g /= 2 * pi * sigma;
	//		gauss[i][j] = g/org;
	//	}
	//}


	for (int y = 0; y < mag.rows; y++) {
		float* data = mag.ptr<float>(y);
		for (int x = 0; x < mag.cols; x++) {
			double d = sqrt(pow((y - cy), 2) + pow((x - cx), 2));
			double s2 = pow(d,2) / (2*pow(sigma,2));
			double h = exp(-s2);

			//if (h <= 0.607)
			//{
			//	data[2*x] = 0;
			//	data[2*x+1] = 0;
			//}
			//else {
			data[2 * x] *= h;
			data[2 * x + 1] *= h;
			//}
		}
	}

	//for (int i = 0;i<padded.cols;i++)
	//	delete gauss[i];
	//delete[] gauss;

	split(mag, planes);//分离通道  
	magnitude(planes[0], planes[1], planes[0]);//获取幅度图像，0通道为实数通道，1为虚数，因为二维傅立叶变换结果是复数  

	planes[0] += Scalar::all(1);//傅立叶变换后的图片不好分析，进行对数处理，结果比较好看  
	log(planes[0], planes[0]);
	normalize(planes[0], planes[0], 1, 0, CV_MINMAX);
	imshow("滤波器", planes[0]);

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	//逆变换  
	Mat invDFT, invDFTcvt;
	idft(complexImg, invDFT, DFT_SCALE | DFT_REAL_OUTPUT); // Applying IDFT  
	invDFT.convertTo(invDFTcvt, CV_8U);
	return invDFTcvt;
}

/*********************************************************************
*                         高斯噪声                  *
*********************************************************************/

//Mat GaussNoise(const Mat src, int sm)
//{
//	Mat im = src.clone();
//    
//}

