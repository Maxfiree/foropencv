#include "stdafx.h"
#include "dps.h"
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

Mat Hisg(const Mat M)
{

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
		for (p = 0;p < Mhis.cols;p++)    //描绘直方图
		{
			amount[p] = amount[p] / 10;
			if (amount[p] > 299)
				amount[p] = 299;
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

Mat Glvls(const Mat M)
{
	Mat Me = M.clone();
	int amount[256] = { 0 };
	int p = 0;

	for (int i = 0;i<Me.rows;i++)        //检测不同像素的个数
	{
		for (int j = 0;j<Me.cols;j++)
		{
			p = Me.at<uchar>(i, j);
			if (p < 100 || p>180) Me.at<uchar>(i, j) = 30;
			else Me.at<uchar>(i, j) = 160;


		}
	}


	return Me;
}