#include "stdafx.h"
#include "Rec.h"


//创建白图
Mat CreatW(long x,long y)
{
	Mat M(x,y, CV_8UC1); 


	for (int i = 0;i<M.rows;i++)        //遍历每一行每一列并设置其像素值  
	{
		for (int j = 0;j<M.cols;j++)
		{
			M.at<uchar>(i, j) = 255;
		}
	}
	return M;
}


//即时作图
void PaintW(Mat* M, long x, long y)
{
	int row = M->rows;
	int col = M->cols;
	for (int i = y - 3;i < y + 4;i++)
	{
		for (int j = x - 3;j < x + 4;j++)
		{
			if(y>3&&y<row-3&&x>3&&x<col-3)
			M->at<uchar>(i, j) = 0;
		}

    }
}


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


//提取样本特征
void ExFea(int p[][5][25])
{
	int a, b, c,d,amt=0 ;
	Mat M;
	CString aC, bC;
	CString PathNa;
	for (a = 0;a < 10;a++)
	{
		for (b = 0;b < 5;b++)
		{
			for (c = 0;c < 5;c++)
			{
				for (d = 0;d < 5;d++)
				{
					amt = 0;
					aC.Format(_T("%d"), a);
					bC.Format(_T("%d"), b);
					PathNa = _T("C:\\Users\\Cashion\\Desktop\\something\\数字识别样本\\") + aC + _T("\\") + bC + _T(".jpg");
					M = imread(StrToChar(PathNa), 0);
					for (int i = c * 205 / 5;i < (c + 1) * 205 / 5;i++)         //遍历每一行每一列并提取特征  
					{
						for (int j = d * 205 / 5;j < (d + 1) * 205 / 5;j++)
						{
							if (!M.at<uchar>(i, j))
								amt++;

						}
					}
					p[a][b][5*c+d] = amt;   //提取特征值
				}
				
				
			}

		}
	}
}


//算法1  逐个样本对比
int Alg1(const Mat im, int Fea[][5][25])
{
	int Em[10][5] = { 0 };
	int Num[25] = { 0 };
	int ddd=0;




	for (int c = 0;c < 5;c++)
	{
		for (int d = 0;d < 5;d++)
		{
			for (int i = c * 205 / 5;i < (c + 1) * 205 / 5;i++)         //遍历输入的每一行每一列并提取特征  
			{
				for (int j = d * 205 / 5;j < (d + 1) * 205 / 5;j++)
				{
					if (!im.at<uchar>(i, j))
						Num[5 * c + d]++;

				}
			}
		}


	}


	for (int i = 0;i < 10;i++)        //计算输入与每个样本的欧式距离
	{
		for (int j = 0;j <5;j++)
		{
			for (int k = 0;k < 25;k++)
			{
				ddd = Num[k] - Fea[i][j][k];
				Em[i][j] += (ddd*ddd) / 25;
			}
			Em[i][j] = (int)sqrt(Em[i][j]);
		}
	}

	int min = Em[0][0];
	int TheN = 0;
	for (int i = 0;i <10;i++)           //取出欧式距离最小值的式
	{
		for (int j = 0;j < 5;j++)
		{
			if (Em[i][j] < min)
			{
				min = Em[i][j];
				TheN = i;
			}
		}
	}
	return TheN;
}

//算法2  逐个特征对比
int Alg2(const Mat im, int Fea[][5][25])
{
	int Em[10] = { 0 };
	int Dta[10][25] = { 0 };
	int Num[25] = { 0 };
	int ddd=0;
	int min2 = 0;




	for (int c = 0;c < 5;c++)
	{
		for (int d = 0;d < 5;d++)
		{
			for (int i = c * 205 / 5;i < (c + 1) * 205 / 5;i++)         //遍历输入的每一行每一列并提取特征  
			{
				for (int j = d * 205 / 5;j < (d + 1) * 205 / 5;j++)
				{
					if (!im.at<uchar>(i, j))
						Num[5 * c + d]++;

				}
			}
		}


	}


	for (int i = 0;i < 10;i++)        //提取最小距离样本
	{
		for (int c = 0;c < 5;c++)
		{
			for (int d = 0;d < 5;d++)
			{
				Dta[i][5 * c + d] = abs(Num[5 * c + d] - Fea[i][0][5 * c + d]);
				for (int k = 0;k < 5;k++)
				{
					min2= abs(Num[5 * c + d] - Fea[i][k][5 * c + d]);
					if (min2 < Dta[i][5 * c + d])
					{
						Dta[i][5 * c + d] = min2;

					}

				}
			}
		}

	}



	for (int i = 0;i < 10;i++)        //计算输入与每个样本的欧式距离
	{
			for (int k = 0;k < 25;k++)
			{
				Em[i]+= Dta[i][k]* Dta[i][k] / 25;
			}
			Em[i] = (int)sqrt(Em[i]);
	}

	int min = Em[0];
	int TheN = 0;
	for (int i = 0;i <10;i++)           //取出欧式距离最小值的式
	{
			if (Em[i]< min)
			{
				min = Em[i];
				TheN = i;
			}
	}
	return TheN;
}

//算法3  样本特征平均后计算距离
int Alg3(const Mat im, int Fea[][5][25])
{
	int Em[10] = { 0 };
	int Dta[10][25] = { 0 };
	int Num[25] = { 0 };
	int ddd = 0;
	int min2 = 0;




	for (int c = 0;c < 5;c++)
	{
		for (int d = 0;d < 5;d++)
		{
			for (int i = c * 205 / 5;i < (c + 1) * 205 / 5;i++)         //遍历输入的每一行每一列并提取特征  
			{
				for (int j = d * 205 / 5;j < (d + 1) * 205 / 5;j++)
				{
					if (!im.at<uchar>(i, j))
						Num[5 * c + d]++;

				}
			}
		}


	}


	for (int i = 0;i < 10;i++)        //提取平均距离样本
	{
		for (int c = 0;c < 5;c++)
		{
			for (int d = 0;d < 5;d++)
			{
				for (int k = 0;k < 5;k++)
				{
					min2 = abs(Num[5 * c + d] - Fea[i][k][5 * c + d]);
					Dta[i][5 * c + d] += min2;

				}
				Dta[i][5 * c + d] = Dta[i][5 * c + d] / 5;
			}
		}

	}



	for (int i = 0;i < 10;i++)        //计算输入与每个样本的欧式距离
	{
		for (int k = 0;k < 25;k++)
		{
			Em[i] += Dta[i][k] * Dta[i][k] / 25;
		}
		Em[i] = (int)sqrt(Em[i]);
	}

	int min = Em[0];
	int TheN = 0;
	for (int i = 0;i <10;i++)           //取出欧式距离最小值的式
	{
		if (Em[i]< min)
		{
			min = Em[i];
			TheN = i;
		}
	}
	return TheN;
}


//算法4 样本计算距离后取平均
int Alg4(const Mat im, int Fea[][5][25])
{
	int Em[10][5] = { 0 };
	int eM[10] = { 0 };
	int Num[25] = { 0 };
	int ddd;




	for (int c = 0;c < 5;c++)
	{
		for (int d = 0;d < 5;d++)
		{
			for (int i = c * 205 / 5;i < (c + 1) * 205 / 5;i++)         //遍历输入的每一行每一列并提取特征  
			{
				for (int j = d * 205 / 5;j < (d + 1) * 205 / 5;j++)
				{
					if (!im.at<uchar>(i, j))
						Num[5 * c + d]++;

				}
			}
		}


	}


	for (int i = 0;i < 10;i++)        //计算输入与每个样本的欧式距离
	{
		for (int j = 0;j <5;j++)
		{
			for (int k = 0;k < 25;k++)
			{
				ddd = Num[k] - Fea[i][j][k];
				Em[i][j] += (ddd*ddd) / 25;
			}
			Em[i][j] = (int)sqrt(Em[i][j]);
			eM[i] = Em[i][j]++;
		}
		eM[i] = eM[i]/5;
	}

	int min = eM[0];
	int TheN = 0;
	for (int i = 0;i <10;i++)           //取出欧式距离最小值的式
	{
	
			if (eM[i] < min)
			{
				min = eM[i];
				TheN = i;
			}
	}
	return TheN;
}
