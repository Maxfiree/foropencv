#pragma once
#include"stdafx.h"
#include<iostream>
#include<opencv.hpp>
using namespace std;
using namespace cv;
Mat CreatW(long x, long y);
void PaintW(Mat* M, long x, long y);
char* StrToChar(CString str);
void ExFea(int p[][5][25]);
int Alg1(const Mat im, int Fea[][5][25]);
int Alg2(const Mat im, int Fea[][5][25]);
int Alg3(const Mat im, int Fea[][5][25]);
int Alg4(const Mat im, int Fea[][5][25]);