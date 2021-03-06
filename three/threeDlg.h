
// threeDlg.h: 头文件
//
#include <opencv.hpp>
#include "CvvImage.h"
using namespace cv;
#pragma once


// CthreeDlg 对话框
class CthreeDlg : public CDialog
{
// 构造
public:
	CthreeDlg(CWnd* pParent = NULL);	// 标准构造函数

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_THREE_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持


// 实现
protected:
	HICON m_hIcon;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	CWnd * m_pWndDW;
	CDC* m_pDCDW;
	CRect m_rectDW;
	IplImage* iplImage;
	CvvImage imgDW;
	CPoint ptCursor;        //判断鼠标是否在框内
	CPoint insidePoint;     //鼠标相对框的坐标
	CRect rc,oprc;          //方框位置和方框相对屏幕位置
	CString str;
	Mat im;               //实时图像
	int mouse = 0;   //鼠标按下标志
	int n = 0;    //文件名
	int Fea[10][5][25]; //样本特征
	CString _PAT;    //文件路径
	CString _FileName;   //文件名
	afx_msg void OnLButtonDown(UINT nFlags, CPoint point);
	afx_msg void OnMouseMove(UINT nFlags, CPoint point);
	afx_msg void OnLButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnBnClickedButton1();
	afx_msg void OnBnClickedButton2();
	afx_msg void OnBnClickedButton3();
	afx_msg void OnBnClickedButton4();
	afx_msg void OnTimer(UINT_PTR nIDEvent);
};
