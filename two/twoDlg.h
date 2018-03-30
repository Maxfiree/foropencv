
// twoDlg.h: 头文件
//

#pragma once
#include <opencv.hpp>
#include "CvvImage.h"
using namespace cv;

// CtwoDlg 对话框
class CtwoDlg : public CDialogEx
{
// 构造
public:
	CtwoDlg(CWnd* pParent = NULL);	// 标准构造函数

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_TWO_DIALOG };
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
//	afx_msg void OnStnDblclickSrc();
//	afx_msg void OnStnClickedSrc();
//	afx_msg void OnStnDblclickSrc();

	CWnd* m_pWndSrc;
	CWnd* m_pWndDst;
	CDC* m_pDCSrc;
	CDC* m_pDCDst;
	CRect m_rectSrc;
	CRect m_rectDst;
	IplImage* iplImage;
	CvvImage imgSrc;
	CvvImage imgDst;

	int ON_INPUT = 0;                           //判断是否输入图像
	Mat im, im1, im2, im3, inputim, outputim;   //图像对象
	CString Gedit,Gedit2,Gedit3;                //文本框内容
	afx_msg void OnBnClickedButton1();
	afx_msg void OnBnClickedButton2();
	afx_msg void OnBnClickedButton3();
	afx_msg void OnBnClickedButton4();
	afx_msg void OnBnClickedButton5();

	afx_msg void OnBnClickedButton7();
	afx_msg void OnBnClickedButton6();
};


