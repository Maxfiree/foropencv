
// twoDlg.h: 头文件
//

#pragma once
#include <opencv.hpp>
#include "CvvImage.h"
#include <gdiplus.h>
#include "afxcmn.h"
#include "afxwin.h"
#pragma comment(lib,"gdiplus.lib")
using namespace cv;
using Gdiplus::Graphics;
using Gdiplus::Image;
using Gdiplus::Bitmap;


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
	CDC* m_pDCHis;
	CRect m_rectSrc;
	CRect m_rectDst;
	CRect m_rectHis;
	IplImage* iplImage;
	CvvImage imgSrc;
	CvvImage imgDst;
	CvvImage imgHis;


	int ON_INPUT = 0;                           //判断是否输入图像
	Mat im, inputim, outputim,Hisim;   //图像对象
	CString Gedit1,Gedit2,Gedit4,Gedit5;                //编辑框内容
	afx_msg void OnBnClickedButton1();
	afx_msg void OnBnClickedButton2();
	afx_msg void OnBnClickedButton3();
	afx_msg void OnBnClickedButton4();
	afx_msg void OnBnClickedButton5();
	afx_msg HBRUSH OnCtlColor(CDC* pDC, CWnd* pWnd, UINT nCtlColor); //透明化用
	afx_msg void OnBnClickedButton7();
	afx_msg void OnBnClickedButton6();
	afx_msg void OnBnClickedButton8();
	afx_msg void OnStnDblclickSrc();
	afx_msg void OnStnDblclickDst();
	afx_msg void OnBnClickedButton9();
	afx_msg void OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar);
	afx_msg void OnLbnSelchangeList4();
	afx_msg void OnStnDblclickHis();
	afx_msg void OnSize(UINT nType, int cx, int cy);


	CSliderCtrl m_slider;
	CSliderCtrl m_slider2;
	CListBox m_list;

	//GDI背景图初始化
	Gdiplus::GdiplusStartupInput m_GdiplusStarupInput;
	ULONG_PTR m_uGdiplusToken;
	Image* m_img=NULL;

	BOOL ImageFromIDResource(UINT nID, LPCTSTR sTR, Image * & pImg);
	



};



