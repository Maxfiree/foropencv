
// twoDlg.cpp: 实现文件
//

#include "stdafx.h"

#include "two.h"
#include "twoDlg.h"
#include "afxdialogex.h"

#include"dps.h"


#ifdef _DEBUG
#define new DEBUG_NEW
#endif



// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CtwoDlg 对话框



CtwoDlg::CtwoDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_TWO_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CtwoDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CtwoDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
//	ON_STN_DBLCLK(IDC_SRC, &CtwoDlg::OnStnDblclickSrc)
//ON_STN_CLICKED(IDC_SRC, &CtwoDlg::OnStnClickedSrc)
//ON_STN_DBLCLK(IDC_SRC, &CtwoDlg::OnStnDblclickSrc)
ON_BN_CLICKED(IDC_BUTTON1, &CtwoDlg::OnBnClickedButton1)
ON_BN_CLICKED(IDC_BUTTON2, &CtwoDlg::OnBnClickedButton2)
ON_BN_CLICKED(IDC_BUTTON3, &CtwoDlg::OnBnClickedButton3)
ON_BN_CLICKED(IDC_BUTTON4, &CtwoDlg::OnBnClickedButton4)
ON_BN_CLICKED(IDC_BUTTON5, &CtwoDlg::OnBnClickedButton5)
ON_BN_CLICKED(IDC_BUTTON7, &CtwoDlg::OnBnClickedButton7)
END_MESSAGE_MAP()


// CtwoDlg 消息处理程序

BOOL CtwoDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码

	//CEdit*  pEdit = (CEdit*)GetDlgItem(IDC_EDIT1);//获取相应的编辑框ID
	GetDlgItem(IDC_EDIT1)->SetWindowText(_T("8000")); //设置编辑框默认显示的内容 
	GetDlgItem(IDC_EDIT2)->SetWindowText(_T("3")); //设置编辑框默认显示的内容 
	GetDlgItem(IDC_EDIT3)->SetWindowText(_T("3")); //设置编辑框默认显示的内容 

	GetDlgItem(IDC_STATIC_1)->SetWindowTextW(_T("噪点数"));    //设置静态文本框
	GetDlgItem(IDC_STATIC_2)->SetWindowTextW(_T("领域边长"));

	//m_pWndSrc = GetDlgItem(IDC_SRC);
	//m_pWndDst = GetDlgItem(IDC_DST);
	m_pDCSrc = GetDlgItem(IDC_SRC)->GetDC();                    //设置方框图片区域参数
	m_pDCDst = GetDlgItem(IDC_DST)->GetDC();
	GetDlgItem(IDC_DST)->GetClientRect(&m_rectSrc);
	GetDlgItem(IDC_DST)->GetClientRect(&m_rectDst);


	//CStatic test;
	//test.Create("my static",
	//	WS_CHILD | WS_VISIBLE | SS_CENTERIMAGE | SS_NOTIFY, CRect(1, 3, 90, 90),
	//	this, IDC_STA_TEST);
	//ON_BN_CLICKED(IDC_STA_TEST, OnTest);
	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CtwoDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialog::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CtwoDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialog::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CtwoDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}





/*************************************************************************************************
**************************************************************************************************

                                  以下为消息时间处理函数
**************************************************************************************************
*************************************************************************************************/





/***********************************************************************************************
                                         读取原图
*************************************************************************************************/

void CtwoDlg::OnBnClickedButton1()
{
	// TODO: 在此添加控件通知处理程序代码

	CButton* pBtn = (CButton*)GetDlgItem(IDC_CHECK1);   //获取按钮ID
	int state = pBtn->GetCheck();                       //读取按钮值

	TCHAR szFilter[] = _T("图像文件(*.bmp,*.jpg)|*.bmp;*.jpg|视频文件(*.avi)|*.avi|所有文件(*.*)|*.*||");   //设置过滤格式
	CFileDialog  dlgFile(TRUE, NULL, NULL, OFN_HIDEREADONLY, szFilter, NULL);    //设置窗口属性
	if (IDOK == dlgFile.DoModal())           //弹出文件浏览窗
	{

		CString strOpenFilePath = dlgFile.GetPathName();

		//将CString类型的文件名转化为char*类型
		int n = strOpenFilePath.GetLength();
		int len = WideCharToMultiByte(CP_ACP, 0, strOpenFilePath, strOpenFilePath.GetLength(), NULL, 0, NULL, NULL);
		char * _src = new char[len + 1];   //以字节为单位
		WideCharToMultiByte(CP_ACP, 0, strOpenFilePath, strOpenFilePath.GetLength() + 1, _src, len + 1, NULL, NULL);
		_src[len] = '\0';               //多字节字符以'\0'结束

	
		im = imread(_src, state);   //读取原图
		namedWindow("原图", 1);
		namedWindow("原图直方图", 1);
		imshow("原图", im);           //弹出新窗口
		imshow("原图直方图",Hisg(im));

		iplImage = &(IplImage)im;
		imgSrc.CopyOf(iplImage, 1);
		imgSrc.DrawToHDC(m_pDCSrc->m_hDC, &m_rectSrc);         //将图片放置图片框内

		outputim = im;//原图为im
		ON_INPUT = 1;



	}
}

/***********************************************************************************************
                                      加噪处理
*************************************************************************************************/

void CtwoDlg::OnBnClickedButton2()
{

	if (!ON_INPUT)
		return;
	// TODO: 在此添加控件通知处理程序代码
	GetDlgItem(IDC_EDIT1)->GetWindowText(Gedit);    //获取编辑框内容
	int n= _ttoi(Gedit);               //Unicode转int

	if (!(n > 0)) {
		MessageBox(_T("必须输入大于0的数"));
		return;
	}

	inputim = outputim;
	im1 = addSaltNoise(inputim, n);    //加椒盐噪声处理
	namedWindow("处理后", 1);
	imshow("处理后", im1);

	iplImage = &(IplImage)im1;
	imgDst.CopyOf(iplImage, 1);
	imgDst.DrawToHDC(m_pDCDst->m_hDC, &m_rectDst);

	outputim = im1;
	imwrite(_dst1_, im1);                       //输出为im1
}

/***********************************************************************************************
                                      平滑滤波处理
*************************************************************************************************/

void CtwoDlg::OnBnClickedButton3()
{
	if (!ON_INPUT)
		return;
	// TODO: 在此添加控件通知处理程序代码
	GetDlgItem(IDC_EDIT2)->GetWindowText(Gedit2);    //获取编辑框内容
	int n = _ttoi(Gedit2);               //CSstring转int

	if (!(n%2)||n<0 || n>100) {
		MessageBox(_T("必须输入合理范围的奇数"));
		return;
	}

	inputim = outputim; ;                               //平滑（均值）滤波处理
	blur(inputim, im2, Size(n, n));
	namedWindow("处理后", 1);
	imshow("处理后", im2);

	iplImage = &(IplImage)im2;
	imgDst.CopyOf(iplImage, 1);
	imgDst.DrawToHDC(m_pDCDst->m_hDC, &m_rectDst);

	outputim = im2;
	imwrite(_dst2_, im2);                  //输出位im2
}

/***********************************************************************************************
                                        中值滤波处理
*************************************************************************************************/

void CtwoDlg::OnBnClickedButton4()
{
	if (!ON_INPUT)
		return;
	// TODO: 在此添加控件通知处理程序代码
	GetDlgItem(IDC_EDIT3)->GetWindowText(Gedit3);    //获取编辑框内容
	int n = _ttoi(Gedit3);               //Unicode转int

	if (!(n % 2) || n<0 || n>100) {
		MessageBox(_T("必须输入合理范围的奇数"));
		return;
	}

	inputim = outputim;                                         //中值滤波处理
	medianBlur(inputim, im3, n);
	namedWindow("处理后", 1);
	imshow("处理后", im3);

	iplImage = &(IplImage)im3;
	imgDst.CopyOf(iplImage, 1);
	imgDst.DrawToHDC(m_pDCDst->m_hDC, &m_rectDst);

	outputim = im3;
	imwrite(_dst3_, im3);                 //输出为Im3
}


/***********************************************************************************************
                                    初始化图片
*************************************************************************************************/

void CtwoDlg::OnBnClickedButton5()
{
	if (!ON_INPUT)
		return;
	// TODO: 在此添加控件通知处理程序代码
	inputim = im;
	namedWindow("处理后", 1);
	imshow("处理后", inputim);

	iplImage = &(IplImage)im;
	imgDst.CopyOf(iplImage, 1);
	imgDst.DrawToHDC(m_pDCDst->m_hDC, &m_rectDst);

	outputim = im;
}




/***********************************************************************************************
                               均衡化
*************************************************************************************************/


void CtwoDlg::OnBnClickedButton7()
{
	if (!ON_INPUT)
		return;
	// TODO: 在此添加控件通知处理程序代码
	static int mode = 0;
	inputim = outputim;                                        
	outputim=Eqlz(inputim,mode);
	mode++;
	if (mode > 2)
		mode = 0;
	namedWindow("处理后", 1);
	namedWindow("处理后直方图", 1);
	imshow("处理后", outputim);
	imshow("处理后直方图", Hisg(outputim));

	iplImage = &(IplImage)outputim;
	imgDst.CopyOf(iplImage, 1);
	imgDst.DrawToHDC(m_pDCDst->m_hDC, &m_rectDst);
}


