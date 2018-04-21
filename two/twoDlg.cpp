
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
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_SLIDER1, m_slider);
	DDX_Control(pDX, IDC_SLIDER2, m_slider2);
	DDX_Control(pDX, IDC_LIST4, m_list);
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
	ON_BN_CLICKED(IDC_BUTTON5, &CtwoDlg::OnBnClickedButton5)
	ON_STN_DBLCLK(IDC_SRC, &CtwoDlg::OnStnDblclickSrc)
	ON_STN_DBLCLK(IDC_DST, &CtwoDlg::OnStnDblclickDst)
	ON_BN_CLICKED(IDC_BUTTON9, &CtwoDlg::OnBnClickedButton9)
	ON_WM_HSCROLL()
	ON_LBN_SELCHANGE(IDC_LIST4, &CtwoDlg::OnLbnSelchangeList4)
	ON_STN_DBLCLK(IDC_HIS, &CtwoDlg::OnStnDblclickHis)
	ON_WM_CTLCOLOR()
	ON_WM_SIZE()
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
	GetDlgItem(IDC_EDIT1)->SetWindowText(_T("0")); //设置编辑框默认显示的内容 
	GetDlgItem(IDC_EDIT2)->SetWindowText(_T("0")); //设置编辑框默认显示的内容 
	GetDlgItem(IDC_EDIT3)->SetWindowText(_T("0")); //设置编辑框默认显示的内容 
	GetDlgItem(IDC_EDIT4)->SetWindowText(_T("255")); //设置编辑框默认显示的内容 

	
	GetDlgItem(IDC_STATIC_1)->SetWindowTextW(_T("无"));    //设置静态文本框
	GetDlgItem(IDC_STATIC_2)->SetWindowTextW(_T("无"));
	GetDlgItem(IDC_STATIC_3)->SetWindowTextW(_T("-"));


	//m_pWndSrc = GetDlgItem(IDC_SRC);
	//m_pWndDst = GetDlgItem(IDC_DST);
	m_pDCSrc = GetDlgItem(IDC_SRC)->GetDC();                    //设置方框图片区域参数
	m_pDCDst = GetDlgItem(IDC_DST)->GetDC();
	m_pDCHis = GetDlgItem(IDC_HIS)->GetDC();
	GetDlgItem(IDC_DST)->GetClientRect(&m_rectSrc);
	GetDlgItem(IDC_DST)->GetClientRect(&m_rectDst);
	GetDlgItem(IDC_HIS)->GetClientRect(&m_rectHis);

	//滚动条设置
	m_slider.SetRange(0, 255);//设置滑动范围为0到255
	m_slider.SetTicFreq(1);//每1个单位画一刻度
	m_slider.SetPos(0);//设置滑块初始位置为1
	m_slider2.SetRange(0, 255);//设置滑动范围为0到255
	m_slider2.SetTicFreq(1);//每1个单位画一刻度
	m_slider2.SetPos(255);//设置滑块初始位置为1
	m_slider.ShowWindow(FALSE);   //隐藏
	m_slider2.ShowWindow(FALSE);

	//隐藏控件
	GetDlgItem(IDC_EDIT1)->ShowWindow(FALSE);    //隐藏
	GetDlgItem(IDC_EDIT2)->ShowWindow(FALSE);    //隐藏
	GetDlgItem(IDC_EDIT3)->ShowWindow(FALSE);    //隐藏
	GetDlgItem(IDC_EDIT4)->ShowWindow(FALSE);    //隐藏
	GetDlgItem(IDC_STATIC_1)->ShowWindow(FALSE); //隐藏
	GetDlgItem(IDC_STATIC_2)->ShowWindow(FALSE); //隐藏
	GetDlgItem(IDC_STATIC_3)->ShowWindow(FALSE); //隐藏

	//列表框设置
	m_list.AddString(_T("椒盐噪声"));
	m_list.AddString(_T("平滑处理"));
	m_list.AddString(_T("中值滤波"));
	m_list.AddString(_T("均衡化归一化"));
	m_list.AddString(_T("灰度级分层"));
	m_list.AddString(_T("锐化"));
	m_list.AddString(_T("FFT"));
	m_list.AddString(_T("伽玛变换"));
	m_list.AddString(_T("灰度级分层(二值化)"));
	m_list.AddString(_T("理想低通滤波器"));
	m_list.AddString(_T("巴特沃斯低通滤波器"));
	m_list.AddString(_T("高斯低通滤波器"));
	m_list.SetCurSel(0);
	

	//GDI+初始化
	Gdiplus::GdiplusStartup(&m_uGdiplusToken, &m_GdiplusStarupInput, nullptr);
	//背景图设置
	//m_img = Image::FromFile(_T("C:\\Users\\Cashion\\Desktop\\pconline1484901626457\\Nier-2b.bmp"));  //加载图片
	//ImageFromIDResource(IDB_BITMAP4,_T("bmp"),(Image * &)m_img);



	// 获取对话框初始大小    
	GetClientRect(&m_rect);  //获取对话框的大小
	old.x = m_rect.right - m_rect.left;
	old.y = m_rect.bottom - m_rect.top;


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
		CDialogEx::OnSysCommand(nID, lParam);
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
		/*CDialog::OnPaint();*/
		CPaintDC   dc(this);
		//CRect   rect;
		//GetClientRect(&rect);    //获取对话框长宽      
		//CDC   dcBmp;             //定义并创建一个内存设备环境
		//dcBmp.CreateCompatibleDC(&dc);             //创建兼容性DC
		//CBitmap*   bmpBackground=nullptr;
		//bmpBackground->LoadBitmap(IDB_BITMAP2);    //载入资源中图片
		//BITMAP   m_bitmap;                         //图片变量               
		//bmpBackground.GetBitmap(&m_bitmap);       //将图片载入位图中
		//										  //将位图选入临时内存设备环境
		//CBitmap  *pbmpOld = dcBmp.SelectObject(&bmpBackground);
		////调用函数显示图片StretchBlt显示形状可变
		//dc.StretchBlt(0, 0, rect.Width(), rect.Height(), &dcBmp, 0, 0, m_bitmap.bmWidth, m_bitmap.bmHeight, SRCCOPY);
		CBitmap bmp;
		bmp.LoadBitmap(IDB_BITMAP2);
		HBITMAP hbi = (HBITMAP)bmp;
		Bitmap* bitmap = Bitmap::FromHBITMAP(hbi,NULL);
		//另一种方法
		CRect rect = { 0 };
		GetClientRect(&rect);   //获取客户区大小
		Graphics g(dc);
		g.DrawImage(bitmap, 0, 0, rect.Width(), rect.Height());

		// 用于不改变界面内容
		//if (ON_INPUT)
		//{
		//	iplImage = &(IplImage)im;
		//	imgSrc.CopyOf(iplImage, 1);
		//	imgSrc.DrawToHDC(m_pDCSrc->m_hDC, &m_rectSrc);         //将图片放置图片框内
		//	Hisim = Hisg(im);
		//	iplImage = &(IplImage)Hisim;
		//	imgHis.CopyOf(iplImage, 1);
		//	imgHis.DrawToHDC(m_pDCHis->m_hDC, &m_rectHis);         //将直方图放置图片框内

		//}

		//if (ON_OUTPUT)
		//{

		//	iplImage = &(IplImage)outputim;      //框内显示图片
		//	imgDst.CopyOf(iplImage, 1);
		//	imgDst.DrawToHDC(m_pDCDst->m_hDC, &m_rectDst);
		//	Hisim = Hisg(outputim);                     //框内显示直方图
		//	iplImage = &(IplImage)Hisim;
		//	imgHis.CopyOf(iplImage, 1);
		//	imgHis.DrawToHDC(m_pDCHis->m_hDC, &m_rectHis);         //将直方图放置图片框内
		//}
		GetDlgItem(IDC_DST)->GetClientRect(&m_rectSrc);
		GetDlgItem(IDC_DST)->GetClientRect(&m_rectDst);
		GetDlgItem(IDC_HIS)->GetClientRect(&m_rectHis);
	


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

                                  以下为消息处理函数
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
		char * _src = StrToChar(strOpenFilePath);


	
		im = imread(_src, state);   //读取原图

		iplImage = &(IplImage)im;
		imgSrc.CopyOf(iplImage, 1);
		imgSrc.DrawToHDC(m_pDCSrc->m_hDC, &m_rectSrc);         //将图片放置图片框内

		Hisim = Hisg(im);
		iplImage = &(IplImage)Hisim;
		imgHis.CopyOf(iplImage, 1);
		imgHis.DrawToHDC(m_pDCHis->m_hDC, &m_rectHis);         //将直方图放置图片框内

		outputim = im.clone();//原图为im

		OnLbnSelchangeList4();

		ON_INPUT = 1;



	}
}

/***********************************************************************************************
                                      执行处理事件
*************************************************************************************************/

void CtwoDlg::OnBnClickedButton2()
{

	if (!ON_INPUT)
		return;
	ON_OUTPUT = 1;
	// TODO: 在此添加控件通知处理程序代码
	CString   strText;
	int Mode = 0;
	Mode = m_list.GetCurSel();
	m_list.GetText(m_list.GetCurSel(), strText);

	if(strText== "椒盐噪声")    //椒盐噪声实现
	{
		int n = 0;
		GetDlgItem(IDC_EDIT1)->GetWindowText(Gedit1);    //获取编辑框内容
		n = _ttoi(Gedit1);               //Unicode转int

		if (!(n > 0)) {
			MessageBox(_T("必须输入大于0的数"));
			return;
		}
		inputim = outputim.clone();
		outputim = addSaltNoise(inputim, n);    //加椒盐噪声处理
	}
	else if(strText == "平滑处理")       //平滑处理实现
	{
		int n = 0;
		GetDlgItem(IDC_EDIT1)->GetWindowText(Gedit1);    //获取编辑框内容
		n = _ttoi(Gedit1);               //CSstring转int

		if (!(n % 2) || n < 0 || n>100) {
			MessageBox(_T("必须输入合理范围的奇数"));
			return;
		}
		inputim = outputim.clone(); ;                               //平滑（均值）滤波处理
		blur(inputim, outputim, Size(n, n));
	}
	else if (strText == "中值滤波")        //中值滤波实现
	{
		int n = 0;
		GetDlgItem(IDC_EDIT1)->GetWindowText(Gedit1);    //获取编辑框内容
		n = _ttoi(Gedit1);               //Unicode转int

		if (!(n % 2) || n < 0 || n>100) {
			MessageBox(_T("必须输入合理范围的奇数"));
			return;
		}
		inputim = outputim.clone();                                         //中值滤波处理
		medianBlur(inputim, outputim, n);
		
	}
	else if (strText == "均衡化归一化")        //归一化实现
	{
		int Range[2] = { 0 };
		GetDlgItem(IDC_EDIT3)->GetWindowText(Gedit3);    //获取编辑框内容
		GetDlgItem(IDC_EDIT4)->GetWindowText(Gedit4);    //获取编辑框内容
		Range[0] = _ttoi(Gedit3);               //Unicode转int
		Range[1] = _ttoi(Gedit4);               //Unicode转int

		inputim = outputim.clone();
		outputim = Eqlz(inputim, Range);
		//equalizeHist(inputim, outputim);
		
	}
	else if (strText == "灰度级分层")        //灰度分层实现
	{
		int Range[2] = { 0 };
		GetDlgItem(IDC_EDIT3)->GetWindowText(Gedit3);    //获取编辑框内容
		GetDlgItem(IDC_EDIT4)->GetWindowText(Gedit4);    //获取编辑框内容
		Range[0] = _ttoi(Gedit3);               //Unicode转int
		Range[1] = _ttoi(Gedit4);               //Unicode转int
		inputim = outputim.clone();
		outputim = Glvls(inputim, Range,1);
	}
	else if (strText == "锐化")       //锐化实现
	{
		inputim = outputim.clone();
		Mat kern = (Mat_<char>(3, 3) << -1, -1, -1,
			-1, 9, -1,
			-1, -1, -1);
		filter2D(inputim, outputim, -1, kern, Point(-1, -1), 0, BORDER_DEFAULT);
	}
	else if (strText == "FFT")       //DFT实现
	{
		Dft_IDft(outputim);
	}
	else if (strText == "伽玛变换")        //ganma变换
	{
		double n = 0;
		GetDlgItem(IDC_EDIT1)->GetWindowText(Gedit1);    //获取编辑框内容
		n = _wtof(Gedit1);               //Unicode转int

		if (n > 100) {
			MessageBox(_T("数值太大"));
			return;
		}
	    inputim = outputim.clone();                                         
	    outputim= Gamma_ma(inputim,n);
	}
	else if (strText == "灰度级分层(二值化)")       //灰度分层实现
	{
		int Range[2] = { 0 };
		GetDlgItem(IDC_EDIT3)->GetWindowText(Gedit3);    //获取编辑框内容
		GetDlgItem(IDC_EDIT4)->GetWindowText(Gedit4);    //获取编辑框内容
		Range[0] = _ttoi(Gedit3);               //Unicode转int
		Range[1] = _ttoi(Gedit4);               //Unicode转int
		inputim = outputim.clone();
		outputim = Glvls(inputim, Range,0);
	}
	else if (strText == "理想低通滤波器")       //理想低通滤波器
	{
		int n = 0;
		GetDlgItem(IDC_EDIT1)->GetWindowText(Gedit1);    //获取编辑框内容
		n = _ttoi(Gedit1);               //Unicode转int
		inputim = outputim.clone();
		outputim=ideal_Low_Pass_Filter(inputim,n);
	}
	else if (strText == "巴特沃斯低通滤波器")       //巴特沃斯低通滤波器
	{
		int n = 0;
		int D0 = 0;
		GetDlgItem(IDC_EDIT1)->GetWindowText(Gedit1);    //获取编辑框内容
		GetDlgItem(IDC_EDIT2)->GetWindowText(Gedit2);    //获取编辑框内容
		D0 = _ttoi(Gedit1);               //Unicode转int
		n = _ttoi(Gedit2);
		inputim = outputim.clone();
		outputim = Butterworth_Low_Paass_Filter(inputim, D0, n);
	}
	else if (strText == "高斯低通滤波器")       //高斯低通滤波器
	{
		int n = 0;
		GetDlgItem(IDC_EDIT1)->GetWindowText(Gedit1);    //获取编辑框内容
		n = _ttoi(Gedit1);
		inputim = outputim.clone();
		outputim = Gauss_Low_Paass_Filter(inputim,n);
	}


	iplImage = &(IplImage)outputim;      //框内显示图片
	imgDst.CopyOf(iplImage, 1);
	imgDst.DrawToHDC(m_pDCDst->m_hDC, &m_rectDst);


	Hisim = Hisg(outputim);                     //框内显示直方图
	iplImage = &(IplImage)Hisim;
	imgHis.CopyOf(iplImage, 1);
	imgHis.DrawToHDC(m_pDCHis->m_hDC, &m_rectHis);         //将直方图放置图片框内

	
}


/***********************************************************************************************
                                    初始化图片
*************************************************************************************************/

void CtwoDlg::OnBnClickedButton5()
{
	if (!ON_INPUT)
		return;
	// TODO: 在此添加控件通知处理程序代码


	inputim = im.clone();
	outputim = im.clone();

	iplImage = &(IplImage)im;
	imgDst.CopyOf(iplImage, 1);
	imgDst.DrawToHDC(m_pDCDst->m_hDC, &m_rectDst);

	Hisim = Hisg(outputim);                     //框内显示直方图
	iplImage = &(IplImage)Hisim;
	imgHis.CopyOf(iplImage, 1);
	imgHis.DrawToHDC(m_pDCHis->m_hDC, &m_rectHis);         //将直方图放置图片框内

	
}



void CtwoDlg::OnStnDblclickSrc()
{
	if (!ON_INPUT)
		return;
	// TODO: 在此添加控件通知处理程序代码
	namedWindow("原图", 1);
	imshow("原图", im);           
}


void CtwoDlg::OnStnDblclickDst()
{
	if (!ON_INPUT)
		return;
	// TODO: 在此添加控件通知处理程序代码
	namedWindow("处理后", 1);
	imshow("处理后", outputim);

}

void CtwoDlg::OnStnDblclickHis()
{
	if (!ON_INPUT)
		return;
	// TODO: 在此添加控件通知处理程序代码
	namedWindow("直方图", 1);
	imshow("直方图", Hisim);

}




/***********************************************************************************************
                                      保存图片
*************************************************************************************************/

void CtwoDlg::OnBnClickedButton9()
{
	if (!ON_INPUT)
		return;
	// TODO: 在此添加控件通知处理程序代码
	TCHAR szFilter[] = _T("图像文件(*.bmp,*.jpg)|*.bmp;*.jpg|视频文件(*.avi)|*.avi|所有文件(*.*)|*.*||");   //设置过滤格式
	CFileDialog  dlgFile(false, _T("jpg"), _T("Oject"), OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, szFilter, NULL);    //设置窗口属性
	if (IDOK == dlgFile.DoModal())           //弹出文件浏览窗
	{

		CString strSaveFilePath = dlgFile.GetPathName();

		//	//将CString类型的文件名转化为char*类型
		char * _src = StrToChar(strSaveFilePath);


		imwrite(_src, outputim);  //保存文件
	}


}


/***********************************************************************************************
                                         二维FFT
*************************************************************************************************/
//void CtwoDlg::OnBnClickedButton10()
//{
//	if (!ON_INPUT)
//		return;
//	// TODO: 在此添加控件通知处理程序代码
//
//
//	// Read as grayscale image
//	Mat image = outputim.clone();
//
//
//	Mat padded;
//	int m = getOptimalDFTSize(image.rows);  // Return size of 2^x that suite for FFT
//	int n = getOptimalDFTSize(image.cols);
//	// Padding 0, result is @padded
//	copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));
//
//	// Create planes to storage REAL part and IMAGE part, IMAGE part init are 0
//	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
//	Mat complexI;
//	merge(planes, 2, complexI);
//
//	dft(complexI, complexI);
//
//	// compute the magnitude and switch to logarithmic scale
//	split(complexI, planes);
//	magnitude(planes[0], planes[0], planes[1]);
//	Mat magI = planes[0];
//
//	// => log(1+sqrt(Re(DFT(I))^2+Im(DFT(I))^2))
//	magI += Scalar::all(1);
//	log(magI, magI);
//
//	// crop the spectrum
//	magI = magI(Rect(0, 0, magI.cols & (-2), magI.rows & (-2)));
//	Mat _magI = magI.clone();
//	normalize(_magI, _magI, 0, 1, CV_MINMAX);
//
//	// rearrange the quadrants of Fourier image so that the origin is at the image center
//	int cx = magI.cols / 2;
//	int cy = magI.rows / 2;
//
//	Mat q0(magI, Rect(0, 0, cx, cy));    // Top-Left
//	Mat q1(magI, Rect(cx, 0, cx, cy));   // Top-Right
//	Mat q2(magI, Rect(0, cy, cx, cy));   // Bottom-Left
//	Mat q3(magI, Rect(cx, cy, cx, cy));  // Bottom-Right
//
//										 // exchange Top-Left and Bottom-Right
//	Mat tmp;
//	q0.copyTo(tmp);
//	q3.copyTo(q0);
//	tmp.copyTo(q3);
//
//	// exchange Top-Right and Bottom-Left
//	q1.copyTo(tmp);
//	q2.copyTo(q1);
//	tmp.copyTo(q2);
//
//	normalize(magI, magI, 0, 1, CV_MINMAX);
//
//
//
//	imshow("Spectrum magnitude before shift frequency", _magI);
//	imshow("Spectrum magnitude after shift frequency", magI);
//	
//}


/***********************************************************************************************
                               滚动条事件
*************************************************************************************************/
void CtwoDlg::OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar)
{
	UINT ID=GetWindowLong(pScrollBar->GetSafeHwnd(), GWL_ID);
	// TODO: 在此添加消息处理程序代码和/或调用默认值

	CString strtemp;
	int i;
	switch (ID)
	{
	case IDC_SLIDER1:
		i = m_slider.GetPos();
		strtemp.Format(_T("%d"), i);
		GetDlgItem(IDC_EDIT3)->SetWindowText(strtemp); //设置编辑框默认显示的内容 
		break;
	case IDC_SLIDER2:
		i = m_slider2.GetPos();
		strtemp.Format(_T("%d"), i);
		GetDlgItem(IDC_EDIT4)->SetWindowText(strtemp); //设置编辑框默认显示的内容 
		break;
	}

	

	CDialogEx::OnHScroll(nSBCode, nPos, pScrollBar);
}


/***********************************************************************************************
                                     列表框单击事件
*************************************************************************************************/

void CtwoDlg::OnLbnSelchangeList4()
{
  //隐藏控件
	GetDlgItem(IDC_EDIT1)->ShowWindow(FALSE);    //隐藏
	GetDlgItem(IDC_EDIT2)->ShowWindow(FALSE);    //隐藏
	GetDlgItem(IDC_EDIT3)->ShowWindow(FALSE);    //隐藏
	GetDlgItem(IDC_EDIT4)->ShowWindow(FALSE);    //隐藏
	GetDlgItem(IDC_STATIC_1)->ShowWindow(FALSE); //隐藏
	GetDlgItem(IDC_STATIC_2)->ShowWindow(FALSE); //隐藏
	GetDlgItem(IDC_STATIC_3)->ShowWindow(FALSE); //隐藏
	GetDlgItem(IDC_SLIDER1)->ShowWindow(FALSE); //隐藏
	GetDlgItem(IDC_SLIDER2)->ShowWindow(FALSE); //隐藏
	// TODO: 在此添加控件通知处理程序代码
	CString   strText;
	int Mode = 0;
	Mode = m_list.GetCurSel();
	m_list.GetText(m_list.GetCurSel(), strText);
	if (strText == "椒盐噪声")    //椒盐噪声实现
	{	
		GetDlgItem(IDC_EDIT1)->ShowWindow(TRUE);    //显示
		GetDlgItem(IDC_STATIC_1)->ShowWindow(TRUE);    //显示
		GetDlgItem(IDC_EDIT1)->SetWindowText(_T("8000")); //设置编辑框默认显示的内容 
		GetDlgItem(IDC_STATIC_1)->SetWindowTextW(_T("噪点数"));    //设置静态文本框
	}
	else if (strText == "平滑处理")       //平滑处理实现
	{
		GetDlgItem(IDC_EDIT1)->ShowWindow(TRUE);    //显示
		GetDlgItem(IDC_STATIC_1)->ShowWindow(TRUE);    //显示
		GetDlgItem(IDC_EDIT1)->SetWindowText(_T("3")); //设置编辑框默认显示的内容 
		GetDlgItem(IDC_STATIC_1)->SetWindowTextW(_T("领域大小"));    //设置静态文本框
	}
	else if (strText == "中值滤波")        //中值滤波实现
	{
		GetDlgItem(IDC_EDIT1)->ShowWindow(TRUE);    //显示
		GetDlgItem(IDC_STATIC_1)->ShowWindow(TRUE);    //显示
		GetDlgItem(IDC_EDIT1)->SetWindowText(_T("3")); //设置编辑框默认显示的内容 
		GetDlgItem(IDC_STATIC_1)->SetWindowTextW(_T("领域大小"));    //设置静态文本框
	}
	else if (strText == "均衡化归一化")        //归一化实现
	{
		GetDlgItem(IDC_EDIT1)->SetWindowText(_T("0")); //设置编辑框默认显示的内容 
		GetDlgItem(IDC_STATIC_1)->SetWindowTextW(_T("无"));    //设置静态文本框
		m_slider.ShowWindow(TRUE);   //滚动条显示
		m_slider2.ShowWindow(TRUE);
		GetDlgItem(IDC_EDIT3)->ShowWindow(TRUE);    //显示
		GetDlgItem(IDC_EDIT4)->ShowWindow(TRUE);    //显示
		GetDlgItem(IDC_STATIC_3)->ShowWindow(TRUE); //显示
	}
	else if (strText == "灰度级分层")        //灰度分层实现
	{
		m_slider.ShowWindow(TRUE);   //滚动条显示
		m_slider2.ShowWindow(TRUE);  //滚动条显示
		GetDlgItem(IDC_EDIT3)->ShowWindow(TRUE);    //显示
		GetDlgItem(IDC_EDIT4)->ShowWindow(TRUE);    //显示
		GetDlgItem(IDC_STATIC_3)->ShowWindow(TRUE); //显示
	}
	else if (strText == "锐化")       //锐化实现
	{

	}
	else if (strText == "FFT")       //DFT实现
	{

	}
	else if (strText == "伽玛变换")        //ganma变换      
	{
		GetDlgItem(IDC_EDIT1)->ShowWindow(TRUE);    //显示
		GetDlgItem(IDC_STATIC_1)->ShowWindow(TRUE);    //显示
		GetDlgItem(IDC_EDIT1)->SetWindowText(_T("0.20")); //设置编辑框默认显示的内容 
		GetDlgItem(IDC_STATIC_1)->SetWindowTextW(_T("γ值"));    //设置静态文本框
	}
	else if (strText == "灰度级分层(二值化)")       //灰度分层实现
	{
		m_slider.ShowWindow(TRUE);   //滚动条显示
		m_slider2.ShowWindow(TRUE);  //滚动条显示
		GetDlgItem(IDC_EDIT3)->ShowWindow(TRUE);    //显示
		GetDlgItem(IDC_EDIT4)->ShowWindow(TRUE);    //显示
		GetDlgItem(IDC_STATIC_3)->ShowWindow(TRUE); //显示
	}
	else if (strText == "理想低通滤波器")       //理想低通滤波器
	{
		GetDlgItem(IDC_EDIT1)->ShowWindow(TRUE);    //显示
		GetDlgItem(IDC_STATIC_1)->ShowWindow(TRUE);    //显示
		GetDlgItem(IDC_EDIT1)->SetWindowText(_T("60")); //设置编辑框默认显示的内容 
		GetDlgItem(IDC_STATIC_1)->SetWindowTextW(_T("阈值"));    //设置静态文本框
	}
	else if (strText == "巴特沃斯低通滤波器")       //巴特沃斯低通滤波器
	{
		GetDlgItem(IDC_EDIT1)->ShowWindow(TRUE);    //显示
		GetDlgItem(IDC_STATIC_1)->ShowWindow(TRUE);    //显示
		GetDlgItem(IDC_EDIT2)->ShowWindow(TRUE);    //显示
		GetDlgItem(IDC_STATIC_2)->ShowWindow(TRUE);    //显示
		GetDlgItem(IDC_EDIT1)->SetWindowText(_T("100")); //设置编辑框默认显示的内容 
		GetDlgItem(IDC_EDIT2)->SetWindowText(_T("3")); //设置编辑框默认显示的内容 
		GetDlgItem(IDC_STATIC_1)->SetWindowTextW(_T("阈值"));    //设置静态文本框
		GetDlgItem(IDC_STATIC_2)->SetWindowTextW(_T("阶数"));    //设置静态文本框
	}
	else if (strText == "高斯低通滤波器")       //高斯低通滤波器
	{
		GetDlgItem(IDC_EDIT1)->ShowWindow(TRUE);    //显示
		GetDlgItem(IDC_STATIC_1)->ShowWindow(TRUE);    //显示
		GetDlgItem(IDC_EDIT1)->SetWindowText(_T("40")); //设置编辑框默认显示的内容 
		GetDlgItem(IDC_STATIC_1)->SetWindowTextW(_T("σ"));    //设置静态文本框

	}



}

/***********************************************************************************************
                                  控件透明化
*************************************************************************************************/
HBRUSH CtwoDlg::OnCtlColor(CDC* pDC, CWnd* pWnd, UINT nCtlColor)
{
	HBRUSH hbr = CDialogEx::OnCtlColor(pDC, pWnd, nCtlColor);



	// 如果不做判断的话，全部静态文本背景都是透明的，做了判断就指定ID其中一个变成透明
	if (  pWnd->GetDlgCtrlID() == (IDC_STATIC_1)
		||pWnd->GetDlgCtrlID() == (IDC_STATIC_2)
		||pWnd->GetDlgCtrlID() == (IDC_STATIC_3)
		)
	{
		//MessageBox(_T("static text"));
		pDC->SetBkMode(TRANSPARENT);
		pDC->SetTextColor(RGB(0, 225, 225));
		return HBRUSH(GetStockObject(HOLLOW_BRUSH));
	}
	// TODO:  Return a different brush if the default is not desired


	return hbr;
}

//BOOL CtwoDlg::ImageFromIDResource(UINT nID, LPCTSTR sTR, Image * & pImg)
//{
//	HINSTANCE hInst = AfxGetResourceHandle();
//	HRSRC hRsrc = ::FindResource(hInst, MAKEINTRESOURCE(nID), sTR); // type  
//	if (!hRsrc)
//		return FALSE;
//	 load resource into memory  
//	DWORD len = SizeofResource(hInst, hRsrc);
//	BYTE* lpRsrc = (BYTE*)LoadResource(hInst, hRsrc);
//	if (!lpRsrc)
//		return FALSE;
//	 Allocate global memory on which to create stream  
//	HGLOBAL m_hMem = GlobalAlloc(GMEM_FIXED, len);
//	BYTE* pmem = (BYTE*)GlobalLock(m_hMem);
//	memcpy(pmem, lpRsrc, len);
//	IStream* pstm;
//	CreateStreamOnHGlobal(m_hMem, FALSE, &pstm);
//	 load from stream  
//	pImg = Gdiplus::Image::FromStream(pstm);
//	 free/release stuff  
//	GlobalUnlock(m_hMem);
//	pstm->Release();
//	FreeResource(lpRsrc);
//	return TRUE;
//}



//当启动和窗口大小改变时执行
void CtwoDlg::OnSize(UINT nType, int cx, int cy)
{
	CDialogEx::OnSize(nType, cx, cy);

	// TODO: 在此处添加消息处理程序代码
	//当窗口大小改变的时候，使客户区无效
	if (nType != SIZE_MINIMIZED)  //判断窗口是不是最小化了，因为窗口最小化之后 ，窗口的长和宽会变成0，当前一次变化的时就会出现除以0的错误操作
	{
		ReSize();
	}

	Invalidate(FALSE);
}

//改变控件大小
void CtwoDlg::ReSize(void)
{
	double fsp[2];
	POINT Newp; //获取现在对话框的大小  
	CRect recta;
	GetClientRect(&recta);     //取客户区大小    
	Newp.x = recta.right - recta.left;
	Newp.y = recta.bottom - recta.top;
	fsp[0] = (double)Newp.x / old.x;
	fsp[1] = (double)Newp.y / old.y;
	CRect Rect;
	int woc;
	CPoint OldTLPoint, TLPoint; //左上角  
	CPoint OldBRPoint, BRPoint; //右下角  
	HWND  hwndChild = ::GetWindow(m_hWnd, GW_CHILD);  //列出所有控件    
	while (hwndChild) {
		woc = ::GetDlgCtrlID(hwndChild);//取得ID  
		GetDlgItem(woc)->GetWindowRect(Rect);
		ScreenToClient(Rect);
		OldTLPoint = Rect.TopLeft();
		TLPoint.x = long(OldTLPoint.x*fsp[0]);
		TLPoint.y = long(OldTLPoint.y*fsp[1]);
		OldBRPoint = Rect.BottomRight();
		BRPoint.x = long(OldBRPoint.x *fsp[0]);
		BRPoint.y = long(OldBRPoint.y *fsp[1]);
		Rect.SetRect(TLPoint, BRPoint);
		GetDlgItem(woc)->MoveWindow(Rect, TRUE);
		hwndChild = ::GetWindow(hwndChild, GW_HWNDNEXT);
	}
	old = Newp;
}
