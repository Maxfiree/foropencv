
// threeDlg.cpp: 实现文件
//

#include "stdafx.h"
#include "three.h"
#include "threeDlg.h"
#include "afxdialogex.h"
#include<math.h>
#include"Rec.h"

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


// CthreeDlg 对话框



CthreeDlg::CthreeDlg(CWnd* pParent /*=NULL*/)
	: CDialog(IDD_THREE_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CthreeDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CthreeDlg, CDialog)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_WM_LBUTTONDOWN()
	ON_WM_MOUSEMOVE()
	ON_WM_LBUTTONUP()
	ON_BN_CLICKED(IDC_BUTTON1, &CthreeDlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &CthreeDlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON3, &CthreeDlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON4, &CthreeDlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON5, &CthreeDlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON6, &CthreeDlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON7, &CthreeDlg::OnBnClickedButton4)
	ON_WM_TIMER()
END_MESSAGE_MAP()


// CthreeDlg 消息处理程序

BOOL CthreeDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

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

	GetDlgItem(IDC_STATIC_1)->SetWindowTextW(_T("保存路径"));    //设置静态文本框
	GetDlgItem(IDC_STATIC_2)->SetWindowTextW(_T("文件名"));

	GetDlgItem(IDC_EDIT1)->SetWindowText(_T("C:\\Users\\Cashion\\Desktop\\something\\数字识别样本\\0\\")); //设置编辑框默认显示的内容 
	GetDlgItem(IDC_EDIT2)->SetWindowText(_T("0.jpg")); //设置编辑框默认显示的内容 

	CRect IniRect;                                                 //调整方框大小
	GetDlgItem(IDC_DW)->GetWindowRect(&IniRect);                   
	ScreenToClient(&IniRect);
	GetDlgItem(IDC_DW)->MoveWindow(IniRect.left, IniRect.top,205, 205, true); //调整方框大小



	m_pDCDW = GetDlgItem(IDC_DW)->GetDC();                    //设置方框图片区域参数
	GetDlgItem(IDC_DW)->GetClientRect(&m_rectDW);
	im = CreatW(m_rectDW.bottom, m_rectDW.right);  //创建白框

	SetTimer(1, 100, NULL);

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CthreeDlg::OnSysCommand(UINT nID, LPARAM lParam)
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

void CthreeDlg::OnPaint()
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
HCURSOR CthreeDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


/*************************************************************************************************
**************************************************************************************************

                                  以下为消息处理函数
**************************************************************************************************
*************************************************************************************************/



/***********************************************************************************************
                              按下鼠标
*************************************************************************************************/


void CthreeDlg::OnLButtonDown(UINT nFlags, CPoint point)
{
	// TODO: 在此添加消息处理程序代码和/或调用默认值

	CDialog::OnLButtonDown(nFlags, point);


	GetCursorPos(&ptCursor);//获取鼠标位置
	GetDlgItem(IDC_DW)->GetWindowRect(&rc);
	if (rc.PtInRect(ptCursor))//如果鼠标在picture区域抬起则开始绘制图片
	{
		mouse = 1;

	}
}

/***********************************************************************************************
                             移动鼠标
*************************************************************************************************/

void CthreeDlg::OnMouseMove(UINT nFlags, CPoint point)
{
	// TODO: 在此添加消息处理程序代码和/或调用默认值

	CDialog::OnMouseMove(nFlags, point);



	GetCursorPos(&ptCursor);//获取鼠标位置
	GetDlgItem(IDC_DW)->GetWindowRect(&rc);  //获取方框大小
	if (rc.PtInRect(ptCursor))//如果鼠标在picture区域抬起则开始绘制图片
	{

		if (mouse == 1) 
		{


			oprc = rc;

			ScreenToClient(&oprc);
			insidePoint.x = point.x - oprc.left;    //当前相对客户区的x坐标减去static的左侧位置即为鼠标指针相对static的坐标
			insidePoint.y = point.y - oprc.top;    //同上

			PaintW(&im, insidePoint.x, insidePoint.y);

			iplImage = &(IplImage)im;
			imgDW.CopyOf(iplImage, 1);
			imgDW.DrawToHDC(m_pDCDW->m_hDC, &m_rectDW);         //将图片放置图片框内


		}
	}
}


/***********************************************************************************************
                       松开鼠标
*************************************************************************************************/
void CthreeDlg::OnLButtonUp(UINT nFlags, CPoint point)
{
	// TODO: 在此添加消息处理程序代码和/或调用默认值

	CDialog::OnLButtonUp(nFlags, point);
	mouse = 0;
}

/***********************************************************************************************
                            清除
*************************************************************************************************/

void CthreeDlg::OnBnClickedButton1()
{
	
	// TODO: 在此添加控件通知处理程序代码
	im = CreatW(m_rectDW.bottom, m_rectDW.right);  //创建白框
	iplImage = &(IplImage)im;
	imgDW.CopyOf(iplImage, 1);
	imgDW.DrawToHDC(m_pDCDW->m_hDC, &m_rectDW);         //将图片放置图片框内

}

/***********************************************************************************************
                   保存图片
*************************************************************************************************/
void CthreeDlg::OnBnClickedButton2()
{
	// TODO: 在此添加控件通知处理程序代码


	CString _Numb;

	GetDlgItem(IDC_EDIT1)->GetWindowText(_PAT);    //获取编辑框内容
	GetDlgItem(IDC_EDIT2)->GetWindowText(_FileName);    //获取编辑框内容
	imwrite(StrToChar(_PAT + _FileName), im);
	n++;
	_Numb.Format(_T("%d"), n);

	GetDlgItem(IDC_EDIT2)->SetWindowText(_Numb+_T(".jpg")); //设置编辑框默认显示的内容 

	//清除
	im = CreatW(m_rectDW.bottom, m_rectDW.right);  //创建白框
	iplImage = &(IplImage)im;
	imgDW.CopyOf(iplImage, 1);
	imgDW.DrawToHDC(m_pDCDW->m_hDC, &m_rectDW);         //将图片放置图片框内


}

/***********************************************************************************************
            初始化文件名
*************************************************************************************************/

void CthreeDlg::OnBnClickedButton3()
{
	// TODO: 在此添加控件通知处理程序代码
	n = 0;
	GetDlgItem(IDC_EDIT2)->SetWindowText(_T("0.jpg")); //设置编辑框默认显示的内容 
}

/***********************************************************************************************
                            识别
*************************************************************************************************/

void CthreeDlg::OnBnClickedButton4()
{
	UINT ID = LOWORD(GetCurrentMessage()->wParam);  //获取按键ID
	// TODO: 在此添加控件通知处理程序代码


	int TheN = 0;
	switch (ID)
	{
	case IDC_BUTTON4:
		TheN = Alg1(im, Fea);
		break;
	case IDC_BUTTON5:
		TheN = Alg2(im, Fea);
		break;
	case IDC_BUTTON6:
		TheN = Alg3(im, Fea);
		break;
	case IDC_BUTTON7:
		TheN = Alg4(im, Fea);
		break;
	}
	CString Res("");
	Res.Format(_T("%d"), TheN);

	MessageBox(_T("你的数字是")+Res,_T("识别结果"));


}


/***********************************************************************************************
                            初始化
*************************************************************************************************/
void CthreeDlg::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: 在此添加消息处理程序代码和/或调用默认值

	CDialog::OnTimer(nIDEvent);

	im = CreatW(m_rectDW.bottom, m_rectDW.right);  //创建白框
	iplImage = &(IplImage)im;
	imgDW.CopyOf(iplImage, 1);
	imgDW.DrawToHDC(m_pDCDW->m_hDC, &m_rectDW);         //将图片放置图片框内

	ExFea(Fea);         //样本特征采集
	KillTimer(1);
}
