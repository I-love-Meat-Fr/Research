
// FXsDlg.h : header file
//

#pragma once
#include <vector>
#include "afxwin.h"
#include "afxcmn.h"
#include "Uers.h"
#include "FingerVeinV3.h"
using std::vector;
#pragma pack(1) 
#define  NameNum  20480
#define  NameLen    20
struct  model_info
{
//	char   name[NameNum][NameLen];//用户名
	vector<BYTE>G_Mode;//模板
//	int    id;//对应ＩＤ
};
#pragma pack()
struct  model_image
{
	vector<BYTE>img;
	bool   bModel;
};

// CFXsDlg dialog
class CFXsDlg : public CDialogEx
{
// Construction
public:
	CFXsDlg(CWnd* pParent = NULL);	// standard constructor
	~CFXsDlg();
// Dialog Data
	enum { IDD = IDD_FXS_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support
public:
	void outPrint(CString strText, BOOL SetColor = FALSE);
	//model_info *  getModelInfo(char *userName);
	void save();
	void Load();
	void    exitThread();
	//void thread_update_pic();
	//unsigned static __stdcall ExecuteThread(void* pArguments);
	afx_msg void OnDeadChar(UINT nChar, UINT nRepCnt, UINT nFlags);
	void setCmd(int pvCmd, int fvCmd)
	{
		WaitForSingleObject(m_Mutex, INFINITE);
		if (pvCmd > -2)
			m_pvCmd = pvCmd;
		if (fvCmd > -2)
			m_fvCmd = fvCmd;

		::ReleaseMutex(m_Mutex);
	}

	int getCmd(bool bpvCmd)
	{
		int cmd;
		WaitForSingleObject(m_Mutex, INFINITE);
		if (bpvCmd)
			cmd = m_pvCmd;
		else
			cmd = m_fvCmd;

		::ReleaseMutex(m_Mutex);
		return cmd;
	}

	void setUp(int bCheckPvUp, int bCheckFvUp)
	{
		WaitForSingleObject(m_MutexUp, INFINITE);
		if (bCheckPvUp > -2)
			m_bCheckPvUp = bCheckPvUp;

		if (bCheckFvUp > -2)
			m_bCheckFvUp = bCheckFvUp;
		::ReleaseMutex(m_MutexUp);
	}

	//afx_msg void OnTimer(UINT_PTR nIDEvent);

	int getUp(bool bCheckPvUp)
	{
		int cmd;
		WaitForSingleObject(m_Mutex, INFINITE);
		if (bCheckPvUp)
			cmd = m_bCheckPvUp;
		else
			cmd = m_bCheckFvUp;

		::ReleaseMutex(m_Mutex);
		return cmd;
	}

	void setWork(bool work)
	{
		WaitForSingleObject(m_MutexFG, INFINITE);
		m_bFingerWork = work;
		::ReleaseMutex(m_MutexFG);
	}

	bool getWork()
	{
		bool work;
		WaitForSingleObject(m_MutexFG, INFINITE);
		work = m_bFingerWork;
		::ReleaseMutex(m_MutexFG);
		return work;
	}
	void updateFeatureCount();
	void updateUserCount();
	void addUser(int index, const char* user, vector<BYTE> &feature);
	configure_info conf_info;
	int openDevIndex;

	int getFlushTime();
	virtual void OnCancel();
	virtual BOOL OnWndMsg(UINT message, WPARAM wParam, LPARAM lParam, LRESULT* pResult);
	void enableIdentifyButtons(bool bEnable = FALSE);
// Implementation
protected:
	HICON m_hIcon;
	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
private:
	void initControls();
	void doOpenDev();
	void doCloseDev();
	void initDevlist(int cnt);
	
	void doFingerIdentify();
	bool getInput1V1Model(CString str1v1ID);
	void relayout();
public:
	cv::VideoCapture video;
	CStatic m_staticuserName;
	CEdit m_edUserName;
	TCHAR  m_userName[NameLen];
	char  m_1v1userName[NameLen];
	CButton m_buildModule;
	CButton m_Button_1V1_Identify;
	CButton m_Button_1VN_Identify;
	CButton m_Button_Identify100;
	CButton m_ButtonOpenDev;
	BOOL   m_bTerminate;
	CRichEditCtrl m_edOut;
	CUers   user;
	HDEVNOTIFY m_hDeviceNotify;
	model_image			  m_modelImage[3];
	//获取名字 

	model_info    m_currentModel;
	vector<vector<BYTE>>  m_vecModellist;
	char  m_strModelPath[MAX_PATH];
	char  m_strAppPath[MAX_PATH];
	char  m_strPhotoPath[MAX_PATH];

	char  m_strErrPhotoPath[MAX_PATH];

	//线程部分
	HANDLE  m_ExecEvent;
	HANDLE  m_EndEvent;
	HANDLE m_Mutex;
	HANDLE m_MutexUp;
	HANDLE m_MutexFG;
	int	 m_bCheckPvUp;
	int  m_bCheckFvUp;
	int  m_pvCmd;
	int  m_fvCmd;
	bool m_bFingerWork;
	int  m_iDeviceCount;

	bool m_bHaveRecordsofBuildModule;
	int  m_iIdentifyType;
	enum{ CMD_OPEN, CMD_GRAB1, CMD_GRAB2, CMD_GRAB3, CMD_MODEL, CMD_IDNETY, CMD_CLOSE };
	HANDLE	m_SampleExecThread;

	CFont m_editFont;
	HANDLE    thread_handle;
	CStatic m_stPreview;
	CComboBox m_cbDevs;

	afx_msg void OnBnClickedButtonOpendev();
	afx_msg void OnBnClickedButtonBuildmodule();
	afx_msg void OnBnClickedButton1v1Identify();
	afx_msg void OnBnClickedButtonIdentify100();
	afx_msg void OnBnClickedButton1vnIdentify();
	CButton m_Button_Quit;
	afx_msg void OnBnClickedButtonQuit();
	CStatic m_static_TotalTIme;
	afx_msg void OnBnClickedButtonQuitidtfy();
	CButton m_buttonQuitIdentify;
	CEdit m_edit1V1ID;
	afx_msg void OnBnClickedButtonRed();
	afx_msg void OnBnClickedButtonGreen();
	afx_msg void OnBnClickedButtonBuzzer();
	afx_msg void OnBnClickedButtonRed2();
	afx_msg void OnBnClickedButtonGreen2();
	afx_msg void OnEnChangeEdit1();
	afx_msg void OnEnChangeEdit2();
	afx_msg void OnBnClickedButtonDel();
	afx_msg void OnBnClickedButtonDelAll();
	afx_msg void OnBnClickedButtonLoad();
	afx_msg void OnBnClickedButtonBuilds();
	afx_msg void OnBnClickedCheckUser();
	afx_msg void OnBnClickedCheckFinger();
	afx_msg void OnBnClickedButtonCfg();
	afx_msg void OnEnChangeEditUsenameid();
	afx_msg void OnEnChangeEdit4();
	afx_msg void OnBnClickedButtonSave();
	CButton m_saveImg;
	afx_msg void OnBnClickedButtonImgTest();
	afx_msg void OnBnClickedButtonImgTest2();
	afx_msg void OnEnChangeRichedit2Allmessages();
};
