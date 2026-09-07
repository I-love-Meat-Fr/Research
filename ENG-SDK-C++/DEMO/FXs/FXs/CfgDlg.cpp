// CfgDlg.cpp : 实现文件
//

#include "stdafx.h"
#include "FXs.h"
#include "CfgDlg.h"
#include "afxdialogex.h"


// CfgDlg 对话框

IMPLEMENT_DYNAMIC(CfgDlg, CDialogEx)

CfgDlg::CfgDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CfgDlg::IDD, pParent)
{

}

CfgDlg::~CfgDlg()
{
}

void CfgDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_RICHEDIT21, m_edit);
}


BEGIN_MESSAGE_MAP(CfgDlg, CDialogEx)
	ON_BN_CLICKED(IDCANCEL, &CfgDlg::OnBnClickedCancel)
	ON_BN_CLICKED(IDOK, &CfgDlg::OnBnClickedOk)
	ON_BN_CLICKED(IDC_BUTTON_SAVE, &CfgDlg::OnBnClickedButtonSave)
	ON_EN_CHANGE(IDC_EDIT3, &CfgDlg::OnEnChangeEdit3)
END_MESSAGE_MAP()


// CfgDlg message handler


BOOL CfgDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	CFingerVeinV3::getInstance()->Dev_GetConfig(&conf_info);

	CString str;
	str.Format(_T("%d.%d"), conf_info.version.m, conf_info.version.s);
	SetDlgItemText(IDC_EDIT1, str);

	SetDlgItemInt(IDC_EDIT2, conf_info.identyTimeOut);
	SetDlgItemInt(IDC_EDIT3, conf_info.securityLevel);
	SetDlgItemInt(IDC_EDIT4, conf_info.deviceId);
	SetDlgItemInt(IDC_EDIT5, conf_info.LowThreshold);
	SetDlgItemInt(IDC_EDIT6, conf_info.HighThreshold);
	SetDlgItemInt(IDC_EDIT7, conf_info.defaultSessor);
	SetDlgItemInt(IDC_EDIT8, conf_info.level);
	for (int id = IDC_EDIT9; id <= IDC_EDIT16;id++)
		SetDlgItemInt(id, conf_info.onLight[id - IDC_EDIT9]);
	
	str.Format(_T("%.6f"), conf_info.fLeakRatioThreshold);
	SetDlgItemText(IDC_EDIT17, str);
	str.Format(_T("%.6f"), conf_info.fPressThreshold);
	SetDlgItemText(IDC_EDIT18, str);
	str.Format(_T("%.6f"), conf_info.modelScore);
	SetDlgItemText(IDC_EDIT19, str);
	str.Format(_T("%.6f"), conf_info.fEnergyThreshold);
	SetDlgItemText(IDC_EDIT20, str);

	CHARFORMAT cFmt;
	cFmt.cbSize = sizeof(CHARFORMAT);

	cFmt.crTextColor = RGB(64, 64, 64);
	cFmt.dwEffects &= ~CFE_BOLD;
	
	cFmt.dwMask = CFM_BOLD | CFM_COLOR | CFM_SIZE;
	_tcscpy_s(cFmt.szFaceName, _T("宋体"));
	cFmt.yHeight = 11 * 20;
	CTime time = CTime::GetCurrentTime();
	CString strcurTime = time.Format(_T("%H:%M:%S"));
	CString szText;
	CHARRANGE  SelDate;
	SelDate.cpMax = -1;
	SelDate.cpMin = -1;

	T_INFO buf;
	memset(&buf, 0, sizeof(T_INFO));
	CFingerVeinV3::getInstance()->Get_dev_info(&buf);

/*	m_edit.SetSel(SelDate);
	str.Format(_T("SDKinformation: Version： %d， Time：%d-%d \r\n"), buf.SdkVer, buf.SdkYear, buf.SdkDate);
	m_edit.SetSelectionCharFormat(cFmt);
	m_edit.ReplaceSel(str);
	*/
	m_edit.SetSel(SelDate);
	str.Format(_T("Firmware information: Version： %d.%d， Time：%d-%d-%d \r\n"), conf_info.version.m, conf_info.version.s, conf_info.product_day.year, conf_info.product_day.month, conf_info.product_day.day);
	m_edit.SetSelectionCharFormat(cFmt);
	m_edit.ReplaceSel(str);

	m_edit.SetSel(SelDate);
	str.Format(_T("USB information: manufacturer ID： 0x % 02x， device ID：0x % 02x， 识别ID：0x % 02x\r\n"), buf.UsbVendor, buf.UsbProduct, buf.SensorId);
	m_edit.SetSelectionCharFormat(cFmt);
	m_edit.ReplaceSel(str);

	m_edit.SetSel(SelDate);
	str.Format(_T("device type: %s \r\n"), buf.DevType == 0x01 ? _T("unilateral") : (buf.DevType == 0x02 ? _T("bilateral") : (buf.DevType == 0x03 ? _T("Top side") : (buf.DevType == 0x010 ? _T("fingerprint") : _T("recombination")))));
	m_edit.SetSelectionCharFormat(cFmt);
	m_edit.ReplaceSel(str);

/*	m_edit.SetSel(SelDate);
	str.Format(_T("CMOS信息: 分辨率： %d， 识别ID：%d， 时钟：%d\r\n"), buf.DevRes, buf.SensorId, buf.SensorClock);
	m_edit.SetSelectionCharFormat(cFmt);
	m_edit.ReplaceSel(str);

	m_edit.SetSel(SelDate);
	str.Format(_T("CMOS信息: 输出宽度： %d， 输出高度：%d， 最大曝光：%d， 最大增益：%d\r\n"), buf.Width, buf.Height, buf.MaxExp, buf.MaxGain);
	m_edit.SetSelectionCharFormat(cFmt);
	m_edit.ReplaceSel(str);
	*/
	m_edit.SetSel(SelDate);
	m_edit.SendMessage(WM_VSCROLL, SB_BOTTOM, 0);

	return TRUE;  // return TRUE unless you set the focus to a control
	// Exception: The OCX property page should return FALSE
}


void CfgDlg::OnBnClickedCancel()
{
	// TODO: Add control notification handler code here
	CDialogEx::OnCancel();
}


void CfgDlg::OnBnClickedOk()
{
	OnBnClickedButtonSave();

	CDialogEx::OnOK();
}


void CfgDlg::OnBnClickedButtonSave()
{
	CString str, csTemp;
	GetDlgItemText(IDC_EDIT1, str);
	AfxExtractSubString(csTemp, (LPCTSTR)str, 0, '.');
	conf_info.version.m = _ttoi((LPCTSTR)csTemp);
	AfxExtractSubString(csTemp, (LPCTSTR)str, 1, '.');
	conf_info.version.s = _ttoi((LPCTSTR)csTemp);

	conf_info.identyTimeOut = GetDlgItemInt(IDC_EDIT2);
	conf_info.securityLevel = GetDlgItemInt(IDC_EDIT3);
	conf_info.deviceId = GetDlgItemInt(IDC_EDIT4);
	conf_info.LowThreshold = GetDlgItemInt(IDC_EDIT5);
	conf_info.HighThreshold = GetDlgItemInt(IDC_EDIT6);
	conf_info.defaultSessor = GetDlgItemInt(IDC_EDIT7);
	conf_info.level = GetDlgItemInt(IDC_EDIT8);
	for (int id = IDC_EDIT9; id <= IDC_EDIT16; id++)
		conf_info.onLight[id - IDC_EDIT9] = GetDlgItemInt(id);

	GetDlgItemText(IDC_EDIT17, str);
	conf_info.fLeakRatioThreshold = _ttof((LPCTSTR)str);
	GetDlgItemText(IDC_EDIT18, str);
	conf_info.fPressThreshold = _ttof((LPCTSTR)str);
	GetDlgItemText(IDC_EDIT19, str);
	conf_info.modelScore = _ttof((LPCTSTR)str);
	GetDlgItemText(IDC_EDIT20, str);
	conf_info.fEnergyThreshold = _ttof((LPCTSTR)str);

	CFingerVeinV3::getInstance()->Dev_SetConfig(&conf_info);
}


void CfgDlg::OnEnChangeEdit3()
{
	// TODO: If the control is a RICHEDIT control, it will not
	// Send this notification unless rewriting CDialogEx::OnInitDialog()
	// and call CRichEditCtrl().SetEventMask(),
	// At the same time, add the ENM_CHANGE flag "or" to the mask.

	// TODO: Add control notification handler code here
}
