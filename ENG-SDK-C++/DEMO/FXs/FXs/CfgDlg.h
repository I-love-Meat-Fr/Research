#pragma once
#include "FingerVeinV3.h"
#include "afxcmn.h"

// CfgDlg 对话框

class CfgDlg : public CDialogEx
{
	DECLARE_DYNAMIC(CfgDlg)

public:
	CfgDlg(CWnd* pParent = NULL);   // 标准构造函数
	virtual ~CfgDlg();

// 对话框数据
	enum { IDD = IDD_CFG_DIALOG };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	configure_info conf_info;

	DECLARE_MESSAGE_MAP()
public:
	virtual BOOL OnInitDialog();
	afx_msg void OnBnClickedCancel();
	afx_msg void OnBnClickedOk();
	CRichEditCtrl m_edit;
	afx_msg void OnBnClickedButtonSave();
	afx_msg void OnEnChangeEdit3();
};
