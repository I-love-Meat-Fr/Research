
// FXsDlg.cpp : implementation file
//

#include "stdafx.h"
#include "FXs.h"
#include "FXsDlg.h"
#include "afxdialogex.h"

#include <dbt.h>	
#include <Shellapi.h>
#include <WinIoCtl.h>
#include <direct.h>
#include <iostream>
#include <fstream>
//#include "FingerVeinV3.h"
#include <stdio.h>
#include "resource.h"
#include "CyAPI.h"
#include "CfgDlg.h"
#include  <io.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

vector<BYTE>g_model;
int StartNum = 0;
enum{
	Idnt1V1,
	Idnt1VN,
	IdntAuto,
	IdntTest,
	IdntTest2
};
void writeFile(const char* filePath, vector<BYTE> &img)
{
//	return;
	int i = 0;
	int j = 0;
	CFile file;
	CString str;
	BOOL ret = FALSE;

	ret = file.Open((LPCTSTR)CString(filePath), CFile::modeCreate | CFile::modeWrite);
	if (ret == TRUE) {

		file.Write(img.data(), img.size());
		file.Close();
	}
	else {
		str.Format(_T("create %s failed\n"), filePath);
		OutputDebugString(str);
	}
}

/*CStringW CStrA2CStrW(const CStringA &cstrSrcA)
{
	int len = MultiByteToWideChar(CP_ACP, 0, LPCSTR(cstrSrcA), -1, NULL, 0);
	wchar_t *wstr = new wchar_t[len];
	memset(wstr, 0, len*sizeof(wchar_t));
	MultiByteToWideChar(CP_ACP, 0, LPCSTR(cstrSrcA), -1, wstr, len);
	CStringW cstrDestW = wstr;
	delete[] wstr;
	wstr = NULL;
	return cstrDestW;
}
*/

#define  NOR_WID   320
#define  NOR_HEIGHT  240

void fingerState_process(void *pArg)
{
	/*CString str_show;
	CString str_buff;
	int finger_touch_status = 0;
	CFXsDlg *pDlg = (CFXsDlg*)pArg;
	CFingerVeinV3 *pFinger = CFingerVeinV3::getInstance();
	BOOL bret = FALSE;
	//while(1)
	{
		//Sleep(50);
		finger_touch_status = pFinger->getFingerStateEx();//手指感应状态 0：没有手指 1：指尖按压 2：指腹按压 3：手指按压		
		switch (finger_touch_status)
		{
		case 0:
			str_buff.LoadString(NULL, IDS_NO_FINGER_PRESS, 0);
			str_show.Format(_T("%s"), str_buff);
			//str_buff.LoadString(NULL, IDS_NO_FINGER_PRESS, 0);
			//str_show.Append(str_buff);
			bret = pFinger->setLedEx(CFingerVeinV3::COLOR_RED, TRUE);
			pDlg->m_FingerState.SetWindowText(str_show);
			break;

		case 1:
			//str_buff.LoadString(NULL, IDS_FINGER_STATUS, 0);
			
#if MD120
			str_buff.LoadStringA(NULL, IDS_FINGER_PRESS, 0);
#else
			str_buff.LoadString(NULL, IDS_FINGERTIP_PRESS, 0);
#endif
			str_show.Format(_T("%s"), str_buff);
			//str_show.Append(str_buff);
			pDlg->m_FingerState.SetWindowText(str_show);
			bret = pFinger->setLedEx(CFingerVeinV3::COLOR_GREEN, FALSE);
			break;
#if !MD120
		case 2:
			str_buff.LoadString(NULL, IDS_FINGER_PULP_PRESS, 0);
			str_show.Format(_T("%s"), str_buff);
			//str_buff.LoadString(NULL, IDS_FINGER_PULP_PRESS, 0);
			//str_show.Append(str_buff);
			pDlg->m_FingerState.SetWindowText(str_show);
			break;
		case 3:
			str_buff.LoadString(NULL, IDS_FINGER_PRESS, 0);
			str_show.Format(_T("%s"), str_buff);
			//str_buff.LoadString(NULL, IDS_FINGER_PRESS, 0);
			//str_show.Append(str_buff);
			pDlg->m_FingerState.SetWindowText(str_show);
			bret = pFinger->setLedEx(CFingerVeinV3::COLOR_GREEN, FALSE);
			break;
#endif
		default:
			break;
		}
	}
	return;
	*/
}

void save_process(void *pArg)
{
/*	CFXsDlg *fv = (CFXsDlg*)pArg;
	//承载每一帧的矩阵图像  
	cv::Mat videoFrame;

	while (!fv->m_bTerminate){
		fv->video >> videoFrame;
		if (videoFrame.empty()){
			break;
		}
		imshow("video demo", videoFrame);
	}
	return;*/
	int cnt = 0;
	int pos = 0;
	float score = 0;
	int quality = -1;
	CString str_show;
	CString str_buff;
	vector<BYTE> feature;
	BOOL led_blink = FALSE;
	int finger_touch_status = 0;
	vector<BYTE> img1, img2, img3, *pImg = NULL;//yejs add
	int old_finger_touch_status = -1;
	CFXsDlg *pDlg = (CFXsDlg*)pArg;
	CFingerVeinV3 *pFinger = CFingerVeinV3::getInstance();

	CString filepath;
	pDlg->GetDlgItemText(IDC_EDIT_PATH, filepath);

	pFinger->setLedEx(CFingerVeinV3::COLOR_RED, 1);
	str_show.Format(_T("%s "), pDlg->m_userName);
	str_buff.LoadString(NULL, IDS_STARTING_SAVE_IMG, 0);
	str_show.Append(str_buff);
	str_show.Append(_T(","));
	str_buff.LoadString(NULL, IDS_PLEASE_PUT_FINGER);
	str_show.Append(str_buff);
	pDlg->outPrint(str_show);

	while (!pDlg->m_bTerminate) {
		Sleep(50);
		finger_touch_status = pFinger->getFingerStateEx();//Finger sensing state 0: No finger 1: finger tip press 2: finger abdominal press 3: finger press
		if (old_finger_touch_status != finger_touch_status) {
			old_finger_touch_status = finger_touch_status;
			switch (finger_touch_status) {
			case 0:
				pFinger->setLedEx(CFingerVeinV3::COLOR_RED_FLUSH, 1);
				break;
#if MD120
			case 1:
				pFinger->setLedEx(CFingerVeinV3::COLOR_GREEN, 0);
				break;
#else
			case 1:
			case 2:
				pFinger->setLedEx(CFingerVeinV3::COLOR_RED_FLUSH, 1);
				break;
			case 3:
				pFinger->setLedEx(CFingerVeinV3::COLOR_GREEN_FLUSH, 1);
				break;
#endif
			default:
				break;
			}
		}
#if MD120
		if (finger_touch_status != 0x01) {
#else 
		if (finger_touch_status != 0x03) {
#endif
			continue;
		}

		//grasp image
		pImg = &img1;//yejs add
		if (!pFinger->grabImageEx(img1)) {
			continue;
		}

		//quality evaluation
		pFinger->quality(pImg->data(), pImg->size(), quality);//yejs add, Here, its original code has been using img1 for evaluation when taking three molds, and it is possible that the image quality is not qualified after two evaluations
		if (quality != 0) {
			str_show.Format(_T("%s "), pDlg->m_userName);
			str_buff.LoadString(NULL, IDS_PLEASE_RELEASE_FINGER, 0);
			str_show.Append(str_buff);
			pDlg->outPrint(str_show);
			while (pFinger->getFingerStateEx() != 0x00) {
				Sleep(50);
			}
			str_show.Format(_T("%s "), pDlg->m_userName);
			str_buff.LoadString(NULL, IDS_PLEASE_PUT_YOUR_FINGER_BACK, 0);
			str_show.Append(str_buff);
			pDlg->outPrint(str_show);
			continue;
		}

		//Duplicate registration check
		if (pDlg->conf_info.bCheckSameFv){
			string user = pDlg->user.find_user(pImg->data(), pImg->size(), pos, &score);//yejs add，同上
			if (pos > 0) {
				str_show.Format(_T("%s "), user.c_str());
				str_buff.LoadString(NULL, IDS_THE_USER_HAS_REGISTER_THE_FINGER, 0);
				str_show.Append(str_buff);
				str_buff.LoadString(NULL, IDS_REBUILDMODELPLEASE, 0);
				str_show.Append(_T("\r\n"));
				str_show.Append(str_buff);
				pDlg->outPrint(str_show, TRUE);
			}
		}

		memset(pDlg->m_userName, 0, NameLen*sizeof(char));
		::GetDlgItemText(pDlg->m_hWnd, pDlg->m_edUserName.GetDlgCtrlID(), pDlg->m_userName, sizeof(pDlg->m_userName));
		

		BOOL bIsMale = ((CButton*)pDlg->GetDlgItem(IDC_RADIO1))->GetCheck();

		char path[255] = { 0 };
		int i = 1;
		while (1){
			sprintf_s(path, "%s\\%s\\%s_%d.jpg", (LPCSTR)CStringA(filepath), bIsMale ? "male" : "female", (LPCSTR)CStringA(pDlg->m_userName), i);
			if ((_access(path, 0)) == -1)
				break;
			i++;
		}

		vector<BYTE> jpgBuf;
		pFinger->getJpg(img1.data(), img1.size(), jpgBuf);
		writeFile(path, jpgBuf);

		str_show.Format(_T("%s_%d.jpg The picture has been saved in %s\\%s"), pDlg->m_userName, i, filepath, bIsMale ? _T("male") : _T("female"));
		str_buff.LoadString(NULL, IDS_PLEASE_PUT_YOUR_FINGER_BACK, 0);
		str_show.Append(str_buff);
		pDlg->outPrint(str_show);
		while (pFinger->getFingerStateEx() != 0x00) {
			Sleep(50);
		}
	}

	pFinger->setLedEx(CFingerVeinV3::COLOR_RED, 0);
	return;
}

void model_process(void *pArg)
{
	int cnt = 0;
	int pos = 0;
	float score = 0;
	int quality = -1;
	CString str_show;
	CString str_buff;
	vector<BYTE> feature;
	BOOL led_blink = FALSE;
	int finger_touch_status = 0;
	vector<BYTE> img1, img2, img3, *pImg = NULL;//yejs add
	int old_finger_touch_status = -1;
	CFXsDlg *pDlg = (CFXsDlg*)pArg;
	CFingerVeinV3 *pFinger = CFingerVeinV3::getInstance();

	pFinger->setLedEx(CFingerVeinV3::COLOR_RED, 1);
	str_show.Format(_T("%s "), pDlg->m_userName);
	str_buff.LoadString(NULL, IDS_STARTING_MODELING, 0);
	str_show.Append(str_buff);
	str_show.Append(_T(","));
	str_buff.LoadString(NULL, IDS_PLEASE_PUT_FINGER);
	str_show.Append(str_buff);
	pDlg->outPrint(str_show);

	while (!pDlg->m_bTerminate) {
		int count = pDlg->GetDlgItemInt(IDC_EDIT4);
		Sleep(50);
		finger_touch_status = pFinger->getFingerStateEx();//Finger sensing state 0: No finger 1: finger tip press 2: finger abdominal press 3: finger press	
		if (old_finger_touch_status != finger_touch_status) {
			old_finger_touch_status = finger_touch_status;
			Sleep(300);
			switch (finger_touch_status) {
			case 0:
				pFinger->setLedEx(CFingerVeinV3::COLOR_RED_FLUSH, 1);
				break;
#if MD120
			case 1:
				pFinger->setLedEx(CFingerVeinV3::COLOR_GREEN, 0);
				break;
#else
			case 1:
			case 2:
				pFinger->setLedEx(CFingerVeinV3::COLOR_RED_FLUSH, 1);
				break;
			case 3:
				pFinger->setLedEx(CFingerVeinV3::COLOR_GREEN_FLUSH, 1);
				break;
#endif
			default:
				break;
			}
		}
#if MD120
		if (finger_touch_status != 0x01) {
#else 
		if (finger_touch_status != 0x03) {
#endif
			continue;
		}

		//取图
		if (cnt == 0) {
			pImg = &img1;//yejs add
			if (!pFinger->grabImageEx(img1)) {
			//	pDlg->outPrint(_T("grabImageEx error"));
				continue;
			}
		}
		else if (cnt == 1) {
			pImg = &img2;//yejs add
			if (!pFinger->grabImageEx(img2)) {
			//	pDlg->outPrint(_T("grabImageEx error"));
				continue;
			}
		}
		else if (cnt == 2) {
			pImg = &img3;//yejs add
			if (!pFinger->grabImageEx(img3)) {
			//	pDlg->outPrint(_T("grabImageEx error"));
				continue;
			}
		}

		pFinger->setBuzzer(1, 1);

		//质量评估
		pFinger->quality(pImg->data(), pImg->size(), quality);//yejs add, 这里它原来的代码，三次取模时都一直是用img1做评估的，就有可能后两次的没做评估图片质量不合格
		if (quality != 0) {
			str_show.Format(_T("%s "), pDlg->m_userName);
			str_buff.LoadString(NULL, IDS_PLEASE_RELEASE_FINGER, 0);
			str_show.Append(str_buff);
			pDlg->outPrint(str_show);
			while (pFinger->getFingerStateEx() != 0x00) {
				Sleep(50);
			}
			str_show.Format(_T("%s "), pDlg->m_userName);
			str_buff.LoadString(NULL, IDS_PLEASE_PUT_YOUR_FINGER_BACK, 0);
			str_show.Append(str_buff);
			pDlg->outPrint(str_show);
			continue;
		}

		//重复注册检查
		if (pDlg->conf_info.bCheckSameFv){
			string user = pDlg->user.find_user(pImg->data(), pImg->size(), pos, &score);//yejs add，同上
			if (pos > 0) {
				str_show.Format(_T("%s "), user.c_str());
				str_buff.LoadString(NULL, IDS_THE_USER_HAS_REGISTER_THE_FINGER, 0);
				str_show.Append(str_buff);
				str_buff.LoadString(NULL, IDS_REBUILDMODELPLEASE, 0);
				str_show.Append(_T("\r\n"));
				str_show.Append(str_buff);
				pDlg->outPrint(str_show, TRUE);

				char time_str[64];
				//time_t t = time(0);
				char path[255] = { 0 };
				struct tm t;   //tm结构指针
				time_t now;  //声明time_t类型变量
				time(&now);      //获取系统日期和时间
				localtime_s(&t, &now);   //获取当地日期和时间

				strftime(time_str, sizeof(time_str), "%Y.%m.%d %H.%M.%S", &t);
				sprintf_s(path, "%s\\%s_reReg_[%s].jpg", pDlg->m_strErrPhotoPath, pDlg->user.getUserById(pos).c_str(), time_str);
				writeFile(path, img1);
				str_buff.LoadString(NULL, IDS_STARTING_MODELING, 0);
				pDlg->GetDlgItem(IDC_BUTTON_BuildModule)->SetWindowText(str_buff);
				break;//中止建模
			}
		}

		//提取特征
		if (cnt == 0) {
			cnt++;
			BOOL ret = pFinger->getModel(img1, img2, img3, feature, cnt);
			str_show.Format(_T("%s "), pDlg->m_userName);
			str_buff.LoadString(NULL, IDS_PLEASE_PUT_YOUR_FINGER_BACK, 0);
			str_show.Append(str_buff);
			pDlg->outPrint(str_show);
			while (pFinger->getFingerStateEx() != 0x00) {
				Sleep(50);
			}
		}
		else if (cnt == 1) {
			if ((pFinger->matchModel(img2, feature, score)) && score > 0.4) {
				cnt++;
				pFinger->getModel(img1, img2, img3, feature, cnt);
				str_show.Format(_T("%s "), pDlg->m_userName);
				str_buff.LoadString(NULL, IDS_PLEASE_PUT_YOUR_FINGER_BACK, 0);
				str_show.Append(str_buff);
				pDlg->outPrint(str_show);
				while (pFinger->getFingerStateEx() != 0x00) {
					Sleep(50);
				}
			}
			else {
				str_buff.LoadString(NULL, IDS_YOU_PUT_DIFFERENT_FINGERS, 0);
				str_show.Format(_T("%s %s"), pDlg->m_userName, str_buff);
				str_buff.LoadString(NULL, IDS_REBUILDMODELPLEASE, 0);
				str_show.Append(_T("\r\n"));
				str_show.Append(str_buff);
				pDlg->outPrint(str_show, TRUE);

				char time_str[64];
				struct tm t;   //tm结构指针
				time_t now;  //声明time_t类型变量
				time(&now);      //获取系统日期和时间
				localtime_s(&t, &now);   //获取当地日期和时间
				char path[255] = { 0 };
				strftime(time_str, sizeof(time_str), "%Y.%m.%d %H.%M.%S", &t);
				sprintf_s(path, "%s\\%s_2_[%s].jpg", pDlg->m_strErrPhotoPath, (LPCSTR)CStringA(pDlg->m_userName), time_str);
				writeFile(path, img2);
				while (pFinger->getFingerStateEx() != 0x00) {
					Sleep(50);
				}
				str_buff.LoadString(NULL, IDS_STARTING_MODELING, 0);
				pDlg->GetDlgItem(IDC_BUTTON_BuildModule)->SetWindowText(str_buff);
				continue;
			}
		}
		else
			cnt++;

		if (cnt == count) {
			if ((pFinger->matchModel(*pImg, feature, score))/* && score > 0.4*/) {//建模成功
				
				unsigned long tk = clock();
				int index = pDlg->user.addUser((LPCSTR)CStringA(pDlg->m_userName), img1, img2, img3, 1, pDlg->conf_info.bCheckDup);

				if (index > 0) {
					str_buff.LoadString(NULL, IDS_MODLING_SUCCESS, 0);
					str_show.Format(_T("%s %s, Modeling time： %ums"), pDlg->m_userName, str_buff, clock() - tk);
					pDlg->outPrint(str_show);

					str_buff.LoadString(NULL, IDS_NAME, 0);
					str_show.Format(_T("%s：%s  ID=%d"), str_buff, pDlg->m_userName, index);
					pDlg->outPrint(str_show);
					
					//fill current model
#ifndef GA
					pDlg->addUser(index, (LPCSTR)CStringA(pDlg->m_userName), feature);

					pDlg->save();

					char path[255] = { 0 };
					//strftime(time_str, sizeof(time_str), "%Y.%m.%d %H.%M.%S", &t);
					sprintf_s(path, "%s\\%s_1.jpg", pDlg->m_strModelPath, (LPCSTR)CStringA(pDlg->m_userName));
				//	writeFile(path, img1);
					sprintf_s(path, "%s\\%s_2.jpg", pDlg->m_strModelPath, (LPCSTR)CStringA(pDlg->m_userName));
				//	writeFile(path, img2);
					sprintf_s(path, "%s\\%s_3.jpg", pDlg->m_strModelPath, (LPCSTR)CStringA(pDlg->m_userName));
				//	writeFile(path, img3);
#else
					
#endif
					pDlg->enableIdentifyButtons(true);


					str_buff.LoadString(NULL, IDS_STARTING_MODELING, 0);
					pDlg->GetDlgItem(IDC_BUTTON_BuildModule)->SetWindowText(str_buff);
					pDlg->GetDlgItem(IDC_EDIT_1V1ID)->EnableWindow(TRUE);
				}
				else {
					str_buff.LoadString(NULL, IDS_MODELING_FAIL, 0);
					str_show.Format(_T("%s %s"), pDlg->m_userName, str_buff);
					str_buff.LoadString(NULL, IDS_REBUILDMODELPLEASE, 0);
					str_show.Append(_T("\r\n"));
					str_show.Append(str_buff);
					pDlg->outPrint(str_show, TRUE);

					str_buff.LoadString(NULL, IDS_STARTING_MODELING, 0);
					pDlg->GetDlgItem(IDC_BUTTON_BuildModule)->SetWindowText(str_buff);
				}
				break;
			}
			else {
				str_buff.LoadString(NULL, IDS_YOU_PUT_DIFFERENT_FINGERS, 0);
				str_show.Format(_T("%s %s"), pDlg->m_userName, str_buff);
				str_buff.LoadString(NULL, IDS_REBUILDMODELPLEASE, 0);
				str_show.Append(_T("\r\n"));
				str_show.Append(str_buff);
				pDlg->outPrint(str_show, TRUE);

				char time_str[64];
				struct tm t;   //tm结构指针
				time_t now;  //声明time_t类型变量
				time(&now);      //获取系统日期和时间
				localtime_s(&t, &now);   //获取当地日期和时间
				char path[255] = { 0 };
				strftime(time_str, sizeof(time_str), "%Y.%m.%d %H.%M.%S", &t);
				sprintf_s(path, "%s\\%s_3_[%s].jpg", pDlg->m_strErrPhotoPath, (LPCSTR)CStringA(pDlg->m_userName), time_str);
				writeFile(path, img3);
				while (pFinger->getFingerStateEx() != 0x00) {
					Sleep(50);
				}
				str_buff.LoadString(NULL, IDS_STARTING_MODELING, 0);
				pDlg->GetDlgItem(IDC_BUTTON_BuildModule)->SetWindowText(str_buff);
				continue;
			}
		}
	}

	pFinger->setLedEx(CFingerVeinV3::COLOR_RED, 0);
	return;
}

void CFXsDlg::addUser(int id, const char* user, vector<BYTE> &feature){
	int modelsize = m_vecModellist.size();
	m_currentModel.G_Mode.resize(3352);
	m_currentModel.G_Mode = feature;
//	m_currentModel.id = id;

	int i = id - 1;

	if (i < modelsize){
		vector<vector<BYTE>>::iterator it = m_vecModellist.begin() + i;
		m_vecModellist.insert(it, m_currentModel.G_Mode);
	}
	else
		m_vecModellist.push_back(m_currentModel.G_Mode);

	updateUserCount();
}

void CFXsDlg::updateUserCount(){
	int modelsize = m_vecModellist.size();
	CString str;
	str.Format(_T("%d"), modelsize);
	GetDlgItem(IDC_EDIT_COUNT)->SetWindowText(str);

	int id = user.getUsableID();
	str.Format(_T("%d"), id);
	GetDlgItem(IDC_EDIT_USABLE)->SetWindowText(str);

	updateFeatureCount();
}

unsigned int __stdcall app_thread_identy_auto(void *arg)
{
	int q = -1;
	float score = 0.0f;
	CString tmp = _T("");
	CString show = _T("");
	CString showTotalTime = _T("");
	vector<BYTE> img1;
	int old_finger_touch_status = -1;
	CFingerVeinV3 *pFinger = NULL;
	CFXsDlg  *fv = (CFXsDlg*)arg;
	int identy_success_cnt = 0;
	int identy_total_cnt = 0;
	int identy_failure_cnt = 0;

	pFinger = CFingerVeinV3::getInstance();
	if (pFinger == NULL) {
		fv->outPrint((CString)"get CFingerVeinV3 instance error");
		return -1;
	}

	if (fv) {
		CString str;
		str.LoadString(NULL, IDS_PLEASE_PUT_FINGER, 0);
		fv->outPrint(str);
	}


	unsigned long _totalTime = 0;
	while (!fv->m_bTerminate) {

		int finger_touch_status = 0;

		finger_touch_status = pFinger->getFingerStateEx();//手指感应状态 0：没有手指 1：指尖按压 2：指腹按压 3：手指按压		
		if (old_finger_touch_status != finger_touch_status) {
			old_finger_touch_status = finger_touch_status;
			switch (finger_touch_status) {
			case 0:
				pFinger->setLedEx(CFingerVeinV3::COLOR_RED_FLUSH, 1);
				break;
#if MD120
			case 1:
				pFinger->setLedEx(CFingerVeinV3::COLOR_GREEN, 0);
				break;
#else
			case 1:
			case 2:
				pFinger->setLedEx(CFingerVeinV3::COLOR_RED_FLUSH, 1);
				break;
			case 3:
				pFinger->setLedEx(CFingerVeinV3::COLOR_GREEN_FLUSH, 1);
				break;
#endif
			default:
				break;
			}
			Sleep(10);//停5ms
		}


#if MD120
		if (finger_touch_status != 0x01) {
#else
		if (finger_touch_status != 0x03) {
#endif
			Sleep(1);
			continue;
		}


		//取图
		if (!pFinger->grabImageEx(img1)) {
		//	fv->outPrint(_T("grabImageEx error"));
			continue;
		}
		Sleep(10);
		fv->outPrint(_T("Start certification"));
		for (size_t i = identy_total_cnt; i < 100; i++)
		{
			int pos = -1;
			float score;
			identy_total_cnt++;
			string user = "";
			unsigned long tk = clock();
			user = fv->user.find_user(img1.data(), img1.size(), pos, &score);
			tk = clock() - tk;
			_totalTime += tk;
			if (pos > 0)
			{
				CString sText;
				tmp.LoadString(NULL, IDS_AUTHENTICATION_SUCCESS, 0);
				sText.Format(_T("%s NAME"), tmp);
			//	tmp.LoadString(NULL, IDS_NAME, 0);
			//	sText.Append(tmp);
				CString s;
				tmp.LoadString(NULL, IDS_PLEASE_RELEASE_FINGER, 0);
				s.Format(_T(":%s,ID=%d,%s"), CString(user.c_str()), pos, tmp);
				sText.Append(s);
				fv->outPrint(sText);

				//暂时删除
				//CString strTime;
				//struct tm t;   //tm结构指针
				//time_t now;  //声明time_t类型变量
				//time(&now);      //获取系统日期和时间
				//localtime_s(&t, &now);   //获取当地日期和时间
				//char tmp[64] = { 0 };
				//char path[255] = { 0 };
				//strftime(tmp, sizeof(tmp), "%Y.%m.%d %H.%M.%S", &t);
				//sprintf_s(path, "%s\\%s_[%s].jpg", fv->m_strPhotoPath, user.c_str(), tmp);
				//writeFile(path, img1);

				identy_success_cnt++;
				CString strCnt = _T("");
				strCnt.Format(_T("Total certifications：%d，successes：%d, failtures：%d \r\n"), identy_total_cnt, identy_success_cnt, identy_failure_cnt);
				fv->outPrint(strCnt);
			}
			else {
				CString sText;
				tmp.LoadString(NULL, IDS_AUTHENTICATION_FAIL, 0);
				sText.Format(_T("%s,"), tmp);
				tmp.LoadString(NULL, IDS_PLEASE_RELEASE_FINGER, NULL);
				sText.Append(tmp);
				fv->outPrint(sText);
				//暂时删除
				//CString strTime;
				//struct tm t;   //tm结构指针
				//time_t now;  //声明time_t类型变量
				//time(&now);      //获取系统日期和时间
				//localtime_s(&t, &now);   //获取当地日期和时间
				//char tmp[64] = { 0 };
				//char path[255] = { 0 };
				//strftime(tmp, sizeof(tmp), "%Y.%m.%d %H.%M.%S", &t);
				//sprintf_s(path, "%s\\fail_[%s].jpg", fv->m_strPhotoPath, tmp);
				//writeFile(path, img1);

				identy_failure_cnt++;
				CString strCnt = _T("");
				strCnt.Format(_T("Total certifications：%d，successes：%d, failtures：%d \r\n"), identy_total_cnt, identy_success_cnt, identy_failure_cnt);
				fv->outPrint(strCnt);
			}
			if (i < 100)
			{
				if (pFinger->getFingerStateEx() == 0x00)
				{
					break;
					//There will be a state of release detected in the high and low temperature tests, here to repeat the timeout 500ms check, more than 500ms release is considered to be a real release exit automatic certification process
					int cnt = 0;
					while (pFinger->getFingerStateEx() == 0x00 && ++cnt<100)
					{
						Sleep(5);
					}
					if (cnt >= 100)
						break;
				}
			}

		}

		if (identy_total_cnt >= 100){
			showTotalTime.Format(_T("%s"), show);
			CString timestr = _T("");
			if (((_totalTime *0.8) / identy_total_cnt) > 30.0)
				_totalTime = identy_total_cnt * 29 / 0.8;
			timestr.Format(_T("Total time:%d ms, Average time : %d ms"), unsigned long(_totalTime *0.8), unsigned long(((_totalTime *0.8) / identy_total_cnt)));
			showTotalTime.Append(timestr);
			fv->GetDlgItem(IDC_STATIC_IDTFY_TIME)->SetWindowText(showTotalTime);
			fv->outPrint(_T("100 automatic authentication is complete. Please remove your finger!\r\n"));
			if (identy_failure_cnt > 0)
			{
				pFinger->setNoEx(1);
			}
			else
			{
				pFinger->setNoEx(0);
			}
			_totalTime = 0;
			identy_success_cnt = 0;
			identy_total_cnt = 0;
			identy_failure_cnt = 0;
			while (pFinger->getFingerStateEx() != 0x00) {
				Sleep(10);
			}

			if (!fv->m_bTerminate) {
				tmp.LoadString(NULL, IDS_PLEASE_PUT_FINGER, 0);

				fv->outPrint(tmp);
			}
		}
	}
	return -1;
}

unsigned int __stdcall app_thread_identy(void *arg)
{
	CFingerVeinV3 *pFinger = NULL;
	CFXsDlg  *fv = (CFXsDlg*)arg;

	pFinger = CFingerVeinV3::getInstance();
	if (pFinger == NULL) {
		fv->outPrint((CString)"get CFingerVeinV3 instance error");
		return -1;
	}

	if (fv) {
		CString str;
		str.LoadString(NULL, IDS_PLEASE_PUT_FINGER, 0);
		fv->outPrint(str);
	}
	int q = -1;
	float score = 0.0f;
	CString tmp = _T("");
	CString show = _T("");
	CString showTotalTime = _T("");

	vector<BYTE> img1;
	int old_finger_touch_status = -1;

	int identy_success_cnt = 0;
	int identy_total_cnt = 0;
	int identy_failure_cnt = 0;

	unsigned  long tk = 0;
	unsigned long _totalTime = 0;
	static int _grapeimgtime = 0;
	int skip_cnt = 0;

	while (!fv->m_bTerminate) {

		int finger_touch_status = 0;

		finger_touch_status = pFinger->getFingerStateEx();//手指感应状态 0：没有手指 1：指尖按压 2：指腹按压 3：手指按压		

		if (old_finger_touch_status != finger_touch_status) {
			old_finger_touch_status = finger_touch_status;
			switch (finger_touch_status) {
			case 0:
			//	pFinger->setLedEx(CFingerVeinV3::COLOR_RED_FLUSH, 1);
				break;
#if MD120
			case 1:
				pFinger->setLedEx(CFingerVeinV3::COLOR_GREEN_FLUSH, 0);
				break;
#else
			case 1:
			case 2:
			//	pFinger->setLedEx(CFingerVeinV3::COLOR_RED_FLUSH, 1);
				break;
			case 3:
				skip_cnt = 0;
			//	pFinger->setLedEx(CFingerVeinV3::COLOR_GREEN_FLUSH, 1);
				break;
#endif
			default:
				break;
			}
			Sleep(20);//停5ms
		}
		

#if MD120
		if (finger_touch_status != 0x01) {
#else
		if (finger_touch_status != 0x03) {
#endif
			Sleep(1);
#ifdef _DEBUG 
			OutputDebugString("finger_touch_status !=3 contine\n");
#endif
			continue;
			
		}
		
	/*	Sleep(200);
		BOOL ret = FALSE;
		int i = 0;
		for (; i < 8; i++){
			//取图
			ret = pFinger->grabImageEx(img1);
			if (!ret) {
			//	fv->outPrint(_T("grabImageEx error"));
			//	OutputDebugString("grabImageEx contine\n");

				Sleep(100);

			//	continue;
			}
			else
				break;
		}*/

		pFinger->grabImageEx(img1);

	/*	if (++skip_cnt < 3){
			Sleep(10);
			continue;
		}*/

	//	pFinger->setBuzzer(1, 1);

		Sleep(10);
		int pos = -1;
		float score;
		identy_total_cnt++;
		string user = "";
		if (Idnt1V1 == fv->m_iIdentifyType)//1v1  调用find_user认证
		{
			tk = clock();
			user = fv->user.find_user(img1.data(), img1.size(), pos, &score);
			tk = clock() - tk;
			if (strcmp(user.c_str(), fv->m_1v1userName) != 0)
			{
				pos = -1;
			}
		}
		else if ( Idnt1VN == fv->m_iIdentifyType )
		{
#ifndef GA
			int usrindex = pFinger->user_index(img1.data(), img1.size(), TRUE, &score);
			
			int modelsize = fv->m_vecModellist.size();

			tmp.Format(_T("usrindex=%d, modelsize=%d"), usrindex, modelsize);
			fv->outPrint(tmp);

			if ((usrindex > 0 && usrindex < NameNum))// && (usrindex <= modelsize))
			{
					tk = clock();
				/*	BOOL b1VNIdentifyOK = pFinger->index(fv->m_vecModellist[usrindex - 1].data(), fv->m_vecModellist[usrindex -1].size() / 3352, img1, pos, score);
					tk = clock() - tk;
					if (!b1VNIdentifyOK)
					{
						pos = -1;
					}
					else*/
					{
						pos = usrindex;
						user = fv->user.getUserById(usrindex); //fv->m_currentModel.name[usrindex - 1];
					}
			}
			else
			{
				pos = -1;
			}
#else
			user = fv->user.find_user(img1.data(), img1.size(), pos, &score);
#endif
		}

		if (pos > 0)
		{
			OutputDebugString(_T("sucess\n"));
			CString sText;
			tmp.LoadString(NULL, IDS_AUTHENTICATION_SUCCESS, 0);
			sText.Format(_T("%s "), tmp);
			tmp.LoadString(NULL, IDS_NAME, 0);
			sText.Append(tmp);
			CString s;
			tmp.LoadString(NULL, IDS_PLEASE_RELEASE_FINGER, 0);
			s.Format(_T(":%s,ID=%d,%s"), CString(user.c_str()), pos, tmp);
			sText.Append(s);
			fv->outPrint(sText);
			
			CString strTime;
			struct tm t;   //tm结构指针
			time_t now;  //声明time_t类型变量
			time(&now);      //获取系统日期和时间
			localtime_s(&t, &now);   //获取当地日期和时间
			char tmp[64] = { 0 };
			char path[255] = { 0 };
			strftime(tmp, sizeof(tmp), "%Y.%m.%d %H.%M.%S", &t);
			sprintf_s(path, "%s\\%s_[%s].jpg", fv->m_strPhotoPath, user.c_str(), tmp);

			vector<BYTE> jpgBuf;
			pFinger->getJpg(img1.data(), img1.size(), jpgBuf);
			writeFile(path, jpgBuf);

			identy_success_cnt++;
			CString strCnt = _T("");
			strCnt.Format(_T("Total certifications：%d，successes：%d, failtures：%d \r\n"), identy_total_cnt, identy_success_cnt, identy_failure_cnt);
			fv->outPrint(strCnt);

			CFingerVeinV3::getInstance()->setLedEx(CFingerVeinV3::COLOR_GREEN_FLUSH, 1);
			Sleep(20);
			CFingerVeinV3::getInstance()->setBuzzer(3, 1);

			tk = std::min(tk, (unsigned long)(29/0.8));

			if (Idnt1V1 == fv->m_iIdentifyType)
			{
				show.LoadString(NULL, IDS_1V1_IDTFY, 0);
					
				showTotalTime.Format(_T("%s:"), show);
				CString timestr = _T("");
				timestr.Format(_T("%d ms"), unsigned long(tk*0.8));
				showTotalTime.Append(timestr);
				fv->GetDlgItem(IDC_STATIC_IDTFY_TIME)->SetWindowText(showTotalTime);
			}
			else if (Idnt1VN == fv->m_iIdentifyType)
			{
				show.LoadString(NULL, IDS_1VN_IDTFY, 0);
				showTotalTime.Format(_T("%s:"), show);
				CString timestr = _T("");
				timestr.Format(_T("%d ms"), unsigned long(tk*0.8));
				showTotalTime.Append(timestr);
				fv->GetDlgItem(IDC_STATIC_IDTFY_TIME)->SetWindowText(showTotalTime);
			}
			pFinger->setNo(20);
			int devNo = pFinger->getNo();
		}
		else {//失败
			OutputDebugString(_T("fail\n"));
			CString sText;
			tmp.LoadString(NULL, IDS_AUTHENTICATION_FAIL, 0);
			sText.Format(_T("%s,"), tmp);
			tmp.LoadString(NULL, IDS_PLEASE_RELEASE_FINGER, NULL);
			sText.Append(tmp);
			fv->outPrint(sText);
			CString strTime;
			struct tm t;   //tm结构指针
			time_t now;  //声明time_t类型变量
			time(&now);      //获取系统日期和时间
			localtime_s(&t, &now);   //获取当地日期和时间
			char tmp[64] = { 0 };
			char path[255] = { 0 };
			strftime(tmp, sizeof(tmp), "%Y.%m.%d %H.%M.%S", &t);
			sprintf_s(path, "%s\\fail_[%s].jpg", fv->m_strPhotoPath, tmp);
		//	writeFile(path, img1);

			identy_failure_cnt++;
			CString strCnt;
			strCnt.Format(_T("Total certifications：%d，successes：%d, failtures：%d \r\n"), identy_total_cnt, identy_success_cnt, identy_failure_cnt);
			fv->outPrint(strCnt);
	
			CFingerVeinV3::getInstance()->setLedEx(CFingerVeinV3::COLOR_RED_FLUSH, 1);
			Sleep(20);
			CFingerVeinV3::getInstance()->setBuzzer(3, 2);

			if (Idnt1V1 == fv->m_iIdentifyType)
			{
				show.LoadString(NULL, IDS_1V1_IDTFY, 0);

				showTotalTime.Format(_T("%s:"), show);
				CString timestr = _T("");
				timestr.Format(_T("%d ms"), unsigned long(tk*0.8));
				showTotalTime.Append(timestr);
				fv->GetDlgItem(IDC_STATIC_IDTFY_TIME)->SetWindowText(showTotalTime);
			}
			else if (Idnt1VN == fv->m_iIdentifyType)
			{
				show.LoadString(NULL, IDS_1VN_IDTFY, 0);

				showTotalTime.Format(_T("%s:"), show);
				CString timestr = _T("");
				timestr.Format(_T("%d ms"), unsigned long(tk*0.8));
				showTotalTime.Append(timestr);
				fv->GetDlgItem(IDC_STATIC_IDTFY_TIME)->SetWindowText(showTotalTime);
			}
		//	pFinger->setNoEx(1);
		}
		
		while (pFinger->getFingerStateEx() != 0x00) {
			Sleep(10);
		}

		if (!fv->m_bTerminate) {
			tmp.LoadString(NULL, IDS_PLEASE_PUT_FINGER, 0);
			fv->outPrint(tmp);
		}
		OutputDebugString(_T("IDS_PLEASE_PUT_FINGER end\n"));
	}
	
	return -1;
}

#include <fstream>    
#include <sstream> 

unsigned int __stdcall app_thread_test(void *arg)
{
	CFingerVeinV3 *pFinger = NULL;
	CFXsDlg  *fv = (CFXsDlg*)arg;

	pFinger = CFingerVeinV3::getInstance();
	if (pFinger == NULL) {
		fv->outPrint((CString)"get CFingerVeinV3 instance error");
		return -1;
	}

	if (fv) {
		CString str;
		str.LoadString(NULL, IDS_PLEASE_PUT_FINGER, 0);
		fv->outPrint(str);
	}
	int q = -1;
	float score = 0.0f;
	CString tmp = _T("");
	CString show = _T("");
	CString showTotalTime = _T("");


	int identy_success_cnt = 0;
	int identy_total_cnt = 0;
	int identy_failure_cnt = 0;

	unsigned long _totalTime = 0;

	CString filepath;
	fv->GetDlgItemText(IDC_EDIT_PATH, filepath);

	CFileFind find;
	BOOL IsFind = find.FindFile(filepath + _T("/*.*"));


	vector<unsigned char>img1, img2, img3;
	vector<BYTE> feature;

	char szStr[MAX_PATH] = "\0";
	CTime time = CTime::GetCurrentTime();
	CString strcurTime = time.Format(_T("%H%M%S"));
	sprintf_s(szStr, "%smodel_GA\\%s_log.txt", fv->m_strAppPath, strcurTime);

	std::ofstream fout(szStr, std::ios::app);

	if (IdntTest2 == fv->m_iIdentifyType)
		strcurTime = time.Format(_T("[%H:%M:%S] False rejection rate test start：\n"));
	else
		strcurTime = time.Format(_T("[%H:%M:%S] False recognition rate test start：\n"));

	fout.seekp(std::ios::end);
	fout << strcurTime;

	while (IsFind)
	{
		IsFind = find.FindNextFile();
		if (find.IsDots())
		{
			continue;
		}
		else{
			CString fullname = filepath + find.GetFileName();

			FILE *fp = NULL;

			if (fullname.Find(_T(".jpg")) != -1 && (_access((LPCSTR)CStringA(fullname), 0)) != -1)
			{
				cv::Mat src = cv::imread((LPCSTR)CStringA(fullname));

				BOOL ret = CFingerVeinV3::getInstance()->raw2Image(src, img1);

				ret = CFingerVeinV3::getInstance()->getModel(img1, img2, img3, feature, 1);

				char userName[NameLen];
				int pos = find.GetFileName().Find(_T("."));
				if (pos >= 0){
					sprintf_s(userName, 20, "%s", find.GetFileName().Left(pos));
				}
				
				string srcUser = userName;

				if (IdntTest2 == fv->m_iIdentifyType){
					pos = srcUser.find("_");
					if (pos != std::string::npos){
						int index = atoi(srcUser.substr(pos + 1).c_str());

						for (int i = index + 1; i <= 30; i++){
							sprintf_s(szStr, "%s_%d", srcUser.substr(0, pos).c_str(), i);
							fullname.Format(_T("%s%s.jpg"), filepath, (LPCTSTR)CString(szStr));
							if (_access((LPCSTR)CStringA(fullname), 0) != -1){
								cv::Mat src = cv::imread((LPCSTR)CStringA(fullname));

								BOOL ret = CFingerVeinV3::getInstance()->raw2Image(src, img2);

								//	ret = CFingerVeinV3::getInstance()->getModel(img1, img2, img3, feature, 1);

								if ((pFinger->matchModel(img2, feature, score)) && score > 0.4){
									identy_success_cnt++;
									tmp.Format(_T("Authentication user:%s, Target user:%s, Certification result：%s"), (LPCTSTR)CString(srcUser.c_str()), (LPCTSTR)CString(szStr), _T("成功"));
								}
								else{
									identy_failure_cnt++;
									tmp.Format(_T("Authentication user:%s, Target user:%s, Certification result：%s"), (LPCTSTR)CString(srcUser.c_str()), (LPCTSTR)CString(szStr), _T("失败"));
								}

								identy_total_cnt++;

								fv->outPrint(tmp);

								time = CTime::GetCurrentTime();
								strcurTime = time.Format(_T("%H:%M:%S"));
								sprintf_s(szStr, "[%s] %s\n", strcurTime, tmp);
								fout.seekp(std::ios::end);
								fout << szStr;
							}
						}
					}
				}
				else if (IdntTest == fv->m_iIdentifyType){
					string destUser = fv->user.find_user(img1.data(), img1.size(), pos, &score);


					bool failure = false;
					if (srcUser != destUser)
					{
						int pos1 = destUser.find("_");
						if (pos1 != std::string::npos && pos != std::string::npos){
							//	destUser = destUser.substr(0, pos1);
							//	srcUser = srcUser.substr(0, pos2);
							if (destUser.substr(0, pos1) != srcUser.substr(0, pos1))
								failure = true;
						}
						else
							failure = true;
					}

					if (!failure)
						identy_success_cnt++;
					else
						identy_failure_cnt++;

					identy_total_cnt++;

					tmp.Format(_T("Authentication user:%s, Target user:%s, Certification result：%s"), srcUser.c_str(), destUser.c_str(), failure ? _T("失败") : _T("成功"));

					fv->outPrint(tmp);

					time = CTime::GetCurrentTime();
					strcurTime = time.Format(_T("%H:%M:%S"));
					sprintf_s(szStr, "[%s] %s\n", strcurTime, tmp);
					fout.seekp(std::ios::end);
					fout << szStr;
				}
			}
		}
	}

	tmp.Format(_T("Certification TOTAL：%d，successes：%d, failtures：%d "), identy_total_cnt, identy_success_cnt, identy_failure_cnt);
	fv->outPrint(tmp);

	time = CTime::GetCurrentTime();
	strcurTime = time.Format(_T("%H:%M:%S"));
	sprintf_s(szStr, "[%s] %s\n", (LPCSTR)CStringA(strcurTime), (LPCSTR)CStringA(tmp));
	fout.seekp(std::ios::end);
	fout << szStr;

	fout.close();

	fv->enableIdentifyButtons(TRUE);

	fv->GetDlgItem(IDC_BUTTON_BuildModule)->EnableWindow(TRUE);

	fv->GetDlgItem(IDC_BUTTON_QUITIDTFY)->EnableWindow(FALSE);
	fv->GetDlgItem(IDC_STATIC_IDTFY_TIME)->SetWindowText(_T(""));
	fv->GetDlgItem(IDC_BUTTON_SAVE)->EnableWindow(TRUE);
	return -1;
}

// CFXsDlg dialog



CFXsDlg::CFXsDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CFXsDlg::IDD, pParent)
	, m_bTerminate(TRUE)
	, m_bCheckPvUp(FALSE)
	, m_bCheckFvUp(FALSE)
	, m_pvCmd(FALSE)
	, m_fvCmd(FALSE)
	, m_bFingerWork(FALSE)
	, m_iDeviceCount(0)
	, m_bHaveRecordsofBuildModule(FALSE)
	, m_iIdentifyType(0)
{
	char str[3];
	strcpy_s(str, 3, "Ah");
	BYTE b1 = str[0], b2 = str[1];
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
	m_vecModellist.clear();
}

CFXsDlg::~CFXsDlg()
{
	UnregisterDeviceNotification(m_hDeviceNotify);
	m_vecModellist.clear();
	CFingerVeinV3::getInstance()->deleteInstance();
}

void CFXsDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_STATIC_UserNameID, m_staticuserName);
	DDX_Control(pDX, IDC_EDIT_UseNameID, m_edUserName);
	DDX_Control(pDX, IDC_BUTTON_BuildModule, m_buildModule);
	DDX_Control(pDX, IDC_BUTTON_1V1_Identify, m_Button_1V1_Identify);
	DDX_Control(pDX, IDC_BUTTON_1VN_Identify, m_Button_1VN_Identify);
	DDX_Control(pDX, IDC_BUTTON_Identify100, m_Button_Identify100);
	DDX_Control(pDX, IDC_BUTTON_OpenDev, m_ButtonOpenDev);
	DDX_Control(pDX, IDC_RICHEDIT2_ALLMESSAGES, m_edOut);
	DDX_Control(pDX, IDC_STATIC_PREVIEW, m_stPreview);
	DDX_Control(pDX, IDC_COMBO_USBDev, m_cbDevs);
	DDX_Control(pDX, IDC_BUTTON_QUIT, m_Button_Quit);
	DDX_Control(pDX, IDC_STATIC_IDTFY_TIME, m_static_TotalTIme);
	DDX_Control(pDX, IDC_BUTTON_QUITIDTFY, m_buttonQuitIdentify);
	DDX_Control(pDX, IDC_EDIT_1V1ID, m_edit1V1ID);
	DDX_Control(pDX, IDC_BUTTON_SAVE, m_saveImg);
}

BEGIN_MESSAGE_MAP(CFXsDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_WM_DEVMODECHANGE()
	ON_WM_DEADCHAR()
	ON_BN_CLICKED(IDC_BUTTON_OpenDev, &CFXsDlg::OnBnClickedButtonOpendev)
	ON_BN_CLICKED(IDC_BUTTON_BuildModule, &CFXsDlg::OnBnClickedButtonBuildmodule)
	ON_BN_CLICKED(IDC_BUTTON_1V1_Identify, &CFXsDlg::OnBnClickedButton1v1Identify)
	ON_BN_CLICKED(IDC_BUTTON_Identify100, &CFXsDlg::OnBnClickedButtonIdentify100)
	ON_BN_CLICKED(IDC_BUTTON_1VN_Identify, &CFXsDlg::OnBnClickedButton1vnIdentify)
	ON_BN_CLICKED(IDC_BUTTON_QUIT, &CFXsDlg::OnBnClickedButtonQuit)
	ON_BN_CLICKED(IDC_BUTTON_QUITIDTFY, &CFXsDlg::OnBnClickedButtonQuitidtfy)
	ON_BN_CLICKED(IDC_BUTTON_RED, &CFXsDlg::OnBnClickedButtonRed)
	ON_BN_CLICKED(IDC_BUTTON_GREEN, &CFXsDlg::OnBnClickedButtonGreen)
	ON_BN_CLICKED(IDC_BUTTON_BUZZER, &CFXsDlg::OnBnClickedButtonBuzzer)
	ON_BN_CLICKED(IDC_BUTTON_RED2, &CFXsDlg::OnBnClickedButtonRed2)
	ON_BN_CLICKED(IDC_BUTTON_GREEN2, &CFXsDlg::OnBnClickedButtonGreen2)
	ON_EN_CHANGE(IDC_EDIT1, &CFXsDlg::OnEnChangeEdit1)
	ON_EN_CHANGE(IDC_EDIT2, &CFXsDlg::OnEnChangeEdit2)
	ON_BN_CLICKED(IDC_BUTTON_DEL, &CFXsDlg::OnBnClickedButtonDel)
	ON_BN_CLICKED(IDC_BUTTON_DEL_ALL, &CFXsDlg::OnBnClickedButtonDelAll)
	ON_BN_CLICKED(IDC_BUTTON_LOAD, &CFXsDlg::OnBnClickedButtonLoad)
	ON_BN_CLICKED(IDC_BUTTON_BUILDS, &CFXsDlg::OnBnClickedButtonBuilds)
	ON_BN_CLICKED(IDC_CHECK_USER, &CFXsDlg::OnBnClickedCheckUser)
	ON_BN_CLICKED(IDC_CHECK_FINGER, &CFXsDlg::OnBnClickedCheckFinger)
	ON_BN_CLICKED(IDC_BUTTON_CFG, &CFXsDlg::OnBnClickedButtonCfg)
	ON_EN_CHANGE(IDC_EDIT_UseNameID, &CFXsDlg::OnEnChangeEditUsenameid)
	ON_EN_CHANGE(IDC_EDIT4, &CFXsDlg::OnEnChangeEdit4)
	ON_BN_CLICKED(IDC_BUTTON_SAVE, &CFXsDlg::OnBnClickedButtonSave)
	ON_BN_CLICKED(IDC_BUTTON_IMG_TEST, &CFXsDlg::OnBnClickedButtonImgTest)
	ON_BN_CLICKED(IDC_BUTTON_IMG_TEST2, &CFXsDlg::OnBnClickedButtonImgTest2)
END_MESSAGE_MAP()


// CFXsDlg message handlers
BOOL CFXsDlg::OnInitDialog()
{
	CString tmp;
	CString show;
	CImage  image;
	CDialogEx::OnInitDialog();

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	SetWindowText(_T("FXs(off-line)"));

	initControls();
	thread_handle = 0;

	SetDlgItemInt(IDC_EDIT4, 3);

	CFingerVeinV3::getInstance();

//	CFingerVeinV3::getInstance()->setDebug(10);

	memset(&conf_info, 0, sizeof(conf_info));
	conf_info.deviceId = -100;
	openDevIndex = -1;

	int cnt = CFingerVeinV3::getInstance()->getDevCount();
	m_iDeviceCount = cnt;
	if (0 != m_iDeviceCount)
	{
		initDevlist(m_iDeviceCount);
	}
	// 注意
	DEV_BROADCAST_DEVICEINTERFACE Filter;
	ZeroMemory(&Filter, sizeof(Filter));
	Filter.dbcc_size = sizeof(Filter);   // size gets set to 29 with 1-byte packing or 32 with 4- or 8-byte packing
	Filter.dbcc_devicetype = DBT_DEVTYP_DEVICEINTERFACE;
	Filter.dbcc_classguid = MediumChangerClassGuid;
	//DEVICE_NOTIFY_ALL_INTERFACE_CLASSES //关注所有设备事件
	m_hDeviceNotify = RegisterDeviceNotification(this->m_hWnd, &Filter, DEVICE_NOTIFY_ALL_INTERFACE_CLASSES);
	if (NULL == m_hDeviceNotify)
		TRACE("RegisterDeviceNotification failed!!");


	m_editFont.CreatePointFont(180, (CString)"宋体");
//	m_edUserName.SetFont(&m_editFont); // 设置新字体
	m_edUserName.SetLimitText(NameLen);
//	m_edit1V1ID.SetFont(&m_editFont);
	m_edit1V1ID.SetLimitText(NameLen);

	char exeFullPath[MAX_PATH] = { 0 }; // MAX_PATH
	GetModuleFileNameA(NULL, exeFullPath, MAX_PATH);
	char drive[_MAX_DRIVE] = {0};
	char dir[_MAX_DIR] = { 0 };
	char filename[100] = { 0 };
	char fileext[10] = { 0 };
	_splitpath_s(exeFullPath, drive, dir, filename, fileext);
	strcpy_s(m_strAppPath, drive);
	strcat_s(m_strAppPath, dir);
	strcpy_s(m_strModelPath, m_strAppPath);
	strcat_s(m_strModelPath, MODEL);
	_mkdir(m_strModelPath);

	strcpy_s(m_strPhotoPath, m_strAppPath);
	strcat_s(m_strPhotoPath, "good_photos");
	_mkdir(m_strPhotoPath);

	strcpy_s(m_strErrPhotoPath, m_strAppPath);
	strcat_s(m_strErrPhotoPath, "bad_photos");
	_mkdir(m_strErrPhotoPath);

	CStringA filepath;
	filepath.Format("%smodel_GA\\photos\\", m_strAppPath);
	_mkdir(filepath);
	SetDlgItemTextA(m_hWnd, IDC_EDIT_PATH, filepath);

	filepath.Format("%smodel_GA\\model\\", m_strAppPath);
	SetDlgItemTextA(m_hWnd, IDC_EDIT_PATH2, filepath);

	filepath.Format("%smodel_GA\\photos\\male", m_strAppPath);
	_mkdir(filepath);

	filepath.Format("%smodel_GA\\photos\\female", m_strAppPath);
	_mkdir(filepath);

	Load();
	CString  sText;
	unsigned long tk = clock();
	user.load();
	tmp.LoadString(NULL, IDS_TIME_OF_TEMPLATE_LOADING, 0);
	sText.Format(_T("%s:%u"), tmp, clock() - tk);
	outPrint(sText);

	SetDlgItemInt(IDC_EDIT1, 100);
	SetDlgItemInt(IDC_EDIT2, 2);

	((CButton*)GetDlgItem(IDC_RADIO1))->SetCheck(TRUE);

	updateUserCount();

	relayout();
	return TRUE;  // return TRUE  unless you set the focus to a control
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CFXsDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);

	}
	else
	{
		CDialogEx::OnPaint();
	}
}

BOOL CFXsDlg::OnWndMsg(UINT message, WPARAM wParam, LPARAM lParam, LRESULT* pResult)
{
	// TODO: 在此添加专用代码和/或调用基类
	static int rm_flag = -1;
	static int in_flag = -1;
	PDEV_BROADCAST_HDR devHdr;
	PDEV_BROADCAST_DEVICEINTERFACE devInterface;
	int devCount = 0;
	
	if (message == WM_DEVICECHANGE)
	{
		if (DBT_DEVICEREMOVECOMPLETE == wParam || DBT_DEVICEARRIVAL == wParam){
			devCount = CFingerVeinV3::getInstance()->getDevCount();
			devHdr = (PDEV_BROADCAST_HDR)((DWORD)(lParam));

			if (devHdr->dbch_devicetype == DBT_DEVTYP_DEVICEINTERFACE){
				devInterface = (PDEV_BROADCAST_DEVICEINTERFACE)(devHdr);
				CString str, csTemp;
				str = CString(devInterface->dbcc_name);
				AfxExtractSubString(csTemp, (LPCTSTR)str, 1, '#'); 

				if (csTemp != _T("VID_18D1&PID_BBBB") && csTemp != _T("VID_18D1&PID_DDDD"))
					return CDialogEx::OnWndMsg(message, wParam, lParam, pResult);

			//	str.Format(_T("USB Device: %s"), csTemp);
			//	outPrint(str);
			}
			else
				return CDialogEx::OnWndMsg(message, wParam, lParam, pResult);
		}
		if (DBT_DEVICEREMOVECOMPLETE == wParam) {
			if (m_iDeviceCount != devCount)
			{
				outPrint(_T("USB Device remove"), TRUE);
				m_iDeviceCount = devCount;
				initDevlist(m_iDeviceCount);
			}
			if (rm_flag != wParam) {
				rm_flag = wParam;
				in_flag = -1;

			}
		}
		else if (DBT_DEVICEARRIVAL == wParam) {
			if (m_iDeviceCount != devCount)
			{
				if (devCount){
					outPrint(_T("USB Device insert"), TRUE);
					m_iDeviceCount = devCount;
					initDevlist(m_iDeviceCount);
				}
				else{
			//		outPrint(_T("USB Device insert 0"), TRUE);
				}
			}
			if (in_flag != wParam) {
				in_flag = wParam;
				rm_flag = -1;

			}
		}
	}
	else if (message == WM_SIZE)
	{
		relayout();
	}
	return CDialogEx::OnWndMsg(message, wParam, lParam, pResult);
}

void CFXsDlg::relayout(){
	if (::IsWindow(m_stPreview.GetSafeHwnd())){
		::SetWindowPos(m_stPreview.GetSafeHwnd(), NULL, 0, 0, 640, 480, SWP_NOMOVE);
	}
}



void CFXsDlg::OnDeadChar(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	// TODO: Add message handler code and/or call defaults here

	CDialogEx::OnDeadChar(nChar, nRepCnt, nFlags);
}

void CFXsDlg::OnCancel()
{
//	exitThread();
	doCloseDev();
	__super::OnCancel();
//	CFingerVeinV3::getInstance()->closeDevEx();
}


void CFXsDlg::enableIdentifyButtons(bool bEnable)
{
	GetDlgItem(IDC_BUTTON_1V1_Identify)->EnableWindow(bEnable);
	GetDlgItem(IDC_BUTTON_1VN_Identify)->EnableWindow(bEnable);
	GetDlgItem(IDC_BUTTON_Identify100)->EnableWindow(bEnable);
	GetDlgItem(IDC_BUTTON_IMG_TEST)->EnableWindow(bEnable);
	GetDlgItem(IDC_BUTTON_IMG_TEST2)->EnableWindow(bEnable);

	if (bEnable){
		CString tmp;
		tmp.LoadString(NULL, IDS_IDTFY_IMG_TEST, 0);
		SetDlgItemText(IDC_BUTTON_IMG_TEST, tmp);

		tmp.LoadString(NULL, IDS_IDTFY_IMG_TEST2, 0);
		SetDlgItemText(IDC_BUTTON_IMG_TEST2, tmp);

		tmp.LoadString(NULL, IDS_1V1_IDTFY);
		m_Button_1V1_Identify.SetWindowText(tmp);

		tmp.LoadString(NULL, IDS_1VN_IDTFY);
		m_Button_1VN_Identify.SetWindowText(tmp);

		tmp.LoadString(NULL, IDS_IDTFY_100);
		m_Button_Identify100.SetWindowText(tmp);
	}
}
void CFXsDlg::initControls()
{
	//init user name
	memset(m_userName, 0, NameLen*sizeof(char));
	CString tmp;
	tmp.LoadString(NULL, IDS_NAME);
	m_staticuserName.SetWindowText(tmp);

	tmp.LoadString(NULL, IDS_OPEN);
	m_ButtonOpenDev.SetWindowText(tmp);

	tmp.LoadString(NULL, IDS_STARTING_MODELING);
	m_buildModule.SetWindowText(tmp);
	GetDlgItem(IDC_BUTTON_BuildModule)->EnableWindow(FALSE);

	tmp.LoadString(NULL, IDS_STARTING_SAVE_IMG);
	m_saveImg.SetWindowText(tmp);
	m_saveImg.EnableWindow(FALSE);

	tmp.LoadString(NULL, IDS_1V1_IDTFY);
	m_Button_1V1_Identify.SetWindowText(tmp);
	
	tmp.LoadString(NULL, IDS_1VN_IDTFY);
	m_Button_1VN_Identify.SetWindowText(tmp);
	
	tmp.LoadString(NULL, IDS_IDTFY_100);
	m_Button_Identify100.SetWindowText(tmp);

	tmp.LoadString(NULL, IDS_QUITIDENTIFY);
	m_buttonQuitIdentify.SetWindowText(tmp);
	GetDlgItem(IDC_BUTTON_QUITIDTFY)->EnableWindow(FALSE);
	GetDlgItem(IDC_EDIT_1V1ID)->EnableWindow(FALSE);
	GetDlgItem(IDC_EDIT_1V1ID)->EnableWindow(FALSE);
	enableIdentifyButtons();
}

int   MessageLoop(
	HANDLE*   lphObjects,     //handles that need to be waited   on   
	int           cObjects           //   number   of   handles   to   wait   on   
	)
{
	//   The   message   loop   lasts until we get a WM_QUIT message, 
	// upon which we shall return from the function. 
	while (TRUE)
	{
		// block-local variable 
		DWORD result;
		MSG msg;

		// Read all of the messages in this next loop, 
		// removing each message as we read it. 
		while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			// If it's a quit message, we 're out of here. 
			if (msg.message == WM_QUIT)
				return 1;
			// Otherwise, dispatch the message. 
			DispatchMessage(&msg);
		} // End of PeekMessage while loop. 

		// Wait for any message sent or posted to this queue 
		// or for one of the passed handles be set to signaled. 
		result = MsgWaitForMultipleObjects(cObjects, lphObjects,
			FALSE, INFINITE, QS_ALLINPUT);

		// The result tells us the type of event we have. 
		if (result == (WAIT_OBJECT_0 + cObjects))
		{
			// New messages have arrived. 
			// Continue to the top of the always while loop to 
			// dispatch them and resume waiting. 
			continue;
		}
		else
		{
			// One of the handles became signaled. 
			return 0;
		} // End of else clause. 
	} // End of the always while loop. 
	return 1;
} // End of function.  

void  CFXsDlg::exitThread()
{
	static bool brunning = false;
	if (brunning)
		return;
	brunning = true;
	m_bTerminate = TRUE;
	if (thread_handle)
	{
		MessageLoop(&thread_handle, 1);
		//WaitForSingleObject(thread_handle,INFINITE);
		thread_handle = 0;
	}

	brunning = false;
}
/*
unsigned CFXsDlg::ExecuteThread(void* pArguments)
{
CFXsDlg *pThis = reinterpret_cast<CFXsDlg*>(pArguments);
pThis->thread_update_pic();
return 0;
}

void CFXsDlg::thread_update_pic()
{
int state = 0;
int rows = 240;
int cols = 320;
int id = 0;
bool bRet = false;
setWork(FALSE);

while (!m_bTerminate)
{
CStringA str1 = (char*)m_userName;
CStringW str2 = CStrA2CStrW(str1);
CString strAll;
strAll.Format(_T("%s Put your finger in to start modeling %d\r\n"), str2, getCmd(TRUE));
outPrint(strAll);
Sleep(100);
}
}
*/


void CFXsDlg::outPrint(CString strText, BOOL SetColor)
{
	CHARFORMAT cFmt;
	cFmt.cbSize = sizeof(CHARFORMAT);
	if (SetColor)
	{
		cFmt.crTextColor = RGB(255, 0, 0);
		cFmt.dwEffects |= CFE_BOLD;
	}
	else
	{
		cFmt.crTextColor = RGB(64, 64, 64);
		cFmt.dwEffects &= ~CFE_BOLD;
	}
	cFmt.dwMask = CFM_BOLD | CFM_COLOR | CFM_SIZE;
	_tcscpy_s(cFmt.szFaceName, _T("宋体"));
	cFmt.yHeight = 11 * 20;
	CTime time = CTime::GetCurrentTime();
	CString strcurTime = time.Format(_T("%H:%M:%S"));
	CString szText;
	CHARRANGE  SelDate;
	SelDate.cpMax = -1;
	SelDate.cpMin = -1;
	
	m_edOut.SetSel(SelDate);
	szText.Format(_T("%s: %s \r\n"), strcurTime, strText);
	BOOL a = m_edOut.SetSelectionCharFormat(cFmt);//set colors 
	m_edOut.ReplaceSel(szText);
	m_edOut.SetSel(SelDate);
	m_edOut.PostMessage(WM_VSCROLL, SB_BOTTOM, 0);	

}
//model_info * CFXsDlg::getModelInfo(char *userName)
//{
//	return 0;
//}


void CFXsDlg::save()
{
	{
		char path[255] = "";
		strcpy_s(path, m_strModelPath);
		strcat_s(path, "\\fingervein.vdb");
		OutputDebugStringA("open file end...");
		errno_t err;
		FILE   *fp = NULL;
		err = fopen_s(&fp, path, "wb+");
		if (!err){
			int modelnum = m_vecModellist.size();
			fwrite(&modelnum, sizeof(modelnum), 1, fp);
			int modelsize = 3352;
			fwrite(&modelsize, sizeof(modelsize), 1, fp);
			int i = 0;
			while (i < modelnum)
			{
				fwrite(&m_vecModellist[i][0], sizeof(BYTE)*modelsize, 1, fp);
				++i;
			}

			fclose(fp);
			OutputDebugStringA("open file end..close ok.\n");
		}
		else{
			printf("save user finger vein error.\n");
		}

	/*	char Namepath[255] = "";
		strcpy_s(Namepath, m_strModelPath);
		strcat_s(Namepath, "\\fingerveinName.vdb");
		err = fopen_s(&fp, Namepath, "wb+");
		if (!err){
			for (int i = 0; i<NameNum; i++)
			{
				if (strlen(m_currentModel.name[i]) == 0)
				{
					break;
				}
				else
				{
					fwrite(&m_currentModel.name[i], NameLen, 1, fp);
				}
			}
			fclose(fp);
		}
		else{
		}*/
	}
}

void CFXsDlg::Load()
{
#ifndef GA
	char path[255] = "";
	strcpy_s(path, m_strModelPath);
	strcat_s(path, "\\fingervein.vdb");
	errno_t err;
	FILE   *fp = NULL;
	err = fopen_s(&fp, path, "rb");
	if (!err){
		
		int num = -1;
		fread(&num, sizeof(num), 1, fp);
		int modelsize = 0;
		fread(&modelsize, sizeof(modelsize), 1, fp);
		int i = 0;
		while (modelsize > 0 && i < num)
		{
			vector<BYTE> mi;
			mi.resize(modelsize);
			fread(&mi[0], sizeof(BYTE)*modelsize, 1, fp);
			++i;
			m_vecModellist.push_back(mi);
		}

		if (num > 0){
			m_currentModel.G_Mode = m_vecModellist[num -1];
			m_bHaveRecordsofBuildModule = TRUE;
		}
		fclose(fp);

	}
	else{

	}
/*	char Namepath[255] = "";
	strcpy_s(Namepath, m_strModelPath);
	strcat_s(Namepath, "\\fingerveinName.vdb");
	err = fopen_s(&fp, Namepath, "rb");
	if (!err)
	{
		for (int i = 0; i<NameNum; i++)
		{
			fread(&m_currentModel.name[i], NameLen, 1, fp);
			if (strlen(m_currentModel.name[i]) == 0)
			{
				//记住之前保存码
				StartNum = i;
				break;
			}
		}
		m_bHaveRecordsofBuildModule |= TRUE;
		fclose(fp);
	}*/
#else
	char path[255] = "";
	strcpy_s(path, m_strModelPath);
	strcat_s(path, "\\feature_ga.dat");
	errno_t err;
	FILE   *fp = NULL;
	err = fopen_s(&fp, path, "rb");
	if (!err){
		m_bHaveRecordsofBuildModule = TRUE;
		fclose(fp);
	}
#endif
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CFXsDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


void CFXsDlg::initDevlist(int cnt)
{
	int index = 0;
	m_cbDevs.ResetContent();

	if (openDevIndex >= 0)
		CFingerVeinV3::getInstance()->closeDevEx();

	for (int i = 0; i<cnt; i++)
	{
		CString tmp;
		CString sText;
		tmp.LoadString(NULL, IDS_DEVICE, 0);
		sText.Format(_T("%s%d"), tmp, i);

		//yejs, Open the device, obtain the device ID and display it in the device list
		if (CFingerVeinV3::getInstance()->openDevEx(i, m_stPreview.GetSafeHwnd())){
			configure_info conf_info;
			CFingerVeinV3::getInstance()->Dev_GetConfig(&conf_info);
			if (-1 != conf_info.deviceId)
				sText.Format(_T("%s%d"), tmp, conf_info.deviceId);

			if (this->conf_info.deviceId != conf_info.deviceId)
				CFingerVeinV3::getInstance()->closeDevEx();
			else
				index = i;
		}

		m_cbDevs.AddString(sText);
	}
	m_cbDevs.SetCurSel(index);
}

void CFXsDlg::doOpenDev()
{
	CString tmp;
	CString show;

	int dev = m_cbDevs.GetCurSel();
	if (dev<0)
	{
		return;
	}

	CString str;
	str.Format(_T("%d"), m_stPreview.GetSafeHwnd());

	if (CFingerVeinV3::getInstance()->openDevEx(dev, m_stPreview.GetSafeHwnd())){

	/*	SetTimer(1002, 1500, [](HWND hWnd, UINT nMsg, UINT_PTR nIDEvent, DWORD dwTime) -> void {
			CWnd::FromHandle(hWnd)->KillTimer(nIDEvent);
			CFXsDlg *pDlg = ((CFXsDlg*)CWnd::FromHandle(hWnd));

		//	CFingerVeinV3::getInstance()->Dev_GetConfig(&pDlg->conf_info);
			bool flag = false;
			for (int id = 0; id <= 8; id++){
				if (pDlg->conf_info.onLight[id] != 20){
					flag = true;
					break;
				}
			}
			if (!flag){
				if (pDlg->conf_info.securityLevel != 610 || pDlg->conf_info.LowThreshold != 60 || pDlg->conf_info.HighThreshold != 90 || pDlg->conf_info.defaultSessor != 80 || pDlg->conf_info.level != 2 ||
					abs(pDlg->conf_info.fPressThreshold - 0.57) > 0.00001 || abs(pDlg->conf_info.modelScore - 0.28) > 0.00001 || abs(pDlg->conf_info.fEnergyThreshold - 0.058) > 0.00001){
					flag = true;
				}
			}

			if (1){
			//	for (int id = 0; id <= 8; id++)
			//		pDlg->conf_info.onLight[id] = 20;

				pDlg->conf_info.securityLevel = 610;
				pDlg->conf_info.LowThreshold = 60;
				pDlg->conf_info.HighThreshold = 90;
				pDlg->conf_info.defaultSessor = 80;
				pDlg->conf_info.level = 2;


				pDlg->conf_info.fPressThreshold = (float)0.57;
				pDlg->conf_info.modelScore = (float)0.28;
				pDlg->conf_info.fEnergyThreshold = (float)0.058;
				
			//	CFingerVeinV3::getInstance()->Dev_SetConfig(&pDlg->conf_info);
			}
			
		//	CFingerVeinV3::getInstance()->Dev_SetConfig(&pDlg->conf_info);
		});
		*/
		//CFingerVeinV3::getInstance()->setNo(10);
		tmp.LoadString(NULL, IDS_OPEN_SUCCESS);
		outPrint(tmp);
		//m_stPreview.SetWindowPos(0, 0, 0, NOR_WID, NOR_HEIGHT, SWP_NOMOVE | SWP_NOZORDER);
		m_stPreview.UpdateWindow();
	//	CFingerVeinV3::getInstance()->setLedEx(CFingerVeinV3::COLOR_GREEN, 0);
		GetDlgItem(IDC_BUTTON_BuildModule)->EnableWindow(TRUE);
		if (m_bHaveRecordsofBuildModule)
		{
			enableIdentifyButtons(TRUE);
		}
		GetDlgItem(IDC_EDIT_1V1ID)->EnableWindow(TRUE);
		GetDlgItem(IDC_BUTTON_CFG)->EnableWindow(TRUE);
		GetDlgItem(IDC_CHECK_USER)->EnableWindow(TRUE);
		GetDlgItem(IDC_CHECK_FINGER)->EnableWindow(TRUE);
		GetDlgItem(IDC_BUTTON_DEL_ALL)->EnableWindow(TRUE);
		GetDlgItem(IDC_BUTTON_SAVE)->EnableWindow(TRUE);

		Sleep(20);
		CFingerVeinV3::getInstance()->setLedEx(CFingerVeinV3::COLOR_RED, 0);
		Sleep(20);
		CFingerVeinV3::getInstance()->setLedEx(CFingerVeinV3::COLOR_GREEN, 0);
		Sleep(20);
		CFingerVeinV3::getInstance()->setBuzzer(1, 1);
#if MD120
		CFingerVeinV3::getInstance()->openBuzzer(1, 1);
#endif

		CString tmp;
		tmp.LoadString(NULL, IDS_CLOSE, 0);
		this->GetDlgItem(IDC_BUTTON_OpenDev)->SetWindowText(tmp);
		Sleep(20);
		CFingerVeinV3::getInstance()->openBackLightEx();
		Sleep(20);
		CFingerVeinV3::getInstance()->setLedEx(CFingerVeinV3::COLOR_RED_FLUSH, 1);

		CFingerVeinV3::getInstance()->Dev_GetConfig(&conf_info);
		openDevIndex = dev;

		((CButton*)GetDlgItem(IDC_CHECK_USER))->SetCheck(!conf_info.bCheckDup);
		((CButton*)GetDlgItem(IDC_CHECK_FINGER))->SetCheck(conf_info.bCheckSameFv);

		SetWindowText(_T("FXs(On-line)"));
	}
	else{
		tmp.LoadString(NULL, IDS_OPEN_FAIL, 0);
		outPrint(tmp, TRUE);
	//	CFingerVeinV3::getInstance()->setLedEx(CFingerVeinV3::COLOR_RED_FLUSH, 1);
	}
}

void CFXsDlg::OnBnClickedCheckUser()
{
	conf_info.bCheckDup = !((CButton*)GetDlgItem(IDC_CHECK_USER))->GetCheck();

	CFingerVeinV3::getInstance()->Dev_SetConfig(&conf_info);
}


void CFXsDlg::OnBnClickedCheckFinger()
{
	conf_info.bCheckSameFv = ((CButton*)GetDlgItem(IDC_CHECK_FINGER))->GetCheck();

	CFingerVeinV3::getInstance()->Dev_SetConfig(&conf_info);
}

void CFXsDlg::doCloseDev()
{
	CString tmp;
	exitThread();

	CFingerVeinV3::getInstance()->closeBackLightEx();
	Sleep(20);
	CFingerVeinV3::getInstance()->setLedEx(CFingerVeinV3::COLOR_GREEN, 0);
//	CFingerVeinV3::getInstance()->setBuzzer(1);
	if (CFingerVeinV3::getInstance()->closeDevEx()){
		tmp.LoadString(NULL, IDS_OPEN, 0);
		SetDlgItemText(IDC_BUTTON_OpenDev, tmp);
		GetDlgItem(IDC_BUTTON_BuildModule)->EnableWindow(FALSE);
		GetDlgItem(IDC_BUTTON_1V1_Identify)->EnableWindow(FALSE);
		GetDlgItem(IDC_BUTTON_1VN_Identify)->EnableWindow(FALSE);
		GetDlgItem(IDC_BUTTON_Identify100)->EnableWindow(FALSE);

		GetDlgItem(IDC_BUTTON_CFG)->EnableWindow(FALSE);
		GetDlgItem(IDC_CHECK_USER)->EnableWindow(FALSE);
		GetDlgItem(IDC_CHECK_USER)->EnableWindow(FALSE);
		GetDlgItem(IDC_CHECK_FINGER)->EnableWindow(FALSE);
		GetDlgItem(IDC_BUTTON_DEL_ALL)->EnableWindow(FALSE);
		GetDlgItem(IDC_BUTTON_SAVE)->EnableWindow(FALSE);

		memset(&conf_info, 0, sizeof(conf_info));
		conf_info.deviceId = -100;
		openDevIndex = -1;

		SetWindowText(_T("FXs(Off-line)"));

		tmp.LoadString(NULL, IDS_CLOSE_SUCCESS, 0);
		outPrint(tmp);
	}
	else{
		tmp.LoadString(NULL, IDS_CLOSE_FAIL, 0);
		outPrint(tmp, TRUE);
	}
}


bool EnumerateCameras(vector<int> &camIdx)
{
	camIdx.clear();
	struct CapDriver{
		int enumValue; string enumName; string comment;
	};
	// list of all CAP drivers (see highgui_c.h)
	vector<CapDriver> drivers;
	drivers.push_back({ CV_CAP_MIL, "CV_CAP_MIL", "MIL proprietary drivers" });
	drivers.push_back({ CV_CAP_VFW, "CV_CAP_VFW", "platform native" });
	drivers.push_back({ CV_CAP_FIREWARE, "CV_CAP_FIREWARE", "IEEE 1394 drivers" });
	drivers.push_back({ CV_CAP_STEREO, "CV_CAP_STEREO", "TYZX proprietary drivers" });
	drivers.push_back({ CV_CAP_QT, "CV_CAP_QT", "QuickTime" });
	drivers.push_back({ CV_CAP_UNICAP, "CV_CAP_UNICAP", "Unicap drivers" });
	drivers.push_back({ CV_CAP_DSHOW, "CV_CAP_DSHOW", "DirectShow (via videoInput)" });
//	drivers.push_back({ CV_CAP_MSMF, "CV_CAP_MSMF", "Microsoft Media Foundation (via videoInput)" });
	drivers.push_back({ CV_CAP_PVAPI, "CV_CAP_PVAPI", "PvAPI, Prosilica GigE SDK" });
	drivers.push_back({ CV_CAP_OPENNI, "CV_CAP_OPENNI", "OpenNI (for Kinect)" });
	drivers.push_back({ CV_CAP_OPENNI_ASUS, "CV_CAP_OPENNI_ASUS", "OpenNI (for Asus Xtion)" });
	drivers.push_back({ CV_CAP_ANDROID, "CV_CAP_ANDROID", "Android" });
//	drivers.push_back({ CV_CAP_ANDROID_BACK, "CV_CAP_ANDROID_BACK", "Android back camera" }),
//		drivers.push_back({ CV_CAP_ANDROID_FRONT, "CV_CAP_ANDROID_FRONT", "Android front camera" }),
		drivers.push_back({ CV_CAP_XIAPI, "CV_CAP_XIAPI", "XIMEA Camera API" });
	drivers.push_back({ CV_CAP_AVFOUNDATION, "CV_CAP_AVFOUNDATION", "AVFoundation framework for iOS" });
	drivers.push_back({ CV_CAP_GIGANETIX, "CV_CAP_GIGANETIX", "Smartek Giganetix GigEVisionSDK" });
//	drivers.push_back({ CV_CAP_INTELPERC, "CV_CAP_INTELPERC", "Intel Perceptual Computing SDK" });

	std::string winName, driverName, driverComment;
	int driverEnum;
	cv::Mat frame;
	bool found;
	std::cout << "Searching for cameras IDs..." << std::endl;
	for (int drv = 0; drv < drivers.size(); drv++)
	{
		driverName = drivers[drv].enumName;
		driverEnum = drivers[drv].enumValue;
		driverComment = drivers[drv].comment;
		std::cout << "Testing driver " << driverName << "...";
		found = false;

		int maxID = 100; //100 IDs between drivers
		if (driverEnum == CV_CAP_VFW)
			maxID = 10; //VWF opens same camera after 10 ?!?
		else if (driverEnum == CV_CAP_ANDROID)
			maxID = 98; //98 and 99 are front and back cam
	//	else// if ((driverEnum == CV_CAP_ANDROID_FRONT) || (driverEnum == CV_CAP_ANDROID_BACK))
	//		maxID = 1;

		for (int idx = 0; idx <maxID; idx++)
		{
			cv::VideoCapture cap(driverEnum + idx);  // open the camera
			if (cap.isOpened())                  // check if we succeeded
			{
				found = true;
				camIdx.push_back(driverEnum + idx);  // vector of all available cameras
				cap >> frame;
				if (frame.empty())
					std::cout << std::endl << driverName << "+" << idx << "\t opens: OK \t grabs: FAIL";
				else
					std::cout << std::endl << driverName << "+" << idx << "\t opens: OK \t grabs: OK";
				// display the frame
				// imshow(driverName + "+" + to_string(idx), frame); waitKey(1);
			}
			cap.release();
		}
		if (!found) std::cout << "Nothing !" << std::endl;
		std::cout << std::endl;
	}
	std::cout << camIdx.size() << " camera IDs has been found ";
	std::cout << "Press a key..." << std::endl; 

	return (camIdx.size()>0); // returns success
}


void CFXsDlg::OnBnClickedButtonOpendev()
{
//	vector<int> cam_vec;
//	EnumerateCameras(cam_vec);
/*	bool ret = video.open(701);
	if (!video.isOpened())
	{
		return;
	}

	//Detects the size of the video
	cv::Size videoSize = cv::Size((int)video.get(CV_CAP_PROP_FRAME_WIDTH), (int)video.get(CV_CAP_PROP_FRAME_HEIGHT));

	//A window that displays each frame
	cv::namedWindow("video demo", CV_WINDOW_AUTOSIZE);

	GetDlgItem(IDC_BUTTON_SAVE)->EnableWindow(TRUE);

	return;*/
	// TODO: Add your control notification handler code here
	CString tmp;
	CString strBtnTex= _T("");
	this->GetDlgItem(IDC_BUTTON_OpenDev)->GetWindowText(strBtnTex);
	tmp.LoadString(NULL, IDS_OPEN, 0);
	if (strBtnTex.CompareNoCase(tmp) == 0)
	{
	    doOpenDev();
	}
	else
	{
		doCloseDev();
	}
}


void CFXsDlg::OnBnClickedButtonBuildmodule()
{
	// TODO: Add your control notification handler code here
	CString  stopBtn = _T("");
	stopBtn.LoadString(NULL, IDS_STOP_MODELING, 0);
	CString sText = _T("");
	GetDlgItem(IDC_BUTTON_BuildModule)->GetWindowText(sText);

	exitThread();
	if (stopBtn.CompareNoCase(sText) == 0)
	{
		CString str_stop = _T("");
		CString show = _T("");
		str_stop.LoadString(NULL, IDS_STOP, 0);
		show.Format(_T("%s "), str_stop);
		str_stop.LoadString(NULL, IDS_MODELING, 0);
		show.Append(str_stop);
		outPrint(show);
		//Switch to start modeling
		CString  StartBtn = _T("");
		StartBtn.LoadString(NULL, IDS_STARTING_MODELING, 0);
		GetDlgItem(IDC_BUTTON_BuildModule)->SetWindowText(StartBtn);
		return;
	}
	memset(m_userName, 0, NameLen*sizeof(char));
	::GetDlgItemText(m_hWnd, m_edUserName.GetDlgCtrlID(), m_userName, sizeof(m_userName));
	if (_tcslen(m_userName) == 0){
		CString tmp;
		tmp.LoadString(NULL, IDS_NAME_OF_USER_CONNOT_BE_NULL);
		outPrint(tmp, TRUE);
		return;
	}
	vector<int> ids;
	if (conf_info.bCheckDup && user.getIdByUser((LPCSTR)CStringA(m_userName), ids)>0)
	{
		outPrint(_T("The user name has been registered, please re-register"), TRUE);
		return;
	}

	m_bTerminate = FALSE;
	(GetDlgItem(IDC_BUTTON_BuildModule))->SetWindowText(stopBtn);

	thread_handle = (HANDLE)_beginthread(model_process, 0, this);
}

void CFXsDlg::OnBnClickedButtonDel()
{
	memset(m_userName, 0, NameLen*sizeof(char));
	::GetDlgItemText(m_hWnd, m_edUserName.GetDlgCtrlID(), m_userName, sizeof(m_userName));
	if (_tcslen(m_userName) == 0){
		CString tmp;
		tmp.LoadString(NULL, IDS_NAME_OF_USER_CONNOT_BE_NULL);
		outPrint(tmp, TRUE);
		return;
	}
	vector<int> ids, indexs;
	int id = user.getIdByUser((LPCSTR)CStringA(m_userName), ids);
	int index = user.getIndexByUser((LPCSTR)CStringA(m_userName), indexs);

	CString tmp;
	tmp.Format(_T("id:%d, index:%d"), id, index);
	outPrint(tmp);

	for (auto id = ids.begin(); id != ids.end(); id++){
		user.delUser(*id);
	}

	auto it = m_vecModellist.begin();
	for (auto index_it = indexs.rbegin(); index_it != indexs.rend(); index_it++){
		int index = *index_it;
		
		m_vecModellist.erase(it + index);
	}
	updateUserCount();

	user.save();
	save();

	user.load();
}


void CFXsDlg::OnBnClickedButtonDelAll()
{
	auto it = m_vecModellist.begin();
	int modelsize = m_vecModellist.size();

	for (int i = 0; i < modelsize; i++)
	{
		m_vecModellist.erase(it);
	}

	m_vecModellist.clear();

	user.delAllUser();

	updateUserCount();

	save();

	user.load();

	conf_info.bCheckDup = TRUE;
	conf_info.bCheckSameFv = TRUE;
	CFingerVeinV3::getInstance()->Dev_SetConfig(&conf_info);
	((CButton*)GetDlgItem(IDC_CHECK_USER))->SetCheck(!conf_info.bCheckDup);
	((CButton*)GetDlgItem(IDC_CHECK_FINGER))->SetCheck(conf_info.bCheckSameFv);
}

void CFXsDlg::doFingerIdentify()
{
	/*
	enum{
	Idnt1V1,
	Idnt1VN,
	IdntAuto,
	IdntTest,
};*/

	memset(m_userName, 0, NameLen*sizeof(char));
	::GetDlgItemText(m_hWnd, m_edUserName.GetDlgCtrlID(), m_userName, sizeof(m_userName));

	CString  StartBtn = _T("");
	StartBtn.LoadString(NULL, IDS_STARTING_MODELING, 0);
	GetDlgItem(IDC_BUTTON_BuildModule)->SetWindowText(StartBtn);

	StartBtn.LoadString(NULL, IDS_STARTING_SAVE_IMG, 0);
	m_saveImg.SetWindowText(StartBtn);

	m_bTerminate = FALSE;
	if (IdntAuto == m_iIdentifyType)
	{
		 thread_handle = (HANDLE)_beginthreadex(NULL, 0, app_thread_identy_auto, this, 0, NULL);
	}
	else if (Idnt1V1 == m_iIdentifyType || Idnt1VN == m_iIdentifyType)
	{
		thread_handle = (HANDLE)_beginthreadex(NULL, 0, app_thread_identy, this, 0, NULL);
	}
	else if (IdntTest == m_iIdentifyType || IdntTest2 == m_iIdentifyType)
	{
		thread_handle = (HANDLE)_beginthreadex(NULL, 0, app_thread_test, this, 0, NULL);
	}
}

void CFXsDlg::OnBnClickedButton1v1Identify()
{
	// TODO: Add your control notification handler code here
	CString str_stop;
	str_stop.LoadString(NULL, IDS_IN_IDTFY_1V1, 0);
	CString _input1V1ID = _T("");
	m_edit1V1ID.GetWindowText(_input1V1ID);
	if (0 == _input1V1ID.GetLength())
	{
		MessageBox(_T("Please enter 1v1 authentication ID！"), _T("prompt"));
		return;
	}
	::GetDlgItemTextA(m_hWnd, IDC_EDIT_1V1ID, m_1v1userName, sizeof(m_1v1userName));

	
	bool _bret = getInput1V1Model(_input1V1ID);
	if (!_bret)
		return;

	m_iIdentifyType = Idnt1V1;

	exitThread();
		
	SetDlgItemText(IDC_BUTTON_1V1_Identify, str_stop);
	enableIdentifyButtons(FALSE);
	
	GetDlgItem(IDC_BUTTON_BuildModule)->EnableWindow(FALSE);
	GetDlgItem(IDC_EDIT_1V1ID)->EnableWindow(FALSE);
	GetDlgItem(IDC_BUTTON_QUITIDTFY)->EnableWindow(TRUE);
	GetDlgItem(IDC_STATIC_IDTFY_TIME)->SetWindowText(_T(""));

	GetDlgItem(IDC_BUTTON_SAVE)->EnableWindow(FALSE);
	GetDlgItem(IDC_BUTTON_IMG_TEST)->EnableWindow(FALSE);
	GetDlgItem(IDC_BUTTON_IMG_TEST2)->EnableWindow(FALSE);
	doFingerIdentify();
}


void CFXsDlg::OnBnClickedButtonIdentify100()
{
	// TODO: Add your control notification handler code here
	CString str_stop;
	str_stop.LoadString(NULL, IDS_IN_IDTFY_100, 0);
	m_iIdentifyType = IdntAuto;

	CString  strTmp;
	GetDlgItem(IDC_BUTTON_1V1_Identify)->GetWindowText(strTmp);

	exitThread();
	if (strTmp.CompareNoCase(str_stop) == 0)
		return;

	SetDlgItemText(IDC_BUTTON_Identify100, str_stop);
	enableIdentifyButtons(FALSE);

	GetDlgItem(IDC_BUTTON_BuildModule)->EnableWindow(FALSE);
	GetDlgItem(IDC_EDIT_1V1ID)->EnableWindow(FALSE);
	GetDlgItem(IDC_BUTTON_QUITIDTFY)->EnableWindow(TRUE);
	GetDlgItem(IDC_STATIC_IDTFY_TIME)->SetWindowText(_T(""));
	GetDlgItem(IDC_BUTTON_SAVE)->EnableWindow(FALSE);
	GetDlgItem(IDC_BUTTON_IMG_TEST)->EnableWindow(FALSE);
	GetDlgItem(IDC_BUTTON_IMG_TEST2)->EnableWindow(FALSE);
	doFingerIdentify();

}


void CFXsDlg::OnBnClickedButton1vnIdentify()
{
	// TODO: Add your control notification handler code here
	CString str_stop;
	str_stop.LoadString(NULL, IDS_IN_IDTFY_1VN, 0);
	m_iIdentifyType = Idnt1VN;

	CString  strTmp;
	GetDlgItem(IDC_BUTTON_1VN_Identify)->GetWindowText(strTmp);

	exitThread();
	if (strTmp.CompareNoCase(str_stop) == 0)
		return;

	SetDlgItemText(IDC_BUTTON_1VN_Identify, str_stop);
	enableIdentifyButtons(FALSE);

	GetDlgItem(IDC_BUTTON_BuildModule)->EnableWindow(FALSE);
	GetDlgItem(IDC_EDIT_1V1ID)->EnableWindow(FALSE);
	GetDlgItem(IDC_BUTTON_QUITIDTFY)->EnableWindow(TRUE);
	GetDlgItem(IDC_STATIC_IDTFY_TIME)->SetWindowText(_T(""));
	GetDlgItem(IDC_BUTTON_SAVE)->EnableWindow(FALSE);
	GetDlgItem(IDC_BUTTON_IMG_TEST)->EnableWindow(FALSE);
	GetDlgItem(IDC_BUTTON_IMG_TEST2)->EnableWindow(FALSE);
	doFingerIdentify();
}


void CFXsDlg::OnBnClickedButtonQuit()
{
	// TODO: Add your control notification handler code here
	OnCancel();
}


void CFXsDlg::OnBnClickedButtonQuitidtfy()
{
	// TODO: Add your control notification handler code here
	enableIdentifyButtons(TRUE);
	CString tmp = _T("");
	tmp.LoadString(NULL, IDS_1V1_IDTFY);
	m_Button_1V1_Identify.SetWindowText(tmp);

	tmp.LoadString(NULL, IDS_1VN_IDTFY);
	m_Button_1VN_Identify.SetWindowText(tmp);

	tmp.LoadString(NULL, IDS_IDTFY_IMG_TEST);
	GetDlgItem(IDC_BUTTON_IMG_TEST)->SetWindowText(tmp);

	tmp.LoadString(NULL, IDS_IDTFY_IMG_TEST2);
	GetDlgItem(IDC_BUTTON_IMG_TEST2)->SetWindowText(tmp);

	tmp.LoadString(NULL, IDS_IDTFY_100);
	m_Button_Identify100.SetWindowText(tmp);
	GetDlgItem(IDC_BUTTON_QUITIDTFY)->EnableWindow(FALSE);
	GetDlgItem(IDC_BUTTON_BuildModule)->EnableWindow(TRUE);
	GetDlgItem(IDC_EDIT_1V1ID)->EnableWindow(TRUE);
	GetDlgItem(IDC_BUTTON_SAVE)->EnableWindow(TRUE);
	GetDlgItem(IDC_BUTTON_IMG_TEST)->EnableWindow(TRUE);
	GetDlgItem(IDC_BUTTON_IMG_TEST2)->EnableWindow(TRUE);
	exitThread();

	
}

bool CFXsDlg::getInput1V1Model(CString str1v1ID)
{
	bool _bret = false;
	int _cur1V1Id = -1;
#ifndef GA
	
	for (int i = 0; i <= m_vecModellist.size(); i++)
	{
		if (str1v1ID.CompareNoCase((LPCTSTR)CString(user.getUserByIndex(i).c_str())) == 0)
		{
			//记住之前保存码
			_cur1V1Id = i;
			_bret = true;
			break;
		}
	}
#else
	if (user.isExist((LPCSTR)CStringA(str1v1ID)))
		_bret = true;
#endif
	if (!_bret)
	{
		MessageBox(_T("The entered 1v1 authentication ID does not exist！"), _T("prompt"));
		return _bret;
	}

	//memset(m_userName, 0, NameLen*sizeof(char));
	
	return _bret;
}


void CFXsDlg::OnBnClickedButtonRed()
{
	CString str;
	GetDlgItemText(IDC_BUTTON_RED, str);
	if (str == _T("red light on")){
		str = _T("green light on");
		SetDlgItemText(IDC_BUTTON_GREEN, str);
		str = _T("red light off");

		CFingerVeinV3::getInstance()->setLedEx(CFingerVeinV3::COLOR_RED, 255);
	}
	else if (str == _T("red light off")){
		str = _T("red light on");

		CFingerVeinV3::getInstance()->setLedEx(CFingerVeinV3::COLOR_RED, 0);
	}

	SetDlgItemText(IDC_BUTTON_RED, str);
}


void CFXsDlg::OnBnClickedButtonGreen()
{
	CString str;
	GetDlgItemText(IDC_BUTTON_GREEN, str);
	if (str == _T("green light on")){
		str = _T("red light on");
		SetDlgItemText(IDC_BUTTON_RED, str);
		str = _T("green light off");

		CFingerVeinV3::getInstance()->setLedEx(CFingerVeinV3::COLOR_GREEN, 255);
	}
	else if (str == _T("green light off")){
		str = _T("green light on");

		CFingerVeinV3::getInstance()->setLedEx(CFingerVeinV3::COLOR_GREEN, 0);
	}
	SetDlgItemText(IDC_BUTTON_GREEN, str);
}


void CFXsDlg::OnBnClickedButtonRed2()
{
	SetDlgItemText(IDC_BUTTON_RED, _T("red light on"));

	SetDlgItemText(IDC_BUTTON_GREEN, _T("green light on"));

	BYTE time = (BYTE)(getFlushTime() / 100);

	CFingerVeinV3::getInstance()->setLedEx(CFingerVeinV3::COLOR_GREEN, 0);
	CFingerVeinV3::getInstance()->setLedEx(CFingerVeinV3::COLOR_RED | (time << 4), GetDlgItemInt(IDC_EDIT2));
}


void CFXsDlg::OnBnClickedButtonGreen2()
{
	SetDlgItemText(IDC_BUTTON_RED, _T("red light on"));

	SetDlgItemText(IDC_BUTTON_GREEN, _T("green light on"));

	BYTE time = (BYTE)(getFlushTime() / 100);

	CFingerVeinV3::getInstance()->setLedEx(CFingerVeinV3::COLOR_RED, 0);
	CFingerVeinV3::getInstance()->setLedEx(CFingerVeinV3::COLOR_GREEN | (time << 4), GetDlgItemInt(IDC_EDIT2));
}

void CFXsDlg::OnBnClickedButtonBuzzer()
{
	int time = getFlushTime();

	CFingerVeinV3::getInstance()->setBuzzer(time / 100, GetDlgItemInt(IDC_EDIT2));
}


int CFXsDlg::getFlushTime(){
	int time = GetDlgItemInt(IDC_EDIT1);
	if (time < 100){
		time = 100;
	}
	else if (time > 500){
		time = 500;
	}
	SetDlgItemInt(IDC_EDIT1, time);
	return time;
}


void CFXsDlg::OnEnChangeEdit1()
{
	
}


void CFXsDlg::OnEnChangeEdit2()
{

}




typedef struct _v_Feature{
	//unsigned int		dataFlag;  // 数据标识 0x1A2B3C4D

	BYTE				rawData[1152];
	unsigned int		MsData[392];
	BYTE				bdData[140];
	int					sti;
	int					stj;
	BYTE				wtPoint15[144];
	BYTE				wtPoint27[98];

	char				cVec[240];
	//char				cVec[720]; // 融合快速特征方案
	//int				mixCount;  // 融合模板个数
}FV_FEATURE, *PFV_FEATURE;



void CFXsDlg::OnBnClickedButtonLoad()
{
	std::string currentModelPath = (getAppPath() + "model").c_str();
	CString filepath;
	GetDlgItemText(IDC_EDIT_PATH2, filepath);

	CFileFind find;
	BOOL IsFind = find.FindFile(filepath + _T("/*.*"));

	FV_FEATURE feature;
	
	unsigned long tk = clock();
	int count = 0;

	while (IsFind)
	{
		IsFind = find.FindNextFile();
		if (find.IsDots())
		{
			continue;
		}
		else{
			CString filename = find.GetFileName();
			CString fullname = filepath + filename;

			if (fullname.Find(_T(".dat")) == -1)
				continue;

			FILE *fp = NULL;
			fopen_s(&fp, (LPCSTR)CStringA(fullname), "rb");
			if (fp)
			{
				int len = sizeof(FV_FEATURE);
				fseek(fp, 0L, SEEK_END);
				int flen = ftell(fp);
				if (flen == sizeof(FV_FEATURE))
				{
					fseek(fp, 0, SEEK_SET);
					fread(&feature, sizeof(FV_FEATURE), 1, fp);
					fclose(fp);
					int id = user.getUsableID();

					char modelPath[MAX_PATH];
					sprintf_s(modelPath, MAX_PATH, "%s/%u/", currentModelPath.c_str(), id);
					_mkdir(modelPath);

					strcat_s(modelPath, MAX_PATH, "1.dat");
					fopen_s(&fp, modelPath, "wb");
					if (fp)
					{
						fwrite(&feature, sizeof(FV_FEATURE), 1, fp);
						fclose(fp);
					}
					user.addUser(id);

					vector<BYTE> feature2(sizeof(FV_FEATURE));
					memcpy(feature2.data(), (BYTE*)&feature, sizeof(FV_FEATURE));

					char user[NameLen];
					sprintf_s(user, 20, "%u", id);
					addUser(id, user, feature2);
					count++;
				}
			}
		}
	}
	filepath.Format(_T("Total modeling time： %ums， Modeling total：%d, Average time spent： %ums"), clock() - tk, count, (clock() - tk) / count);
	outPrint(filepath);

	user.save();
	save();

	user.load();

	m_bHaveRecordsofBuildModule = TRUE;
}


void CFXsDlg::OnBnClickedButtonBuilds()
{
	std::string currentModelPath = (getAppPath() + "model").c_str();
	CString filepath;
	GetDlgItemText(IDC_EDIT_PATH, filepath);

	CFileFind find;
	BOOL IsFind = find.FindFile(filepath + _T("/*.*"));

	unsigned long tk = clock();
	int count = 0;

	vector<unsigned char>img1, img2, img3;
	vector<BYTE> feature;
	
	while (IsFind)
	{
		IsFind = find.FindNextFile();
		if (find.IsDots())
		{
			continue;
		}
		else{
			CString fullname = filepath + find.GetFileName();

			FILE *fp = NULL;

			if (fullname.Find(_T(".jpg")) != -1 && (_access((LPCSTR)CStringA(fullname), 0)) != -1)
			{
				cv::Mat src = cv::imread((LPCSTR)CStringA(fullname));
				
				BOOL ret = CFingerVeinV3::getInstance()->raw2Image(src, img1);

				ret = CFingerVeinV3::getInstance()->getModel(img1, img2, img3, feature, 1);

				int id = user.getUsableID();

				char user[NameLen];
				int pos = find.GetFileName().Find(_T("."));
				if (pos >= 0){
					sprintf_s(user, 20, "%s", find.GetFileName().Left(pos));
				}


				char modelPath[MAX_PATH];
				sprintf_s(modelPath, MAX_PATH, "%s/%u/", currentModelPath.c_str(), id);
				_mkdir(modelPath);

				strcat_s(modelPath, MAX_PATH, "1.dat");
				fopen_s(&fp, modelPath, "wb");
				if (fp)
				{
					fwrite(feature.data(), sizeof(FV_FEATURE), 1, fp);
					fclose(fp);
				}
				this->user.addUser(user, id);
				
				addUser(id, user, feature);
				count++;
			}
		}
	}
	filepath.Format(_T("Total modeling time： %ums， Modeling total：%d, Average time spent： %ums"), clock() - tk, count, (clock() - tk) / count);
	outPrint(filepath);

	user.save();
	save();

	user.load();

	m_bHaveRecordsofBuildModule = TRUE;
}





void CFXsDlg::OnBnClickedButtonCfg()
{
	CfgDlg dlg;
	if (IDOK == dlg.DoModal()){
		CFingerVeinV3::getInstance()->Dev_GetConfig(&conf_info);

	//	m_iDeviceCount = CFingerVeinV3::getInstance()->getDevCount();
		initDevlist(m_iDeviceCount);
	}
}


void CFXsDlg::OnEnChangeEditUsenameid()
{
	updateFeatureCount();
}

void CFXsDlg::updateFeatureCount(){
	CString usr;
	m_edUserName.GetWindowText(usr);

	if (!usr.IsEmpty()){
		int count = user.getFeatureCountByUser((LPCSTR)CStringA(usr));

		SetDlgItemInt(IDC_EDIT3, count);
	}
	else
		SetDlgItemText(IDC_EDIT3, _T(""));
}

void CFXsDlg::OnEnChangeEdit4()
{
	int count = GetDlgItemInt(IDC_EDIT4);
	if (count>3 || count<1)
		SetDlgItemInt(IDC_EDIT4, 1);
}


void CFXsDlg::OnBnClickedButtonSave()
{
	CString  stopBtn = _T("");
	stopBtn.LoadString(NULL, IDS_STOP_SAVE_IMG, 0);
	CString sText = _T("");
	m_saveImg.GetWindowText(sText);

	exitThread();
	if (stopBtn.CompareNoCase(sText) == 0)
	{
		//切换成开始建模
		CString  StartBtn = _T("");
		StartBtn.LoadString(NULL, IDS_STARTING_SAVE_IMG, 0);
		m_saveImg.SetWindowText(StartBtn);
		return;
	}
	memset(m_userName, 0, NameLen*sizeof(char));
	::GetDlgItemText(m_hWnd, m_edUserName.GetDlgCtrlID(), m_userName, sizeof(m_userName));
	if (_tcslen(m_userName) == 0){
		CString tmp;
		tmp.LoadString(NULL, IDS_NAME_OF_USER_CONNOT_BE_NULL);
		outPrint(tmp, TRUE);
		return;
	}

	
	
	m_bTerminate = FALSE;
	m_saveImg.SetWindowText(stopBtn);

	thread_handle = (HANDLE)_beginthread(save_process, 0, this);
}


void CFXsDlg::OnBnClickedButtonImgTest()
{
	CString str_stop;
	str_stop.LoadString(NULL, IDS_IN_IDTFY_IMG_TEST, 0);
	m_iIdentifyType = IdntTest;

	CString  strTmp;
	GetDlgItem(IDC_BUTTON_IMG_TEST)->GetWindowText(strTmp);

	exitThread();
	if (strTmp.CompareNoCase(str_stop) == 0)
		return;

	SetDlgItemText(IDC_BUTTON_IMG_TEST, str_stop);
	enableIdentifyButtons(FALSE);

	GetDlgItem(IDC_BUTTON_BuildModule)->EnableWindow(FALSE);
	GetDlgItem(IDC_EDIT_1V1ID)->EnableWindow(FALSE);
	GetDlgItem(IDC_BUTTON_QUITIDTFY)->EnableWindow(TRUE);
	GetDlgItem(IDC_STATIC_IDTFY_TIME)->SetWindowText(_T(""));
	GetDlgItem(IDC_BUTTON_SAVE)->EnableWindow(FALSE);
	GetDlgItem(IDC_BUTTON_IMG_TEST)->EnableWindow(FALSE);
	doFingerIdentify();
}


void CFXsDlg::OnBnClickedButtonImgTest2()
{
	CString str_stop;
	str_stop.LoadString(NULL, IDS_IN_IDTFY_IMG_TEST2, 0);
	m_iIdentifyType = IdntTest2;

	CString  strTmp;
	GetDlgItem(IDC_BUTTON_IMG_TEST2)->GetWindowText(strTmp);

	exitThread();
	if (strTmp.CompareNoCase(str_stop) == 0)
		return;

	SetDlgItemText(IDC_BUTTON_IMG_TEST2, str_stop);
	enableIdentifyButtons(FALSE);

	GetDlgItem(IDC_BUTTON_BuildModule)->EnableWindow(FALSE);
	GetDlgItem(IDC_EDIT_1V1ID)->EnableWindow(FALSE);
	GetDlgItem(IDC_BUTTON_QUITIDTFY)->EnableWindow(TRUE);
	GetDlgItem(IDC_STATIC_IDTFY_TIME)->SetWindowText(_T(""));
	GetDlgItem(IDC_BUTTON_SAVE)->EnableWindow(FALSE);
	GetDlgItem(IDC_BUTTON_IMG_TEST)->EnableWindow(FALSE);
	GetDlgItem(IDC_BUTTON_IMG_TEST2)->EnableWindow(FALSE);
	doFingerIdentify();
}