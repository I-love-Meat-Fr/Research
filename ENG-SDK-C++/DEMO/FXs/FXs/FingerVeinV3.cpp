#include "stdafx.h"
#include "FingerVeinV3.h"

#include <string>
#include <time.h>
#include <stdlib.h>
using namespace std;

typedef void(WINAPI *fInit)(void);
typedef int(WINAPI *fGetFeature)(unsigned char *ptr_image, unsigned char *ptr_feature, int im_width, int im_height, int fvwindow[]);
typedef int(WINAPI *fMatFeature)(unsigned char *ptr_feature1, unsigned char *ptr_feature2, float *similarity);
typedef int(WINAPI *fRelease)(void);

static fInit XLFv_Init;
static fGetFeature XLFv_GetFeature;
static fMatFeature XLFv_MatFeature;
static fRelease XLFv_Release;


unsigned char image[2][76800];
unsigned char feature[2][6000];
int width = 320;
int height = 240;
int window[4] = { 34, 186, 18, 306 };

#define USE_FV_DEV_EX 1

#ifdef USE_FV_DEV_EX
	#define USE_FV_DEV_EX_NEW 1
#endif

struct  quality_info
{
	int         r;
	float		fLeakRatio;   // 漏光系数，大于 fLeakThreshold 表示图像漏光区域过大
	float		fPress;       // 指纹系数，小于 fPressThreshold 表示图像中含大量指纹信息
	float		fImageEnergy; // 图像能量，小于 fEnergyThreshold 表示有按压
};

#define declare_fun(name,...)  typedef  BOOL   (_stdcall *pfn##name) (__VA_ARGS__);   \
	static pfn##name name=0
#define load_fun(name)      name = (pfn##name)GetProcAddress(m_hLib,(#name));   \
	if(name == NULL) {  OutputDebugStringA("fun #name  not find.\n"); return FALSE;}
	 
typedef int(_stdcall *pZHY_Decrypt)(unsigned char* dst, unsigned char* src, int nlength);
typedef int(_stdcall *pZHY_Encrypt)(unsigned char* dst, unsigned char* src, int nlength);
static pZHY_Decrypt ZHY_Decrypt = 0;
static pZHY_Encrypt ZHY_Encrypt = 0;

declare_fun(fvdev_setpara);
declare_fun(fvdev_open,int,HWND);
declare_fun(isDevOpen);
declare_fun(fvdev_get_count,int*);
declare_fun(fvdev_close);
declare_fun(fvdev_openBackLight);
declare_fun(fvdev_closeBackLight);
declare_fun(fv_quality,unsigned char *img, int len,  int *avg,int *q);
declare_fun(fv_quality_ex,unsigned char *img, int len,  struct  quality_info *qi);
declare_fun(fvdev_grab,unsigned char *img, int *len);
declare_fun(fvdev_expose, int avg, int *exp1);
declare_fun(fvdev_get_state,int *);
declare_fun(fvdev_setno,int );
declare_fun(fvdev_getno,int *);
declare_fun(fvdev_set_led,int, int);
declare_fun(fv_extract_model,BYTE*,int,BYTE*,int,BYTE*,int,BYTE*,int*);
declare_fun(fv_match,BYTE*,int, BYTE*,float* );
declare_fun(fv_index,unsigned char *, int ,BYTE*,int, int *,float*);
declare_fun(setDebugLevel, int);
declare_fun(drawFingerVein, HDC,unsigned char*, int, RECT,RECT);

declare_fun(fvdev_open_ex,int,HWND,void*);
declare_fun(isDevOpen_ex,void*);
declare_fun(fvdev_openBackLight_ex,void*);
declare_fun(fvdev_closeBackLight_ex,void*);
declare_fun(fvdev_set_buzzer_ex, void *, BYTE, BYTE);		//yejs add
#if MD120
declare_fun(fvdev_set_buzzer_ex,void *,BYTE,BYTE);
declare_fun(fvdev_set_buzzer,BYTE,BYTE);
#endif
declare_fun(fvdev_grab_ex,void*,unsigned char *img, int *len);
declare_fun(fvdev_close_ex,void*);
declare_fun(fvdev_expose_ex,void*,int avg, int *exp1);
declare_fun(fvdev_get_state_ex,void*,int *state);
declare_fun(fvdev_set_led_ex, void *handle, int Color, int nOption);//yejs modify
declare_fun(fvdev_setno_ex,void *handle,int no);
declare_fun(fvdev_getno_ex,void *handle,int* no);
declare_fun(fv_Get_Edition,int *);
declare_fun(fv_Get_Dev_Info,PT_INFO);
declare_fun(fv_Dev_SetConfig, configure_info*);
declare_fun(fv_Dev_GetConfig, configure_info*);

declare_fun(fv_get_default_threshold,float *score);
declare_fun(fv_load_model,const char *, int *);


declare_fun(fv_user_index,unsigned char *img, int len,int *index,float threshold,float *score,BOOL  bUpdate);
declare_fun(fv_user_add,BYTE *picData1,int picLen1,BYTE*picData2,int picLen2,BYTE*picData3,int picLen3, unsigned int *index);
declare_fun(fv_user_del,unsigned int index);
declare_fun(fv_index_addUser,const char *, int *);
declare_fun(fv_Decrypt, unsigned char *img, int len, unsigned char *jpgBuf);
declare_fun(fv_Encrypt, unsigned char *src, int len, vector<unsigned char>&dst);
declare_fun(fvdev_get_feature_count_by_id, int &count, unsigned int id); 
declare_fun(fv_all_del);
declare_fun(fv_Load_FingerVein_v3_dev);


CFingerVeinV3 *CFingerVeinV3::vv3=0;
string  getAppPath(){
	char exeFullPath[MAX_PATH]; // MAX_PATH
	GetModuleFileNameA(NULL,exeFullPath,MAX_PATH);
	char drive[_MAX_DRIVE] = {0};
	char dir[_MAX_DIR] = {0};
	char filename[_MAX_DIR] = { 0 };
	char fileext[_MAX_DIR] = { 0 };
	_splitpath_s(exeFullPath, drive, 3, dir, 256, filename, 256,fileext, 256);
	string  strAppPath=drive;
	strAppPath += dir;
	return strAppPath;
}

//CFingerVeinV3 *CFingerVeinV3::vv3=0;

CFingerVeinV3* CFingerVeinV3::getInstance()
{
	if (!vv3)
	{
		vv3=new CFingerVeinV3();
		string strPath;

#ifndef USE_FV_DEV_EX
		strPath += "FingerVein_v3_dev.dll";
#else
#ifdef USE_FV_DEV_EX_NEW
		strPath += "thin_FingerVein.dll";
#else
		strPath += "thin_FingerVein_old.dll";
#endif

#endif

		if(vv3->loadAlg(strPath.c_str()) == TRUE) {
			return vv3;
		} else {
			MessageBox(NULL,TEXT("FingerVein_v3_dev.dll加载失败!"),NULL,MB_OK);
			delete vv3;
			return NULL;
		}
	}
	return vv3;
}

void CFingerVeinV3::deleteInstance()
{
	if (NULL != vv3)
	{
		delete vv3;
		vv3 = NULL;
	}
}
CFingerVeinV3::CFingerVeinV3(void) :isOpened(FALSE)
{
	handle=0;
}


CFingerVeinV3::~CFingerVeinV3(void)
{
}
BOOL CFingerVeinV3::loadAlg(const char* path)
{
	m_hLib=LoadLibraryA(path);
	if (m_hLib)
	{
		load_fun(fvdev_setpara);
		load_fun(fvdev_open);
		load_fun(fvdev_close);
		load_fun(fvdev_get_count);
		load_fun(isDevOpen);
		load_fun(fvdev_grab);
		load_fun(fvdev_get_state);
		load_fun(fvdev_set_led);
		load_fun(fv_extract_model);
		load_fun(fv_match);
		load_fun(fv_index);
		load_fun(setDebugLevel);
		load_fun(drawFingerVein);
		load_fun(fv_quality);
		//load_fun(fv_quality_ex);
		load_fun(fvdev_openBackLight);
		load_fun(fvdev_closeBackLight);
		load_fun(fvdev_expose);
		load_fun(fvdev_getno);
		load_fun(fvdev_setno);
		load_fun(fvdev_open_ex);
		load_fun(isDevOpen_ex);
		load_fun(fvdev_openBackLight_ex);
		load_fun(fvdev_closeBackLight_ex);
		load_fun(fvdev_grab_ex);
		load_fun(fvdev_close_ex);
		load_fun(fvdev_expose_ex);
		load_fun(fvdev_get_state_ex);
		load_fun(fvdev_setno_ex);
		load_fun(fvdev_getno_ex);
		load_fun(fv_Get_Edition);
		load_fun(fv_Get_Dev_Info);
		load_fun(fv_get_default_threshold);
		load_fun(fv_load_model);
		load_fun(fv_user_index);
		load_fun(fv_user_add);
		load_fun(fv_user_del);


#ifdef USE_FV_DEV_EX
		//load_fun(fv_index_addUser);
		load_fun(fvdev_set_buzzer_ex); 
		load_fun(fvdev_set_led_ex);
	
		load_fun(fv_Dev_SetConfig);
		load_fun(fv_Dev_GetConfig);
		load_fun(fvdev_get_feature_count_by_id); 
		
#ifdef USE_FV_DEV_EX_NEW
		load_fun(fv_Decrypt);
		load_fun(fv_Encrypt);
		load_fun(fv_all_del);

		load_fun(fv_Load_FingerVein_v3_dev);
		fv_Load_FingerVein_v3_dev();
#endif
#endif

		//！*****************************优化编译后报错**************************
		m_hLib2 = LoadLibraryA("zhy_fv401.dll");
		if (m_hLib2){
			ZHY_Encrypt = (pZHY_Decrypt)GetProcAddress(m_hLib2, "ZHY_Encrypt");
			ZHY_Decrypt = (pZHY_Decrypt)GetProcAddress(m_hLib2, "ZHY_Decrypt");
		}


		HINSTANCE hDLL = LoadLibraryA("XLFvDLL.dll");
		XLFv_Init = (fInit)GetProcAddress(hDLL, "XLFv_Init");
		XLFv_GetFeature = (fGetFeature)GetProcAddress(hDLL, "XLFv_GetFeature");
		XLFv_MatFeature = (fMatFeature)GetProcAddress(hDLL, "XLFv_MatFeature");
		XLFv_Release = (fRelease)GetProcAddress(hDLL, "XLFv_Release");

		if (fvdev_open)
		{
			return TRUE;
		}

		if (XLFv_Init)
			XLFv_Init();
	}
	return FALSE;
}
BOOL CFingerVeinV3::openDev(int devIndex,HWND hwnd)
{
	if (fvdev_open)
	{
		if(fvdev_open(devIndex,hwnd)){
			//openBackLight();
			isOpened = TRUE;
			return TRUE;
		}

	}
	return FALSE;
}
BOOL CFingerVeinV3::openDevEx(int devIndex,HWND hwnd)
{
	if (fvdev_open_ex)
	{
		if(fvdev_open_ex(devIndex,hwnd,&handle) && handle){
			//openBackLightEx();
			isOpened = TRUE;
			return TRUE;
		}

	}
	return FALSE;
}
BOOL CFingerVeinV3::isOpen()
{
	return isOpened;

	if (isDevOpen)
	{
		return isDevOpen();
	}
	return FALSE;
}
BOOL CFingerVeinV3::isOpenEx()
{
	return isOpened;

	if (isDevOpen_ex)
	{
		return isDevOpen_ex(handle);
	}
	return FALSE;
}
BOOL CFingerVeinV3::closeDev()
{
	if (fvdev_close)
	{
		fvdev_close();
		closeBackLight();
		isOpened = FALSE;
		return TRUE;
	}
	return FALSE;
}
BOOL CFingerVeinV3::closeDevEx()
{
	if (fvdev_close_ex)
	{
		fvdev_close_ex(handle);
		handle=0;
		isOpened = FALSE;
		return TRUE;
	}
	return FALSE;
}
void CFingerVeinV3::openBackLight()
{
	if (fvdev_openBackLight)
	{
		fvdev_openBackLight();
	}
}
#if MD120
void CFingerVeinV3::openBuzzer(BYTE one_time,BYTE repeat_cnt)
{
    if (fvdev_set_buzzer)
    {
        fvdev_set_buzzer(one_time,repeat_cnt);
    }
}
#endif
void CFingerVeinV3::openBackLightEx()
{
	if (fvdev_openBackLight_ex)
	{
		fvdev_openBackLight_ex(handle);
	}
}
void CFingerVeinV3::closeBackLight()
{
	if (fvdev_closeBackLight)
	{
		fvdev_closeBackLight();
	}
}
void CFingerVeinV3::closeBackLightEx()
{
	if (fvdev_closeBackLight_ex)
	{
		fvdev_closeBackLight_ex(handle);
	}
}
BOOL CFingerVeinV3::grabImage(vector<BYTE>&img)
{
	if (fvdev_grab){

		img.resize(780000);
		int len=img.size();
		BOOL bRet=0;

		if(fvdev_grab(img.data(),&len))
		{	
			img.resize(len);
			return TRUE;
		}
	}
	return FALSE;
}
BOOL CFingerVeinV3::grabImageEx(vector<BYTE>&img)
{
	if (fvdev_grab){

		img.resize(780000);
		int len=img.size();
		BOOL bRet=0;
		if(fvdev_grab_ex(handle,img.data(),&len))
		{
			img.resize(len);
			return TRUE;
		}
	}
	return FALSE;
}



BOOL CFingerVeinV3::getJpg(unsigned char *img, int len, vector<unsigned char>&dst)
{
	TCHAR szErr[256] = _T("\0");
	BOOL ret = FALSE;
//	return FALSE;

	
//	if (fv_Decrypt)
	{
		vector<unsigned char>buf((len + 15) / 16 * 16);

		try{
			if (fv_Decrypt)
				ret = fv_Decrypt(img, len, &buf[0]);// !ZHY_Decrypt(&buf[0], img, len);
			else
				ret = !ZHY_Decrypt(&buf[0], img, len);
		}
		catch (...)
		{
			DWORD err = GetLastError();
			LPVOID lpMsgBuf;
			FormatMessage(
				FORMAT_MESSAGE_ALLOCATE_BUFFER
				| FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
				NULL,
				err,
				MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
				(LPTSTR)&lpMsgBuf,
				0,
				NULL
				);
		//	AfxMessageBox((LPCTSTR)lpMsgBuf);
			LocalFree(lpMsgBuf);
			return FALSE;
		}
		
		cv::Mat srcMat(buf);

		cv::Mat destMat1 = cv::imdecode(srcMat, CV_LOAD_IMAGE_GRAYSCALE);

		if (destMat1.cols != 320 || destMat1.rows != 240)
		{
			return FALSE;
		}

#define SET_DPI 1
		
		cv::Mat dstImage2;
		cv::resize(destMat1, dstImage2, cv::Size(destMat1.cols * SET_DPI, destMat1.rows * SET_DPI), 0, 0, cv::INTER_NEAREST);

		int size = (int)(dstImage2.cols * dstImage2.rows);
		dst.resize(size);

		ret = cv::imencode(".jpg", dstImage2, dst);
/*
		int iResolution = 320 * SET_DPI;
		char * pResolution = (char*)&iResolution;// - iResolution为要设置的分辨率的数值，如72dpi 
		// - 设置JPG图片的分辨率 
		dst[0x0D] = 1;// - 设置使用图片密度单位 
		// - 水平密度，水平分辨率 
		dst[0x0E] = pResolution[1];
		dst[0x0F] = pResolution[0];
		// - 垂直密度，垂直分辨率 
		iResolution = 240 * SET_DPI;
		dst[0x10] = pResolution[1];
		dst[0x11] = pResolution[0];
		*/
		return ret;
	}
	return FALSE;
}

#define IMAGE_ENC        1
#define IMAGE_BMP        0

BOOL CFingerVeinV3::raw2Image(cv::Mat &pic, vector<unsigned char>&dst){
	/*int jpgLen=76800;
	unsigned char *jpgBuf=(unsigned char*)malloc(jpgLen);*/
	unsigned long tk = clock();
	//int ret=encode_jpeg_m(data, jpgBuf,&jpgLen);

//	cv::Mat pic(240, 320, CV_8UC1, data);
	//memcpy(pic.data,data,len);
	std::vector<int> param;

#if  IMAGE_BMP
	string ext = ".bmp";
#else
	string ext = ".jpg";
	param.push_back(CV_IMWRITE_JPEG_QUALITY);
	param.push_back(85);//default(95) 0-100
//	param.push_back(0);
	
#endif
	vector<unsigned char>jpgBuf;
	jpgBuf.resize(100000);
	cv::imencode(ext, pic, jpgBuf, param);
#if IMAGE_ENC
	int jpgLen = (jpgBuf.size() + 15) / 16 * 16;
	jpgBuf.resize(jpgLen);
#else
	int jpgLen = jpgBuf.size();
#endif
	//saveFile(buff);
#if   !IMAGE_BMP
	jpgLen += 16;
#endif

	dst.resize(jpgLen);
#if IMAGE_ENC
	if (fv_Encrypt)
		fv_Encrypt(jpgBuf.data(), jpgBuf.size(), dst);
	else
		ZHY_Encrypt(&dst[0], jpgBuf.data(), jpgBuf.size());
#else
	memcpy(&dst[0], jpgBuf.data(), jpgBuf.size() - 16);
#if IMAGE_BAG
	calcBag(dst.data(), jpgLen - 16);
#endif

#endif


	return TRUE;

}

BOOL CFingerVeinV3::quality(unsigned char *img, int len,int &q)
{

	if (fv_quality)
	{
		int avg;
		return fv_quality(img,len,&avg,&q);
	}
	return FALSE;
}
int  CFingerVeinV3::getDevCount()
{
	if (fvdev_get_count)
	{
		int count;
		fvdev_get_count(&count);
		return count;
	}
	return 0;
}
void CFingerVeinV3::setNo(int no)
{
	if (fvdev_setno)
	{
		fvdev_setno(no);
	}
}
void CFingerVeinV3::setNoEx(int no)
{
	if (fvdev_setno_ex)
	{
		fvdev_setno_ex(handle,no);
	}
}
int CFingerVeinV3::getNo()
{
	if (fvdev_getno)
	{
		int no=-1;
		fvdev_getno(&no);
		return no;
	}
	return -1;
}
int CFingerVeinV3::getNoEx()
{
	if (fvdev_getno_ex)
	{
		int no=-1;
		fvdev_getno_ex(handle,&no);
		return no;
	}
	return -1;
}
int  CFingerVeinV3::getFingerState()
{
	if (fvdev_get_state)
	{
		int state=0;
		if(fvdev_get_state(&state))
		{
			return state; 
		}
	}
	return 0;
}
int  CFingerVeinV3::getFingerStateEx()
{
#ifdef	USE_FV_DEV_EX_NEW
	T_INFO buf;
	memset(&buf, 0, sizeof(T_INFO));
	Get_dev_info(&buf);
	//用新的源码生成的库时，用脉迪的模块有问题：算法库获取手指状态可以，应用同时去获取会报错121（信号灯超时时间已到），但如果应用单独获取也能获取到，这里用了一个折衷的方法，
	//调用 fvdev_get_state 获取算法库里获取到的状态变量值
	if (buf.DevType == 0x02)
		return getFingerState();
#endif

	if (fvdev_get_state_ex)
	{
		int state=0;
		if(fvdev_get_state_ex(handle,&state))
			return state;
	}
	return 0;
}
BOOL CFingerVeinV3::setSed(int color, int nOption)
{
	if (fvdev_set_led)
	{
		return fvdev_set_led(color, nOption);
	}
	return FALSE;
}
BOOL CFingerVeinV3::setLedEx(int color, int nOption)
{
	T_INFO buf;
	memset(&buf, 0, sizeof(T_INFO));
	Get_dev_info(&buf);

	if (buf.DevType == 0x02)
		return FALSE;

	if (fvdev_set_led_ex)
	{
		return fvdev_set_led_ex(handle, color, nOption);
	}
	return FALSE;
}

BOOL   CFingerVeinV3::setBuzzer(BYTE one_time, BYTE repeat_cnt){
	T_INFO buf;
	memset(&buf, 0, sizeof(T_INFO));
	Get_dev_info(&buf);

	if (buf.DevType == 0x02)
		return FALSE;

	if (fvdev_set_buzzer_ex)
	{
		return fvdev_set_buzzer_ex(handle, one_time, repeat_cnt);
	}
	return FALSE;
}

void CFingerVeinV3::setDebug(int level)
{
	if (setDebugLevel)
	{
		setDebugLevel(level);
	}

}

BOOL CFingerVeinV3::getModel(vector<BYTE>&img1, vector<BYTE>&img2, vector<BYTE>&img3, vector<BYTE>&model, int index)
{
#ifndef GA
	if (fv_extract_model)
	{
		model.resize(3352);
		int len=model.size();
		return fv_extract_model(img1.data(),img1.size(),img2.data(),img2.size(),img3.data(),img3.size(),model.data(),&len);
	}
#else
	model.resize(6000);
	assert(model.size() == 6000);

	vector<BYTE> *pImg = NULL;
	if (index == 1) {
		pImg = &img1;
	}
	else if (index == 2) {
		pImg = &img2;
	}
	else if (index == 3) {
		pImg = &img3;
	}


	if (getFeature(pImg->data(), pImg->size(), model.data()))
		return TRUE;

#endif
	return FALSE;
}
BOOL  CFingerVeinV3::matchModel(vector<BYTE>&img,vector<BYTE>model,float &threshold)
{
#ifndef GA
	if (fv_match)
	{
		return fv_match(img.data(),img.size(),model.data(),&threshold);
	}
#else
	unsigned char feature[6000];

	assert(model.size() == 6000);

	float score;
	getFeature(img.data(), img.size(), feature);

	int ret = XLFv_MatFeature(feature, model.data(), &score);
	if (score >= threshold) {
		threshold = score;
		return TRUE;
	}
#endif
	return FALSE;
}
BOOL CFingerVeinV3::index(unsigned char*model, int num,vector<BYTE>&img, int &pos,float &threshold)
{
	if (fv_index)
	{
		return fv_index(model,num,img.data(),img.size(),&pos,&threshold);
	}
	return FALSE;
}
BOOL CFingerVeinV3::setPara()
{
	if (fvdev_setpara)
	{
		fvdev_setpara();
		return TRUE;
	}
	return FALSE;
}

int CFingerVeinV3::Get_Edition(int *buf)
{
	if (fv_Get_Edition)
	{
		fv_Get_Edition(buf);
		return TRUE;
	}
	return FALSE;
}


int CFingerVeinV3::Get_dev_info(PT_INFO buf)
{
	if (fv_Get_Dev_Info)
	{
		fv_Get_Dev_Info(buf);
		return TRUE;
	}
	return FALSE;
}


int CFingerVeinV3::Dev_SetConfig(configure_info* pconf_info)
{
	if (fv_Dev_SetConfig)
	{
		fv_Dev_SetConfig(pconf_info);
		return TRUE;
	}
	return FALSE;
}

int CFingerVeinV3::Dev_GetConfig(configure_info* pconf_info)
{
	if (fv_Dev_GetConfig)
	{
		fv_Dev_GetConfig(pconf_info);
		return TRUE;
	}
	return FALSE;
}

float CFingerVeinV3::getDefaultThreshold()
{
	if (fv_get_default_threshold)
	{
		float score=0;
		fv_get_default_threshold(&score);
		return score;
	}
	return (float)0.6;
}
int CFingerVeinV3::user_index(unsigned char *img,int len,BOOL bUpdate, float *score)
{
	if (fv_user_index)
	{
		int index=0;
		float threshold = getDefaultThreshold();
		*score = 0;
		if (fv_user_index(img, len, &index, threshold, score, bUpdate)){
			return index;
		}
	}
	return 0;
}
#define IMAGE_ENC        1

BOOL Image2Raw(vector<unsigned char>&src, vector<unsigned char>&dst)
{
	if (src.size() == 0)
	{
//		fv_Set_Error(FV_ERROR_NUM_A);//数据内容为空
		return FALSE;
	}
#if IMAGE_BAG
	if (!checkBag(src.data(), src.size()))
	{
		return 0;
	}
#endif
	dst.resize(76800);
	vector<unsigned char>buf((src.size() + 15) / 16 * 16);
	int G_error = 0;
#if  IMAGE_ENC
	G_error = fv_Decrypt(&src[0], src.size(), &buf[0]);
#else
	memcpy(&buf[0], &src[0], src.size());
#endif
//	fv_Set_Error(G_error);//图像大小不匹配
	unsigned long tk = clock();
	//int ret = decode_jpeg_m(buf.data(),buf.size(),dst.data());
	cv::Mat destMat1 = cv::imdecode(cv::Mat(buf), CV_LOAD_IMAGE_GRAYSCALE);
//	dbg_print(0, "decode_jpeg_m  time = %d\n", clock() - tk);
	if (destMat1.cols != 320 || destMat1.rows != 240)
	{
	//	fv_Set_Error(FV_ERROR_NUM_B);//图像大小不匹配
		return FALSE;
	}
	memcpy(dst.data(), destMat1.data, 76800);

	//return ret;
	return TRUE;

}

BOOL Image2Raw(unsigned char *img, int len, vector<unsigned char>&dst)
{

	vector<unsigned char>imgData(len);
	//imgData.assign(img,img+len-1);
	memcpy(imgData.data(), img, len);
	return Image2Raw(imgData, dst);
}

int CFingerVeinV3::getFeature(unsigned char *img, int len, unsigned char *feature){
	std::vector<uchar> v;
	Image2Raw(img, len, v);

	memcpy(image[0], v.data(), 76800);

	memset(feature, 0, 6000);
	int ret = XLFv_GetFeature(image[0], feature, width, height, window);

	return ret;
}

int  CFingerVeinV3::user_index(unsigned char *img, int len, float *score, std::string &user_name){

//	std::string path = getAppPath() + MODEL + "\\*.*";

	unsigned char feature[2][6000];

	getFeature(img, len, feature[0]);

	float threshold = getDefaultThreshold();

	int index = 1;
	for (auto it : m_features){
		for (int j = 0; j < 3; j++){
			int ret = XLFv_MatFeature(feature[0], it.feature[j], score);
			if (*score > threshold) {
				user_name = it.user;
				return index;
			}
		}
		index++;
	}

	/*
	CFileFind finder;
	CString strPath;
	BOOL bWorking = finder.FindFile(CString(path.c_str()));

	while (bWorking)
	{
		bWorking = finder.FindNextFile();
		strPath = finder.GetFilePath();

		if (!finder.IsDirectory() && !finder.IsDots() && strPath.Find(_T(".jpg") >= 0))
		{
			
			FILE *fp = NULL;
			fopen_s(&fp, (LPCSTR)CStringA(strPath), "rb");

			fseek(fp, 0L, SEEK_END);
			int flen = ftell(fp);

			fseek(fp, 0, SEEK_SET);
			fread(image[1], flen, 1, fp);
			fclose(fp);


			getFeature(image[1], flen, feature[1]);

			int ret = XLFv_MatFeature(feature[0], feature[1], score);
			if (*score > threshold) {
				user_name = (LPCSTR)CStringA(finder.GetFileName());
				return 1;
			}
		}
	}*/
	return -1;
}

bool  CFingerVeinV3::isExist(string user){
	for (auto it : m_features){
		if (user == it.user){
			return true;
		}
	}
	return false;

}

int CFingerVeinV3::loadModel(const char *path)
{
#ifndef GA
	if (fv_load_model)
	{
		int num=0;
		fv_load_model(path,&num);
		return num;
	}
#else

	errno_t err;
	FILE *fp = NULL;
	std::string str = path;
	err = fopen_s(&fp, (str + "\\feature_ga.dat").c_str(), "rb");
	if (!err)
	{
		fseek(fp, 0L, SEEK_END);
		int flen = ftell(fp);
		fseek(fp, 0L, SEEK_SET);
		uchar *data = new uchar[flen];
		fread(data, flen, 1, fp);

		m_features.clear();

		FEATURE_GA *features = (FEATURE_GA *)data;
		for (int i = 0; i < flen / sizeof(FEATURE_GA); i++){
			m_features.push_back(features[i]);
		}
		
		fclose(fp);

		return m_features.size();
	}

#endif
	return -1;
}
int CFingerVeinV3::loadNewUser(const char *path)
{
	if (fv_index_addUser)
	{
		int num=0;
		fv_index_addUser(path,&num);
		return num;
	}
	return -1;
}




int CFingerVeinV3::user_add(string user, vector<unsigned char>img1, vector<unsigned char>img2, vector<unsigned char>img3, unsigned int index)
{
#ifndef GA
	if (fv_user_add)
	{
	//	unsigned int index=0;
		if(fv_user_add(img1.data(),img1.size(),img2.data(),img2.size(),img3.data(),img3.size(),&index))
			return index;
	}
#else
	vector<BYTE> feature1, feature2, feature3;
	BOOL ret1 = getModel(img1, img2, img3, feature1, 1);
	BOOL ret2 = getModel(img1, img2, img3, feature2, 2);
	BOOL ret3 = getModel(img1, img2, img3, feature3, 3);

	if (ret1 && ret2 && ret3){
		FEATURE_GA features;
		memcpy(features.feature[0], feature1.data(), 6000);
		memcpy(features.feature[1], feature2.data(), 6000);
		memcpy(features.feature[2], feature3.data(), 6000);

		strcpy_s(features.user, 32, user.c_str());

		errno_t err;
		FILE *fp = NULL;
		err = fopen_s(&fp, (getAppPath() + MODEL + "\\feature_ga.dat").c_str(), "ab+");
		if (!err)
		{
			int size = sizeof(features);
			fwrite(&features, sizeof(features), 1, fp);
			fseek(fp, 0L, SEEK_END);
			int flen = ftell(fp);
			fclose(fp);

			int num = CFingerVeinV3::getInstance()->loadModel((getAppPath() + MODEL).c_str());

			return num;
		}
	}

#endif
	return 0;
}
int CFingerVeinV3::user_del(int index)
{
	if (fv_user_del)
	{
		return fv_user_del(index);
	}
	return 0;
}

int CFingerVeinV3::all_del()
{
	if (fv_all_del)
	{
		return fv_all_del();
	}
	return 0;
}

int CFingerVeinV3::get_feature_count_by_id(unsigned int id)
{
	if (fvdev_get_feature_count_by_id)
	{
		int count = 0;
		fvdev_get_feature_count_by_id(count, id);
		return count;
	}
	return 0;
}
