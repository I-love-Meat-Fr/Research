#pragma once
#include <vector>
#include <Windows.h>
#include <string>

#include "opencv2/opencv.hpp"

#pragma comment(lib, "opencv_core243.lib")
#pragma comment(lib, "opencv_highgui243.lib")
#pragma comment(lib, "opencv_imgproc243.lib")

#define MD120 0

//#define GA 1

#ifndef GA
#define MODEL "model"
#else 
#define MODEL "model_GA"
#endif


/*
1.The length of the template returned by fv extract model is 3352

2.The first argument to fv_index is a one-dimensional array of the template, with no length check.

3.Later modeling and authentication functions also need to call fvdev open first

4.grabImage If you want to loop the image, hibernate for a while

*/
#define  FV_ERROR_NUM_1 0x01   //Failed to open the storage file
#define FV_ERROR_NUM_2 0x02  // function does not start
#define FV_ERROR_NUM_3 0x03  // The shiatsu status function fails to be called
#define FV_ERROR_NUM_4 0x04  // This resolution is not supported
#define FV_ERROR_NUM_5 0x05  // Failed to set the exposure
#define FV_ERROR_NUM_6 0x06  //// Failed to get an image
#define FV_ERROR_NUM_7 0x07  // No device found when open
#define FV_ERROR_NUM_8 0x08  // Device failed to open
#define FV_ERROR_NUM_9 0x09  // Data is null
#define FV_ERROR_NUM_A 0x0A  // The comparison data content is null
#define FV_ERROR_NUM_B 0x0B  // The image size does not match
#define  FV_ERROR_NUM_C 0x0C
#define FV_ERROR_NUM_D 0x0D // The template length is incorrect
#define FV_ERROR_NUM_E 0x0E // Template error
#define FV_ERROR_NUM_10 0x10 // Parameter error
#define FV_ERROR_NUM_11 0x11 // Fingers are not placed properly when exposed
#define FV_ERROR_NUM_FE 0xFE // The function pointer is null
#define FV_ERROR_NUM_FF 0xFF // other
#define FV_ERROR_NUM_M1 1001 //1001 is not initialized
#define FV_ERROR_NUM_M2 1002 //1002 Failed to be extracted
#define FV_ERROR_NUM_M3 3002 //3002 The input image is incorrect
#define  FV_ERROR_NUM_M4  2001     //2001£ºBad Quality with too many pressure
#define  FV_ERROR_NUM_M5  2002     //2002£ºROI Not Find
#define  FV_ERROR_NUM_M6  2003     //2003£ºLight Leak

#pragma pack(push, 1)
typedef struct
{
	WORD SdkVer;		 //SDK version BCD
	WORD SdkYear;		 //SDK year BCD
	WORD SdkDate;		 //SDK date BCD
	WORD FwVer;			 // Firmware version BCD
	WORD FwYear;		 // Firmware year BCD
	WORD FwDate;		 // Firmware date BCD
	WORD UsbVendor;		 //USB vendor ID
	WORD UsbProduct;	 //USB device ID
	WORD UsbVersion;	 //USB version ID
	BYTE DevType;		 // Device Type 0x01: Unilateral 0x02: Bilateral 0x03: Top side 0x10: Fingerprint 0x12: Compound
	BYTE DevRes;		 //CMOS resolution 30: VGA 130:SXGA
	WORD SensorId;		 //CMOS Identification ID 0x8243:MT9V011 0x8431:MT9M001 0x2282:MT9V117 0x9B
	BYTE SensorClock;	 //CMOS MCLK Clock (MHz)
	WORD Width;			 //CMOS output width 640 | 1280
	WORD Height;		 //CMOS output height 480 | 1024
	WORD MaxExp;		 //CMOS Max exposure 4096 | 8192
	WORD MaxGain;		 //CMOS maximum gain 128 | 64
	
	
} T_INFO, *PT_INFO;

union  app_dev_version
{
	struct{
		unsigned short m;
		unsigned short s;
	};
	int ver;

};

struct VERSION_TIME_DETAIL
{
	union{
		struct {
			unsigned short  year : 12;
			unsigned short  month : 4;
			unsigned short  day : 5;
			unsigned short  hour : 5;
			unsigned short  minute : 6;
			//unsigned short  second:6;
		};
		int  d;
	};

};

typedef struct  _configure_info_
{
	unsigned int     sig;
	app_dev_version      version;
	int      identyTimeOut;
	int      bCheckDup;			//Whether to allow double registration
	int      bCheckSameFv;       //Whether to check the same finger
	int      baudrate;           //
	int      securityLevel;      //
	int      device_type;
	int      deviceId;          //
	int      usbMode;           // 1:rs232  2:usb
	int      LowThreshold;
	int      HighThreshold;
	int      avg;
	int      devImageWidth;
	int      devImageHeight;
	int      image_direction;
	int      sessor;
	int      read_error;
	int      defaultSessor;
	int      level;
	unsigned char      onLight[8];
	unsigned char      offLight[8];
	float    fLeakRatioThreshold;
	float    fPressThreshold;
	float    fEnergyThreshold;
	float    modelScore;
	float    updateScore;
	float    tiaozheng[5];

	VERSION_TIME_DETAIL     product_day;
	char    serailkey[16];
	char    DynamicExpo;
	char       numBootUp;
	char    reserver[14];
}  configure_info;

typedef struct
{
	unsigned char feature[3][6000];
	char user[32];
}FEATURE_GA;

#pragma pack(pop)


typedef  unsigned char   md_u8;
typedef  unsigned short  md_u16;
typedef  unsigned int    md_u32;
//°æ±¾ºÅ
#define SUBVERSION(v)          ((md_u8)(((md_u32)(v)) & 0xff))
#define MAINVERSION(N)           ((md_u8)((((md_u32)(N)) >> 8) & 0xff))


using   std::vector;
using   std::string;
string  getAppPath();
class CFingerVeinV3
{
	void *handle;
	static CFingerVeinV3 *vv3;
	HMODULE   m_hLib, m_hLib2;
	
	vector<FEATURE_GA> m_features;

public:
	BOOL loadAlg(const char*  path);
	enum{ COLOR_RED, COLOR_GREEN, COLOR_RED_FLUSH = 0x10, COLOR_GREEN_FLUSH = 0x11 };
	CFingerVeinV3(void);
	~CFingerVeinV3(void);

	BOOL isOpened;

	// Turn on the device
	//devIndex Device index
	//hwnd indicates the display window
	BOOL   openDev(int devIndex, HWND hwnd);
	// Whether it is opened
	BOOL   isOpen();
	// Obtain the number of devices
	int    getDevCount();
	// Shut down the device
	BOOL   closeDev();
	// Get a map of the veins in your finger
	BOOL   grabImage(vector<BYTE>&img);
	//// Finger sensing status 0: No finger 1: fingertip press 2: abdominal press 3: finger press
	int    getFingerState();
	// Turn on the infrared light
	void   openBackLight();
	// Turn off the infrared light
	void   closeBackLight();
	void   setNo(int no);

	int   getNo();

	BOOL getJpg(unsigned char *img, int len, vector<unsigned char>&dst);
	BOOL raw2Image(cv::Mat &pic, vector<unsigned char>&dst);
	// Quality assessment
	//q is the quality,
	//0 QUALITY_GOOD
	//2001: Bad Quality with too many pressure
	//2002: ROI Not found
	//2003: Light Leak
	BOOL   quality(unsigned char *img, int len, int &q);
	// Set the light
	BOOL   setSed(int color, int nOption);

	BOOL   setBuzzer(BYTE one_time, BYTE repeat_cnt);

	// Set debugging information
	void   setDebug(int level);

	// Modeling template length is 3352
	BOOL   getModel(vector<BYTE>&img1, vector<BYTE>&img2, vector<BYTE>&img3, vector<BYTE>&model, int index = 0);
	//1 to 1
	BOOL   matchModel(vector<BYTE>&img, vector<BYTE>model, float &threshold);
	//1 by n The first argument is an array of n templates
	// Number of second parameter templates
	// The third argument matches the array sequence number
	// The threshold of the fourth parameter is 0.63
	BOOL   index(unsigned char*model, int num,vector<BYTE>&img, int &pos,float &threshold);


	// Initialize the device
	// When the device works for the first time, this function is called to automatically optimize the device parameters.
	// Gently press the finger in the normal position and wait for the initial completion and return before removing the finger.
	BOOL   setPara();
	static CFingerVeinV3 *getInstance();
	static void deleteInstance();


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Turn on the device
	//devIndex Device index
	//hwnd indicates the display window
	BOOL   openDevEx(int devIndex, HWND hwnd);
	// Whether it is opened
	BOOL   isOpenEx();

	// Shut down the device
	BOOL   closeDevEx();
	// Get a map of the veins in your finger
	BOOL   grabImageEx(vector<BYTE>&img);
	//// Finger sensing status 0: No finger 1: fingertip press 2: abdominal press 3: finger press
	int    getFingerStateEx();
	// Turn on the infrared light
	void   openBackLightEx();
	// Turn off the infrared light
	void   closeBackLightEx();
	#if MD120
    void   openBuzzer(BYTE one_time,BYTE repeat_cnt);
	#endif
	void   setNoEx(int no);
	BOOL   setLedEx(int color, int nOption);

	int   getNoEx();
	int Get_Edition(int *buf);
	int Get_dev_info(PT_INFO buf);
	float getDefaultThreshold();
	int   loadModel(const char *path);
	int   user_index(unsigned char *img,int len,BOOL bUpdate, float *score);
	int   user_add(string user, vector<unsigned char>img1, vector<unsigned char>img2, vector<unsigned char>img3, unsigned int index);
	int   user_del(int index);
	int   loadNewUser(const char *path);
	int get_feature_count_by_id(unsigned int id);
	int   all_del();

	int getFeature(unsigned char *img, int len, unsigned char *feature);
	int   user_index(unsigned char *img, int len, float *score, std::string &user_name);
	bool   isExist(string user);

	int Dev_SetConfig(configure_info* pconf_info);
	int Dev_GetConfig(configure_info* pconf_info);
};

