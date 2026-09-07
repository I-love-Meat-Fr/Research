#pragma once
#include <vector>
#include <Windows.h>
/*
1.fv_extract_model返回的模板的长度都是3352

2.fv_index的第一个参数是模板的一维数组，没做长度的校验。

3.后面建模和认证的函数也需要先调用fvdev_open

4.grabImage如果要循环取图，中间要休眠一会

*/
#define  FV_ERROR_NUM_1 0x01   //存储文件打开失败
#define  FV_ERROR_NUM_2 0x02   //函数未启动
#define  FV_ERROR_NUM_3 0x03   //指压状态函数调用失败
#define  FV_ERROR_NUM_4 0x04   //不支持该分辨率
#define  FV_ERROR_NUM_5 0x05   //设置曝光失败
#define  FV_ERROR_NUM_6 0x06   ////获取图像失败
#define  FV_ERROR_NUM_7 0x07   //打开的时候没有找到可用设备
#define  FV_ERROR_NUM_8 0x08   //设备打开失败
#define  FV_ERROR_NUM_9 0x09   //数据为空
#define  FV_ERROR_NUM_A 0x0A   //比对数据内容为空
#define  FV_ERROR_NUM_B 0x0B   //图像大小不匹配
#define  FV_ERROR_NUM_C 0x0C
#define  FV_ERROR_NUM_D 0x0D   //模板长度错误
#define  FV_ERROR_NUM_E 0x0E   //模板错误
#define  FV_ERROR_NUM_10 0x10  //参数错误
#define  FV_ERROR_NUM_11 0x11  //曝光的时候手指没有放好
#define  FV_ERROR_NUM_FE 0xFE  //函数指针为空
#define  FV_ERROR_NUM_FF 0xFF  //其他
#define  FV_ERROR_NUM_M1  1001     //1001 未初始化
#define  FV_ERROR_NUM_M2  1002     //1002 提取失败
#define  FV_ERROR_NUM_M3  3002     //3002 输入图像错误
#define  FV_ERROR_NUM_M4  2001     //2001：Bad Quality with too many pressure
#define  FV_ERROR_NUM_M5  2002     //2002：ROI Not Find
#define  FV_ERROR_NUM_M6  2003     //2003：Light Leak
typedef struct
{
	WORD SdkVer;		//SDK版本 BCD
	WORD SdkYear;		//SDK年份 BCD
	WORD SdkDate;		//SDK日期 BCD
	WORD FwVer;			//固件版本 BCD
	WORD FwYear;		//固件年份 BCD
	WORD FwDate;		//固件日期 BCD
	WORD UsbVendor;		//USB厂商ID
	WORD UsbProduct;	//USB设备ID
	WORD UsbVersion;	//USB版本ID
	BYTE DevType;		//设备类型 0x01:单侧 0x02:双侧 0x03:顶侧 0x10:指纹 0x12:复合
	BYTE DevRes;		//CMOS分辨率 30：VGA 130:SXGA
	WORD SensorId;		//CMOS识别ID 0x8243:MT9V011 0x8431:MT9M001 0x2282:MT9V117  0x9B
	BYTE SensorClock;	//CMOS MCLK时钟（MHz）
	WORD Width;			//CMOS输出宽度 640 | 1280
	WORD Height;		//CMOS输出高度 480 | 1024
	WORD MaxExp;		//CMOS最大曝光 4096 | 8192
	WORD MaxGain;		//CMOS最大增益 128 | 64
} T_INFO, *PT_INFO;

typedef  unsigned char   md_u8;
typedef  unsigned short  md_u16;
typedef  unsigned int    md_u32;
//版本号
#define SUBVERSION(v)          ((md_u8)(((md_u32)(v)) & 0xff))
#define MAINVERSION(N)           ((md_u8)((((md_u32)(N)) >> 8) & 0xff))


using   std::vector;
class CFingerVeinV3
{
	static CFingerVeinV3 *vv3;
	HMODULE   m_hLib;
	BOOL loadAlg(const char*  path);
public:
	enum{COLOR_RED,COLOR_GREEN};
	CFingerVeinV3(void);
	~CFingerVeinV3(void);


	//打开设备
	//devIndex  设备索引
	//hwnd为显示窗口
	BOOL   openDev(int devIndex, HWND hwnd);
	//是否已经打开
	BOOL   isOpen();
	//获取设备个数
	int    getDevCount();
	//关闭设备
	BOOL   closeDev();
	//获取一张指静脉图
	BOOL   grabImage(vector<BYTE>&img);
	////手指感应状态 0：没有手指 1：指尖按压 2：指腹按压 3：手指按压
	int    getFingerState();
	//打开红外灯
	void   openBackLight();
	//关闭红外灯
	void   closeBackLight();

	void   setNo(int no);

	int   getNo();
	//质量评估
	//q为质量，
	//0 QUALITY_GOOD
	//2001：Bad Quality with too many pressure
	//2002：ROI Not Find
	//2003：Light Leak
	BOOL   quality(unsigned char *img, int len,int &q);
	//设置灯
	BOOL   setSed(int color, BOOL bFlush);
	//设置调试显示信息
	void   setDebug(int level);
	
	//建模  模板的长度是3352
	BOOL   getModel(vector<BYTE>&img1,vector<BYTE>&img2,vector<BYTE>&img3,vector<BYTE>&model);
	//1比1
	BOOL   matchModel(vector<BYTE>&img,vector<BYTE>model,float &threshold);
	//1比n  第一个参数是n个模板的数组   
	//第二个参数  模板的个数
	//第三个参数   匹配上的数组序号
	//第四个参数   阈值  一般为0.52
	BOOL   index(unsigned char*model, int num,vector<BYTE>&img, int &pos,float &threshold);


	//设备初始化
	//设备第一次工作时，调用此函数，自动优化设备参数。
	//按正常姿势轻轻按下手指，等待初始完成并返回，才可以拿开手指。
	BOOL   setPara();
	static CFingerVeinV3 *getInstance();

	//获取函数使用后返回的错误
	//正常则返回0xFF;
	int GetFun_Error();
	//获取版本号
	int Get_Edition(int *buf);
	//自由设置红灯或者绿灯
	BOOL Set_led_bFlush(int Red_Num,int Rreen_Num);
	//获取设备固件号
	int Get_dev_info(PT_INFO buf);
};

