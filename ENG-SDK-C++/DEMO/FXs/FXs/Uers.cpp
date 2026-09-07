#include "StdAfx.h"
#include "Uers.h"
#include "FingerVeinV3.h"

#include <atlconv.h>
string  getAppPath();
using namespace Json;



CUers::CUers(void)
{
	
}


CUers::~CUers(void)
{
}
void CUers::load()
{
	int num = CFingerVeinV3::getInstance()->loadModel((getAppPath() + MODEL).c_str());

#ifndef GA
	//int maxnum;
	//const char* filepath = (getAppPath() + "user.txt").c_str();
	errno_t err;
	FILE *fp = NULL;
	err = fopen_s(&fp, (getAppPath() + "user.txt").c_str() , "r");
	if (!err)
	{

		mapUsers.clear();
		char *content = new char[1000000];
	//	char content[400000];

		fread(content, sizeof(char)* 1000000, 1, fp);

		Json::Reader reader;
	//	Json::Value json_object;

		if (reader.parse(content, json_object))
		{
			if (json_object.isMember("count"))
			{
				int num=json_object["count"].asInt();
				Value users=json_object["users"];
				for (int i=0;i<users.size();i++)
				{
					int index=users[i]["id"].asInt();
					/*	if (index>maxnum)
					{
					maxnum = index;
					}*/
					string name=users[i]["name"].asCString();
					mapUsers[index]=name;
				}
			}
			
		}
		fclose(fp);
		delete[] content;
		//for (int i=mapUsers.size();i<num;i++)
		//{
		//	char name[20];
		//	itoa(i+1,name,10);
		//	//会导致错误   例如:1,2,4,6,7;会错乱

		//	mapUsers[i+1]=name;
		//}
	}
#endif
}
void CUers::save()
{
#ifndef GA
	int count=mapUsers.size();
	map<int,string>::iterator it=mapUsers.begin();
	Value root;
	root["count"]=count;
	Value val;
	for (;it!=mapUsers.end();it++)
	{
		Value  tmp;
		tmp["id"]=it->first;
		tmp["name"]=it->second;
		val.append(tmp);
	}
	root["users"]=val;
	
	string str=FastWriter().write(root);
	errno_t err;
	FILE *fp = NULL;
	err = fopen_s(&fp,(getAppPath() + "user.txt").c_str(), "w");
	if (!err)
	{
		fwrite(str.c_str(),str.size(),1,fp);
		fclose(fp);
	}
#endif
}

bool CUers::isExist(unsigned char *img, int len)
{
	float score;
	return CFingerVeinV3::getInstance()->user_index(img,len,FALSE, &score)>0;
}

bool CUers::isExist(string user){
#ifdef GA
	return CFingerVeinV3::getInstance()->isExist(user);
#endif
	return 0;
}

int CUers::addUser(string user, vector<unsigned char>img1, vector<unsigned char>img2, vector<unsigned char>img3, BOOL bSave, BOOL bCheckDup)
{
#ifndef GA
	if (bCheckDup){
		map<int, string>::iterator it = mapUsers.begin();
		for (; it != mapUsers.end(); it++)
		{
			if (it->second == user){
				return 0;
			}
		}
	}
	int id = getUsableID();
	id = CFingerVeinV3::getInstance()->user_add(user, img1, img2, img3, id);
	if (id>0)
	{
		mapUsers[id] = user;
		if (bSave)
			save();
		return id;
	}
#else
	int index = CFingerVeinV3::getInstance()->user_add(user, img1, img2, img3);
	if (index>0)
	{
		return index;
	}
#endif
	return 0;
}

int CUers::addUser(int index)
{
#ifndef GA

	if (index>0)
	{
		char user[20];
		sprintf_s(user, 20, "%u", index);

		mapUsers[index] = user;
	//	save();
		return index;
	}
#else

#endif
	return 0;
}

int CUers::addUser(string user, int index)
{
#ifndef GA

	if (index>0)
	{
		mapUsers[index] = user;
		//	save();
		return index;
	}
#else

#endif
	return 0;
}

string CUers::find_user(unsigned char *img, int len,int &index, float *score)
{
#ifndef GA
	index = CFingerVeinV3::getInstance()->user_index(img, len, TRUE, score);
	map<int, string>::iterator it = mapUsers.find(index);
	if (it != mapUsers.end())
	{
		return it->second;
	}
#else
/*	*/

	std::string user_name = "\0";

	if ((index = CFingerVeinV3::getInstance()->user_index(img, len, score, user_name)) > -1){
		return user_name;
	}

#endif

	return "unknown";
}
int CUers::delUser(int id, BOOL bSave)
{
#ifndef GA
	map<int,string>::iterator it=mapUsers.begin();
	for (;it!=mapUsers.end();it++)
	{
		if (it->first == id){
			mapUsers.erase(id);
			if (bSave)
				save();
			return CFingerVeinV3::getInstance()->user_del(id);
		}
	}
#else

#endif

	return -1;
}

int CUers::delAllUser(BOOL bSave)
{
#ifndef GA
/*	map<int, string>::iterator it = mapUsers.begin();
	for (; it != mapUsers.end(); it++)
	{
	//	mapUsers.erase(it->first);
			
		CFingerVeinV3::getInstance()->user_del(it->first);
	}*/
	CFingerVeinV3::getInstance()->all_del();
	mapUsers.clear();
	if (bSave)
		save();
#else

#endif

	return -1;
}

//获取可用的用户id
int CUers::getUsableID(){
#ifndef GA
	int id = 1;
	for (; id <= mapUsers.size(); id++){
		if (mapUsers.find(id) == mapUsers.end())
			return id;
	}
	return mapUsers.size() + 1;
#else

#endif
}

/*
根据用户名，获取用户id,由于允许用户名重复，所以存在同一个用户名可能有多个用户id
*/
int CUers::getIdByUser(string user, vector<int> &ids){
#ifndef GA
//	int id = 0;
	map<int, string>::iterator it = mapUsers.begin();
	for (; it != mapUsers.end(); it++)
	{
		if (it->second == user){
		//	id = it->first;
			ids.push_back(it->first);
		//	break;

		}
	}
	return ids.size() == 0 ? -1 : ids.front();
#else

#endif
}

/*
根据用户名，获取用户模板在模板库中的索引位置,由于允许用户名重复，所以存在同一个用户名可能有多个用户索引
*/
int CUers::getIndexByUser(string user, vector<int> &indexs){
#ifndef GA
	int index = 0;
	map<int, string>::iterator it = mapUsers.begin();
	for (; it != mapUsers.end(); it++)
	{
		if (it->second == user){
			indexs.push_back(index);
		//	break;
		}
		index++;
	}
	return indexs.size() == 0 ? -1 : indexs.front();
#else

#endif
}

//根据用户名，获取此用户的模板数量
int CUers::getFeatureCountByUser(string user){
#ifndef GA
	vector<int> ids;
	int id = getIdByUser(user, ids), count = 0;
	for(auto it=ids.begin();it!=ids.end();it++)
		count += CFingerVeinV3::getInstance()->get_feature_count_by_id(*it);

	return count;
#else

#endif
}

//根据用户id，获取用户名
string CUers::getUserById(int id){
#ifndef GA
	if (mapUsers.find(id) != mapUsers.end()){
		return mapUsers[id];
	}
	return "";
#else

#endif
}

//根据用户模板在模板库中的索引位置，获取用户名
string CUers::getUserByIndex(int index){
#ifndef GA
	if (index >= mapUsers.size()){
		return "";
	}
	map<int, string>::iterator it = mapUsers.begin();
	int i = 0;
	for (; it != mapUsers.end(); it++)
	{
		if (i++ == index){
			return it->second;
		}
	}

	return "";
#else

#endif
}

int  CUers::addNewUser(const char *path,const char *name)
{
	//检查名字
	map<int,string>::iterator it=mapUsers.begin();
	for (;it!=mapUsers.end();it++)
	{
		if(it->second == name){
			return -1;
		}
	}

	int adduser= CFingerVeinV3::getInstance()->loadNewUser(path);
	if (adduser != -1)
	{
		CString Userpath = CString(path);
		Userpath = Userpath.Right(Userpath.GetLength()-1-Userpath.ReverseFind('\\'));
		USES_CONVERSION;
		std::string _strPath((LPCSTR)Userpath.GetBuffer());
		int UserID = atoi((const char *)_strPath.c_str());
		{
			mapUsers[UserID]=name;
			save();
			return UserID;
		}
	}
	return -1;
}