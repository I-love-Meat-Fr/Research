#pragma once

#include <map>
#include <string>
#include <vector>
#include "json/json.h"
using std::vector;
using std::map;
using std::string;


class CUers
{
	
public:
	CUers(void);
	~CUers(void);

	 void   load();
	void   save();
	bool   isExist(unsigned char *img, int len);
	int   addUser(string user, vector<unsigned char>img1, vector<unsigned char>img2, vector<unsigned char>img3, BOOL bSave = TRUE, BOOL bCheckDup = TRUE);
	string find_user(unsigned char *img, int len,int &index, float *score);
	int    delUser(int index, BOOL bSave = TRUE);
	int  addNewUser(const char *path,const char *name);

	//yejs
	int getUsableID();										 // Get the available user id
	int getIdByUser(string user, vector<int> &ids);			 // Obtain the user id based on the user name. The same user name may have multiple user ids because user names are allowed to be repeated
	int getIndexByUser(string user, vector<int> &indexs);	 // Obtain the index location of the user template in the template library based on the user name. The same user name may have multiple user indexes because user names are allowed to be repeated
	int getFeatureCountByUser(string user);					 // Obtain the number of user templates based on the user name
	string getUserById(int id);								 // Obtain the user name based on the user id
	string getUserByIndex(int index);						 // Obtain the user name based on the index position of the user template in the template library
	
	int   addUser(int index);
	int   addUser(string user, int index);
	int    delAllUser(BOOL bSave = TRUE);

	bool   isExist(string user);
	std::map<int, std::string>mapUsers;
	Json::Value json_object;
};

