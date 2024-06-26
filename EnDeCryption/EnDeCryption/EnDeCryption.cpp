// EnDeCryption.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include<iostream>
#include<fstream>
using namespace std;
//using std::fstream;
//using std::streamoff;

int main(int argc, char* argv[])//argc是main内使用参数个数+1,argv[0]是文件路径，argv[1]是文件名
{
	try
	{
		fstream BeforeFile, AfterFile;//代加密或解密文件的流对象
	//保存代加密或解密文件
		char fileNameBefore[256] = { 0 }, fileNameAfter[256] = { 0 };;

		//3打开代加密或解密文件					
		BeforeFile.open(argv[1], ios::in | ios::binary);//读|字节流
		//4打开加密或解密后文件
		AfterFile.open(argv[2], ios::out | ios::binary);

		//5获取代加密或解密文件大小	
		BeforeFile.seekg(0, ios::end);//3.1定位文件内容指针到末尾
		streamoff  size = BeforeFile.tellg();	//3.2获取大小
		BeforeFile.seekg(0, ios::beg);//3.3 定位文件内容指针到文件头

		for (streamoff i = 0; i < size; i++) {
			//把BeforeFile的内容处理后放入AfterFile中
			AfterFile.put(BeforeFile.get() ^ 0x88);//加密或解密编码  0x88 
		}
#if 0
		char temp;
		for (std::streamoff i = 0; i < size; i++)
		{
			//6逐字节读取
			temp = BeforeFile.get();
			//7加密
			temp ^= 0x88;
			//8写入解密后文件
			AfterFile.put(temp);
		}
#endif	

		//9关闭两个文件
		BeforeFile.close();
		AfterFile.close();
		string removeflag = "false";
		if (argc > 3) {
			removeflag = argv[3];
		}
		if (removeflag != "false") {
			remove(argv[1]);
		}
		printf("endecryption success\n");
	}
	catch (char* str)        // 捕获所有异常
	{
		printf("endecryption failed %s\n", str);
	}
	return 0;
}