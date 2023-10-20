#include <stdio.h>
#include<vector>
#include<iostream>
#include <math.h>
using namespace std;
void independnum(int num, int levels, int& numvalue, int& numflag)
{
	if (num % levels != 0 && num / levels !=0)
	{
		numflag = 1;
	}
	if ( num / levels !=0)
	{
		numvalue += 1;
	}
	else{
		return;
	}
	independnum(num / levels, levels, numvalue, numflag);
}

void table(int independvalue, int num, int levels, int factor, vector<vector<int> >& b, const vector<vector<int> >& group_vec)
{
	//int num = (levels - 1) * factor + 1;
	int nu = num / levels;
	for (int i = 0; i < nu; i++)
	{
		for (int j = 0; j < levels; j++)
		{
			for (int dp = 0; dp < independvalue; dp++)
			{
				for (int k = 0; k < num / pow(levels, (independvalue - dp)); k++)
				{
					b[j * num / pow(levels, dp + 1) + i / pow(levels, dp) + k * num / pow(levels, dp)][dp] = j;
				}
			}
			//for (int k = 0; k < nu / levels/levels; k++)
			//{
			//	b[j * nu + i][0] = j;
			//}
			//
			//for (int k = 0; k < nu/levels; k++)
			//{
			//	b[j * nu / levels + i / levels+nu*k][1] = j;
			//}
			//for (int k = 0; k < nu; k++)
			//{
			//	b[j * nu / levels / levels + i / levels / levels + nu / levels * k][2] = j;
			//}

		}
	}

	for (int i = independvalue; i < factor; i++)
	{
		for (int j = 0; j < num; j++)
		{
			int sum = 0;
			for (int k = 0; k < group_vec[i - independvalue].size(); k++)
			{
				sum += b[j][group_vec[i - independvalue][k]];
			}
			b[j][i] = sum % levels;
			/*if (i == 3)
			{
				b[j][3] = (b[j][1] + b[j][0]) % levels;

				b[j][4] = (b[j][1] + b[j][3]) % levels;

				b[j][5] = (b[j][2] + b[j][0]) % levels;
				b[j][6] = (b[j][2] + b[j][1]) % levels;
				b[j][7] = (b[j][2] + b[j][3]) % levels;
				b[j][8] = (b[j][2] + b[j][4]) % levels;

				b[j][9] = (b[j][2] + b[j][5]) % levels;
				b[j][10] = (b[j][2] + b[j][6]) % levels;
				b[j][11] = (b[j][2] + b[j][7]) % levels;
				b[j][12] = (b[j][2] + b[j][8]) % levels;
			}*/

		}

	}
	cout << "正交表为：" << endl;
	for (int j = 0; j < num; j++)
	{
		for (int i = 0; i < factor; i++)
			printf("%d  ", b[j][i]+1);
		printf("\n");
	}
}



unsigned int nextPow2(int num)
{
	num--;
	num |= num >> 1;
	num |= num >> 2;
	num |= num >> 4;
	num |= num >> 8;
	num |= num >> 16;
	num++;
	return num;
}

void combine_increase( int start, int* result, int count, const int NUM, const int independvalue, vector<vector<int>>& group_vec, const int factor)
 {
	if (group_vec.size() >= (factor - independvalue))
		return;
   int i = 0;
   for (i = start; i < independvalue + 1 - count; i++)
   {
     result[count - 1] = i;
     if (count - 1 == 0)
     {
       int j;
	   vector<int> grouptmp;
	   for (j = NUM - 1; j >= 0; j--)
	   {
		   grouptmp.push_back(result[j]);
	   }
	   group_vec.push_back(grouptmp);
     }
     else
		combine_increase(i + 1, result, count - 1, NUM, independvalue, group_vec, factor);
   }
 }

int fac(int n) 
{ 
	int sum; 
	if (n == 0 || n == 1) 
		{ sum = 1; } 
	if (n >= 2) 
		{ sum = n * fac(n - 1); } 
	return sum; 
}

void groupvecadd(int levels, int factor , int independvalue, vector<vector<int> >& group_vec)
{
	if (levels > 2 && group_vec.size() < factor - independvalue)
	{
		int tolnum = 0;
		for (int i = 2; i <= independvalue; i++)
		{
			tolnum += fac(independvalue) / fac(i) / fac(independvalue - i);
		}
		for (int i = levels - 2, grouppre = 0; i > 0; i--, grouppre = group_vec.size() - 1)
		{
			cout << i << endl;
			for (int j = 1; j < independvalue; j++)
			{
				for (int k = independvalue + grouppre + j - 1; k < independvalue + grouppre + j - 1 + tolnum; k++)
				{
					vector<int> grouptmp;
					grouptmp.push_back(j);
					grouptmp.push_back(k);
					cout <<k << grouppre << endl;
					group_vec.push_back(grouptmp);
					if (group_vec.size() >= factor - independvalue)
						return;
				}
			}
		}
	}
}

void table(int levels, int factor)
{
	int  numvalue = 0, numflag = 0;
	independnum((levels - 1) * factor + 1, levels, numvalue, numflag);
	int independvalue = numvalue + numflag;
	int num = pow(levels, independvalue);
	cout << "total num :" << num << endl;
	vector<vector<int>> a(num);
	for (int i = 0; i < num; i++)
	{
		a[i].resize(factor);
	}


	//cout << "independvalue :" << independvalue << endl;
	vector<vector<int>> group_vec;
	for (int i = 2; i <= independvalue; i++)
	{
		int* result = new int[i];
		combine_increase(0, result, i, i, independvalue, group_vec, factor);
		delete[]result;
		if (group_vec.size() >= factor - independvalue)
			break;
	}
	//cout << "group vector size: " << group_vec.size() << endl;
	groupvecadd(levels, factor, independvalue, group_vec);
	if (group_vec.size() < factor - independvalue)
	{
		cout << "add group vector size error: " << group_vec.size() << endl;
		return;
	}

	//for (vector<vector<int>>::iterator it = group_vec.begin(); it < group_vec.end(); it++)
	//{
	//	cout << "total group vector size: " << group_vec.size() << endl;
	//	for (vector<int>::iterator vit = (*it).begin(); vit < (*it).end(); vit++)
	//	{
	//		cout << *vit << " ";
	//	}
	//	cout << endl;
	//}

	table(independvalue, num, levels, factor, a, group_vec);
}
void main()
{
	int levels, factor;
	cout << "输入水平数:" << endl;
	cin >> levels;
	cout << "输入因子数:" << endl;
	cin >> factor;
	table(levels, factor);
}