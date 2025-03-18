#include<iostream>
using namespace std;
int main(){
    float a[5]={18.34,29.99, 36.49,109.78, 2510.34},b[5]={0.27 , 0.69 ,  0.46 ,  3.27 , 19.95};
    cerr<<(a[1]/b[1]+a[0]/b[0]+a[2]/b[2]+a[3]/b[3]+a[4]/b[4])/5<<endl;
}