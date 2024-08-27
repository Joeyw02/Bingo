#include<iostream>
#include"apps.h"
#include"type.h"
using namespace std;
int main(int argc, char* argv[]){
    freopen("../dataset/AM","r",stdin);
    if(argc<2){
        instance(app::deepwalk);
        return 0;
    }
    switch (argv[1][0]){
        case 'd':
            instance(app::deepwalk);
            break;
        case 'n':
            instance(app::node2vec);
            break;
        case 'p':
            instance(app::ppr);
            break;
        case 's':
            instance(app::sampling);
            break;
        default:
            instance(app::deepwalk);
            break;
    }
    fclose(stdin);
    return 0;
}