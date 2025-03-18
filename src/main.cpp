#include <iostream>
#include "apps.h"
#include "type.h"
using namespace std;
void setApplication(app &a, char c)
{
    switch (c)
    {
    case 'd':
        a = app::deepwalk;
        break;
    case 'n':
        a = app::node2vec;
        break;
    case 'p':
        a = app::ppr;
        break;
    case 's':
        a = app::sampling;
        break;
    }
}
int main(int argc, char *argv[])
{
    string s = "AM";
    app a = app::deepwalk;
    utype u = utype::M;
    if (argc < 2)
    {
        instance(a, u, s);
        return 0;
    }
    int pos = 1;
    if (argv[pos][0] != '-')
        setApplication(a, argv[pos][0]), ++pos;
    while (pos < argc)
    {
        if (argv[pos][0] != '-')
        {
            ++pos;
            continue;
        }
        switch (argv[pos][1])
        {
        case 'g':
        {
            ++pos;
            if (pos >= argc)
                break;
            int len = strlen(argv[pos]);
            s.clear();
            for (int i = 0; i < len; ++i)
                s.push_back(argv[pos][i]);
            break;
        }
        case 'd':
            u = utype::D;
            break;
        case 'i':
            u = utype::I;
            break;
        }
        ++pos;
    }
    instance(a, u, s);
    fclose(stdin);
    return 0;
}