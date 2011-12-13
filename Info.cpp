#include "Info.h"


Info::Info(void)
  : diagonal_(0),
    offset_(0)
{
}


Info::Info(int diagonal, int offset=0)
  : diagonal_(diagonal),
    offset_(offset)
{
}

Info::~Info(void)
{
}
