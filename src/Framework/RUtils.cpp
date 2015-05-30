#include "RUtils.hpp"

static void chkIntFn(void *dummy)
{
  R_CheckUserInterrupt();
}

bool checkInterrupt() {
  return (R_ToplevelExec(chkIntFn,NULL) == false);
}
