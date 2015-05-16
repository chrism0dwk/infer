##### 
#
# SYNOPSIS
#
# AX_CHECK_CUDA
#
# DESCRIPTION
#
# Figures out if CUDA Runtime API/nvcc is available, i.e. existence of:
#   cudart.h
#   libcudart.so
#   nvcc
#
# If something isn't found, fails straight away.
#
# Locations of these are included in 
#   CUDA_CPPFLAGS and 
#   CUDA_LDFLAGS.
# Path to nvcc is included as
#   NVCC_PATH
# in config.h
# 
#
# LICENCE
# Public domain
#
# AUTHORS
# wili, Chris Jewell <c.jewell@lancaster.ac.uk>
#
##### 

AC_DEFUN([AX_CHECK_CUDA], [

# Provide your CUDA path with this		
AC_ARG_WITH(cuda, [AS_HELP_STRING([--with-cuda=PREFIX],
		                 [Prefix of your CUDA installation @<:@default=/usr/local/cuda@:>@])],
				 [CUDA_DIR=$withval],
				 [CUDA_DIR="/usr/local/cuda"])

# Setting the prefix to the default if only --with-cuda was given
if test "$CUDA_DIR" == "yes"; then
	if test "$withval" == "yes"; then
		CUDA_DIR="/usr/local/cuda"
	fi
fi

# Checking for nvcc
AC_MSG_CHECKING([nvcc in $CUDA_DIR/bin])
if test -x "$CUDA_DIR/bin/nvcc"; then
	AC_MSG_RESULT([found])
	NVCC="$CUDA_DIR/bin/nvcc"
	AC_DEFINE([HAVE_CUDA],1,[Has cuda framework])
else
	AC_MSG_RESULT([not found!])
	AC_MSG_FAILURE([nvcc was not found in $CUDA_DIR/bin])
fi

# We need to add the CUDA search directories for header and lib searches

# Check header
CUDA_CPPFLAGS="-I$CUDA_DIR/include"
SAVED_CPPFLAGS=${CPPFLAGS}
CPPFLAGS=${CUDA_CPPFLAGS}
AC_CHECK_HEADER([cuda_runtime.h], [], AC_MSG_ERROR([Couldn't find cuda_runtime.h]), [#include <cuda_runtime.h>])
CPPFLAGS=${SAVED_CPPFLAGS}

# Check for lib in CUDA_DIR/lib64 (Linux) and CUDA_DIR/lib (Darwin/Linux32?)
if test -d "${CUDA_DIR}/lib64"; then
   AC_MSG_NOTICE([Using lib64 CUDA])
   CUDA_LDFLAGS="-L${CUDA_DIR}/lib64"
else
   AC_MSG_NOTICE([Using lib CUDA])
   CUDA_LDFLAGS="-L${CUDA_DIR}/lib"
fi
AC_MSG_NOTICE([CUDA LDFLAGS: ${CUDA_LDFLAGS}])
SAVED_LDFLAGS=${LDFLAGS}
LDFLAGS=${CUDA_LDFLAGS}
AC_CHECK_LIB([cudart], [cudaRuntimeGetVersion],,[AC_MSG_ERROR([Cannot find CUDA library.  Check your CUDA installation, and/or supply --with-cuda= to configure])])
LDFLAGS=${SAVED_LDFLAGS}


]) # DEFUN
