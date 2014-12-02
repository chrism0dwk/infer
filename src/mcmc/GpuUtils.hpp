/**************************************************************************
 *  This file is part of Infer.                                           *
 *                                                                        *
 *   Infer is free software: you can redistribute it and/or modify        *
 *   it under the terms of the GNU General Public License as published by *
 *   the Free Software Foundation, either version 3 of the License, or    *
 *   (at your option) any later version.                                  *
 *                                                                        *
 *   Foobar is distributed in the hope that it will be useful,            *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of       *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        *
 *   GNU General Public License for more details.                         *
 *                                                                        *
 *   You should have received a copy of the GNU General Public License    *
 *   along with Infer.  If not, see <http://www.gnu.org/licenses/>.       *
 *************************************************************************/

#ifndef GPUUTILS_HPP
#define GPUUTILS_HPP

#include <stdexcept>
#include <sstream>
#include <cuda_runtime.h>

 
namespace EpiRisk {

  class GpuRuntimeError : public std::exception
  {
  public:
    GpuRuntimeError(const std::string usrMsg, cudaError_t cudaErr)
    {
      msg_ = "GPU Runtime Error: ";
      msg_ += usrMsg;
      msg_ += " (";
      msg_ += cudaErr;
      msg_ += ",";
      msg_ += cudaGetErrorString(cudaErr);
      msg_ += ")";
    }
    ~GpuRuntimeError() throw ()
    {
    }
    ;
    const char*
    what() const throw ()
    {
      return msg_.c_str();
    }

  private:
    std::string msg_;
  };



  void
  __checkCudaError(const cudaError_t err, const char* file, const int line);
#define checkCudaError(err) __checkCudaError(err, __FILE__, __LINE__)


}



#endif
