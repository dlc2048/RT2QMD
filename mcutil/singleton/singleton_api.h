// The MIT License(MIT)
// 
// Copyright(c) 2021 xhawk18 - at - gmail.com
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this softwareand associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
// 
// The above copyright noticeand this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/*
* Singleton API from https://github.com/xhawk18/singleton-cpp/
*/

#pragma once
#ifndef INC_SINGLETON_API_H_
#define INC_SINGLETON_API_H_

#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(singleton_EXPORTS) // add by CMake 
#    ifdef __GNUC__
#      define  SINGLETON_API __attribute__(dllexport)
#    else
#      define  SINGLETON_API __declspec(dllexport)
#    endif
#  else
#    ifdef __GNUC__
#      define  SINGLETON_API __attribute__(dllimport)
#    else
#      define  SINGLETON_API __declspec(dllimport)
#    endif
#  endif // singleton_EXPORTS

#elif defined __GNUC__
#  if __GNUC__ >= 4
#    define SINGLETON_API __attribute__ ((visibility ("default")))
#  else
#    define SINGLETON_API
#  endif

#elif defined __clang__
#  define SINGLETON_API __attribute__ ((visibility ("default")))

#else
#   error "Do not know how to export classes for this platform"
#endif

#endif