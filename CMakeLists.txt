
# vcpkg for WIN32 (optinally possible in UNIX)
set(USE_VCPKG OFF CACHE BOOL "Use vcpkg for third-party packages")

if(USE_VCPKG)
  set(VCPKG_PATH "" CACHE PATH "Vcpkg path")
  set(CMAKE_TOOLCHAIN_FILE "${VCPKG_PATH}/scripts/buildsystems/vcpkg.cmake")
  
  if( WIN32 )
    set(VCPKG_TARGET_TRIPLET
    "x64-windows"
    )
  else()
    if( UNIX AND NOT APPLE )
      set(VCPKG_TARGET_TRIPLET
      "x64-linux"
      )
    else()
      set(VCPKG_TARGET_TRIPLET
      "x64-osx"
      )
    endif()
  endif()
endif()
  
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# This enforces a particular version of CMake that we require to process the script files
# properly.  We rely on VERSION_GREATER_EQUAL which requires CMake v. 3.7.
cmake_minimum_required(VERSION 3.7 FATAL_ERROR)

# As of CMake 2.6 policies were introduced in order to provide a mechanism for
# adding backwards compatibility one feature at a time.  We will just specify
# that all policies will use version 2.8.12 semantics.
cmake_policy(VERSION 2.8.12)

if( POLICY CMP0072 )
  # FindOpenGL prefers GLVND by default when available
  cmake_policy(SET CMP0072 NEW)
endif()

if( POLICY CMP0074 )
  # find_package uses <PackageName>_ROOT variables.
  cmake_policy(SET CMP0074 NEW)
endif()

if( POLICY CMP0077 )
  cmake_policy(SET CMP0077 NEW)
endif()

if( POLICY CMP0102 )
  cmake_policy(SET CMP0102 NEW)
endif()

if( POLICY CMP0126 )
  cmake_policy(SET CMP0126 NEW)
endif()

project(RT2QMD C CXX)

# CPP options

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
set(CUDA_VERBOSE_BUILD ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# RT2QMD options and definitions
option( MCRT2_NAN_THROW "Throw exception if NAN value is detected" OFF)
option( MCRT2_RT2QMD_STANDALONE "RT2QMD standalone mode" ON) # DO NOT OFF this parameter in RT2QMD standalone package
option( MCRT2_PERSISTENT_QMD_INITIALIZER "RT2QMD field dispatcher, using persistent loop" ON)

if ( MCRT2_RT2QMD_STANDALONE )
  add_compile_definitions(RT2QMD_STANDALONE)
endif()

if( MCRT2_NAN_THROW )
  add_compile_definitions(CUDA_THROW_NAN)
endif()

if( MCRT2_PERSISTENT_QMD_INITIALIZER )
  add_compile_definitions(RT2QMD_PERSISTENT_DISPATCHER)
endif()

if( MSVC )
  add_compile_options(/wd4819)
endif()


# Add paths to our CMake code to the module path, so they can be found automatically by
# CMake.
set(CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/CMake"
  ${CMAKE_MODULE_PATH}
  )

# Set the default build to Release.  Note this doesn't do anything for the VS
# default build target which defaults to Debug when you first start it.
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# Tells CMake to build all the libraries as shared libraries by default.  This can be
# overrided by individual libraries later.
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

##########
# Process our custom setup scripts here.

# Enable C++17. Needs to be done before the include of ConfigCompilerFlags.cmake below.
set(GCC_LIBSTDCPP17 ON)

# Include all CMake Macros.
include (Macros)
# Determine information about the compiler
include (CompilerInfo)
# Check for specific machine/compiler options.
include (ConfigCompilerFlags)

# Turn off the warning that NVCC issues when generating PTX from our CUDA samples.  This
# is a custom extension to the FindCUDA code distributed by CMake.
OPTION(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)

# For Xcode 5, gcc is actually clang, so we have to tell CUDA to treat the compiler as
# clang, so that it doesn't mistake it for something else.
if(USING_CLANG_C)
  set(CUDA_HOST_COMPILER "clang" CACHE FILEPATH "Host side compiler used by NVCC")
endif()

# CUDA 8 is broken for generating dependencies during configure
option(CUDA_GENERATE_DEPENDENCIES_DURING_CONFIGURE "Generate dependencies during configure time instead of only during build time." OFF)

# Find at least a 11.0 version of CUDA.
find_package(CUDA 11.0 REQUIRED)

# Find GSL
find_package(GSL REQUIRED)

# Determine if we are going to use the static CRT on windows.
if(WIN32)
  option(RELEASE_USE_STATIC_CRT "Build using the static CRT library" OFF)
endif()

# Helper for fixing compiler flags
function(replace_flag var old_flag new_flag)
  string(REPLACE "${old_flag}" "${new_flag}" ${var} ${${var}})
  set(${var} "${${var}}" CACHE STRING "Default compiler flags" FORCE)
endfunction()

function(replace_flags old_flag new_flag)
  foreach(build "" _DEBUG _MINSIZEREL _RELEASE _RELWITHDEBINFO)
    replace_flag(CMAKE_C_FLAGS${build}   "${old_flag}" "${new_flag}")
    replace_flag(CMAKE_CXX_FLAGS${build} "${old_flag}" "${new_flag}")
  endforeach()
endfunction()

if(WIN32)
  replace_flags("/GS" "/GS-")
  if(RELEASE_USE_STATIC_CRT)
    replace_flags("/MD" "/MT")
  else()
    replace_flags("/MT" "/MD")
  endif()
endif(WIN32)

# Present the CUDA_64_BIT_DEVICE_CODE on the default set of options.
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)

set(CUDA_MIN_SM_TARGET sm_61 CACHE STRING "Minimum CUDA SM architecture to use for compilation.")

# Add some useful default arguments to the NVCC and NVRTC flags.  This is an example of
# how we use PASSED_FIRST_CONFIGURE.  Once you have configured, this variable is TRUE
# and following block of code will not be executed leaving you free to edit the values
# as much as you wish from the GUI or from ccmake.
if( NOT PASSED_FIRST_CONFIGURE )
  set(CUDA_NVCC_FLAGS_DESCRIPTION "Semi-colon delimit multiple arguments.")
  string(REPLACE "sm_" "compute_" CUDA_MIN_SM_COMPUTE_TARGET ${CUDA_MIN_SM_TARGET})

  list(FIND CUDA_NVCC_FLAGS "-arch" index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS -arch ${CUDA_MIN_SM_TARGET})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
  endif()
  
  if(CMAKE_CXX_STANDARD EQUAL 17)
    set(SAMPLES_NVRTC_CXX "-std=c++17")
  else()
    set(SAMPLES_NVRTC_CXX "")
  endif()
  set(CUDA_NVRTC_FLAGS ${SAMPLES_NVRTC_CXX} -arch ${CUDA_MIN_SM_COMPUTE_TARGET} -use_fast_math -lineinfo -default-device -rdc true -D__x86_64 CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
endif()

mark_as_advanced(CUDA_NVRTC_FLAGS)

set(RT2QMD_SOURCE_OPTIONS "-rdc=true" "-Xptxas" "-O3" "--use_fast_math")
 
# This passes a preprocessor definition to cl.exe when processing CUDA code.
if(USING_WINDOWS_CL)
  list(APPEND CUDA_NVCC_FLAGS --compiler-options /D_USE_MATH_DEFINES)
endif()

# Put all the runtime stuff in the same directory.  By default, CMake puts each targets'
# output into their own directory.  We want all the targets to be put in the same
# directory, and we can do this by setting these variables.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Create a flag for mac which will allow apps to add the local cuda toolkit
# install path to the app's rpath.
if( APPLE )
  set( CUDA_TOOLKIT_RPATH_FLAG "-Wl,-rpath,${CUDA_TOOLKIT_ROOT_DIR}/lib" )
endif()

# Select whether to use NVRTC or NVCC to generate PTX
set(CUDA_NVRTC_ENABLED OFF CACHE BOOL "Use NVRTC to compile PTX at run-time instead of NVCC at build-time")
if( CUDA_NVRTC_ENABLED AND SAMPLES_INPUT_GENERATE_OPTIXIR )
  if( NOT SAMPLES_INPUT_GENERATE_PTX )
    message( SEND_ERROR "CUDA_NVRTC_ENABLED is only compatible with SAMPLES_INPUT_GENERATE_PTX. Please disable CUDA_NVRTC_ENABLED unless enabling SAMPLES_INPUT_GENERATE_PTX." )
  endif()
endif()

#########################################################

# RT2 modules
add_subdirectory( mcutil )
add_subdirectory( module )

# Module tester (event generator standalone)
add_subdirectory( RT2QMD )

# resource
add_subdirectory( resource        )

# packaging
# add_subdirectory( packaging       )

# example
# file(COPY "example" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

#################################################################

# Now that everything is done, indicate that we have finished configuring at least once.
# We use this variable to set certain defaults only on the first pass, so that we don't
# continually set them over and over again.
set(PASSED_FIRST_CONFIGURE ON CACHE INTERNAL "Already Configured once?")


