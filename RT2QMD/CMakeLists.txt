#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

#list(APPEND source_subdir "geometry"           )
#list(APPEND source_subdir "ptx"                )
#
#set(source)
#
#foreach(folder ${source_subdir})
#  file(GLOB sub_sources 
#    "${folder}/*.hpp" 
#	"${folder}/*.h" 
#	"${folder}/*.cpp" 
#	"${folder}/*.cu" 
#	"${folder}/*.cuh"
#  )
#  set(source ${source} ${sub_sources})
#endforeach()

#set (source ${source} 
#  "${CMAKE_SOURCE_DIR}/module/photon/xs_dev.cu" 
#)


function(add_cuda_executable_program target_name_base target_name_var)
  set(target_name ${target_name_base})
  set(${target_name_var} ${target_name} PARENT_SCOPE)
  
  object_add_source_groups()

  CUDA_GET_SOURCES_AND_OPTIONS(source_files cmake_options options ${ARGN})

  CUDA_ADD_EXECUTABLE(
    ${target_name_base}
    ${source_files}
	${cmake_options}
	OPTIONS ${options}
  )
  set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
  target_link_libraries(${target_name} ${CUDA_LIBRARIES})
endfunction()


include_directories( "${CMAKE_SOURCE_DIR}/mcutil" )
include_directories( "${CMAKE_SOURCE_DIR}/module" )


add_cuda_executable_program( RT2QMD target_name
  ${source}
  mcrt2_rt2qmd_standalone.cpp
  auxiliary.hpp
  auxiliary.cpp
  auxiliary.cuh
  auxiliary.cu
  aux_score.hpp
  aux_score.cpp
  aux_score.cuh
  aux_score.cu
  OPTIONS ${RT2QMD_SOURCE_OPTIONS}
)

target_link_libraries( RT2QMD
  ${CUDA_LIBRARIES}
  device
  mclog
  parser
  fortran
  prompt
  particles
  scoring
  transport
  qmd
  deexcitation
  nuc_secondary
  genericion
)

# target_include_directories( RT2QMD  PRIVATE "${CMAKE_SOURCE_DIR}/mcutil")
# target_include_directories( RT2QMD  PRIVATE "${CMAKE_SOURCE_DIR}/module")

install(TARGETS ${target_name} DESTINATION bin)