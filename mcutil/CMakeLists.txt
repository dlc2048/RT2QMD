
function(add_mcutil_cuda_library target_name_base target_name_var)

  set( target_name ${target_name_base} )
  set( ${target_name_var} ${target_name} PARENT_SCOPE )
  
  object_add_source_groups()
  
  CUDA_GET_SOURCES_AND_OPTIONS(source_files cmake_options options ${ARGN})
  
  CUDA_ADD_LIBRARY(
    ${target_name}
    SHARED
    ${source_files}
	${cmake_options}
	OPTIONS ${options}
  )
  
  set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
  set_target_properties(${target_name} PROPERTIES FOLDER "mcutil") 
  target_link_libraries(${target_name} ${CUDA_LIBRARIES})
endfunction()


function(add_mcutil_cpp_library target_name)
  grab_sources(${target_name} target_name_sources)
  
  add_library(
    ${target_name}
	SHARED
	${target_name_sources}
  )
  
  set_target_properties(${target_name} PROPERTIES FOLDER "mcutil") 
endfunction()

object_add_source_groups()

grab_sources(device device_sources)
add_mcutil_cuda_library( device device_target ${device_sources} OPTIONS ${RT2QMD_SOURCE_OPTIONS} )

add_mcutil_cpp_library( fortran   )
add_mcutil_cpp_library( singleton )
add_mcutil_cpp_library( mclog     )
add_mcutil_cpp_library( parser    )
add_mcutil_cpp_library( prompt    )

target_link_libraries( mclog    singleton )
target_link_libraries( fortran  mclog )
target_link_libraries( prompt   mclog )
target_link_libraries( parser   mclog prompt )
target_link_libraries( device   parser GSL::gsl GSL::gslcblas )

target_include_directories( mclog   PRIVATE "${CMAKE_SOURCE_DIR}/mcutil")
target_include_directories( fortran PRIVATE "${CMAKE_SOURCE_DIR}/mcutil")
target_include_directories( prompt  PRIVATE "${CMAKE_SOURCE_DIR}/mcutil")
target_include_directories( parser  PRIVATE "${CMAKE_SOURCE_DIR}/mcutil")
target_include_directories( device  PRIVATE "${CMAKE_SOURCE_DIR}/mcutil")