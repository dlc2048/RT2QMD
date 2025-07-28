
function(add_module_cuda_library target_name_base target_name_var)

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
  
  target_link_libraries(${target_name} ${CUDA_LIBRARIES})
  set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
  set_target_properties(${target_name} PROPERTIES FOLDER "module") 
endfunction()


function(add_module_cpp_library target_name)
  grab_sources(${target_name} target_name_sources)
  
  add_library(
    ${target_name}
	SHARED
	${target_name_sources}
  )
  
  set_target_properties(${target_name} PROPERTIES FOLDER "module") 
endfunction()

include_directories( "${CMAKE_SOURCE_DIR}/mcutil" )
include_directories( "${CMAKE_SOURCE_DIR}/module" )

object_add_source_groups()

add_module_cpp_library( particles )

grab_sources( physics physics_sources )
add_module_cuda_library( physics   physics_target   ${physics_sources}   OPTIONS ${RT2QMD_SOURCE_OPTIONS} )

grab_sources( transport transport_sources )
add_module_cuda_library( transport transport_target ${transport_sources} OPTIONS ${RT2QMD_SOURCE_OPTIONS} )

grab_sources( scoring scoring_sources )
add_module_cuda_library( scoring   scoring_target   ${scoring_sources}   OPTIONS ${RT2QMD_SOURCE_OPTIONS} )

grab_sources( hadron hadron_sources )
add_module_cuda_library( hadron      hadron_target      ${hadron_sources}      OPTIONS ${RT2QMD_SOURCE_OPTIONS} )

grab_sources( qmd qmd_sources)
add_module_cuda_library( qmd qmd_target ${qmd_sources} OPTIONS ${RT2QMD_SOURCE_OPTIONS} )

grab_sources( deexcitation deexcitation_sources )
add_module_cuda_library( deexcitation deexcitation_target ${deexcitation_sources} OPTIONS ${RT2QMD_SOURCE_OPTIONS} )

grab_sources( nuc_secondary nuc_secondary_sources )
add_module_cuda_library( nuc_secondary nuc_secondary_target ${nuc_secondary_sources} OPTIONS ${RT2QMD_SOURCE_OPTIONS} )

grab_sources( genericion genericion_sources ) 
add_module_cuda_library( genericion  genericion_target  ${genericion_sources} OPTIONS ${RT2QMD_SOURCE_OPTIONS} )

target_link_libraries( physics         singleton )
target_link_libraries( particles       parser physics device )
target_link_libraries( hadron          particles fortran device prompt )
target_link_libraries( transport       device particles )
target_link_libraries( scoring         fortran transport hadron )
target_link_libraries( qmd             fortran prompt device mclog transport hadron )
target_link_libraries( deexcitation    fortran prompt device mclog transport hadron )
target_link_libraries( nuc_secondary   particles fortran device prompt hadron )
target_link_libraries( genericion      particles fortran device prompt hadron deexcitation nuc_secondary )
