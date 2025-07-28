# - Finds ROOT instalation
# This module sets up ROOT information 
# It defines:
# ROOT_FOUND          If the ROOT is found
# ROOT_INCLUDE_DIR    PATH to the include directory
# ROOT_ETC_DIR        PATH to the etc directory
# ROOT_LIBRARIES      Most common libraries
# ROOT_LIBRARY_DIR    PATH to the library directory 
#
# Modified by Davide Mancusi to take into account minimum version requirements


find_program(ROOT_CONFIG_EXECUTABLE root-config PATHS $ENV{ROOTSYS}/bin NO_DEFAULT_PATH)
find_program(ROOT_CONFIG_EXECUTABLE root-config)

if(NOT ROOT_CONFIG_EXECUTABLE)
  set(ROOT_FOUND FALSE)
else()    

  execute_process(
    COMMAND ${ROOT_CONFIG_EXECUTABLE} --version 
    OUTPUT_VARIABLE ROOT_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  string(REPLACE "/" "." ROOT_VERSION_PROCESSED ${ROOT_VERSION})
  string(REPLACE "/" "." ROOT_FIND_VERSION_PROCESSED ${ROOT_FIND_VERSION})

  if("${ROOT_VERSION_PROCESSED}" VERSION_LESS "${ROOT_FIND_VERSION_PROCESSED}")
    # This version is not high enough.
    if(ROOT_FIND_REQUIRED)
      message(SEND_ERROR "Found ROOT version ${ROOT_VERSION} < ${ROOT_FIND_VERSION}")
    else(ROOT_FIND_REQUIRED)
      if(NOT ROOT_FIND_QUIETLY)
        message(STATUS "Found ROOT version ${ROOT_VERSION} < ${ROOT_FIND_VERSION}")
      endif(NOT ROOT_FIND_QUIETLY)
      set(ROOT_FOUND FALSE)
    endif(ROOT_FIND_REQUIRED)
  else("${ROOT_VERSION_PROCESSED}" VERSION_LESS "${ROOT_FIND_VERSION_PROCESSED}")
    set(ROOT_FOUND TRUE)

    execute_process(
      COMMAND ${ROOT_CONFIG_EXECUTABLE} --prefix 
      OUTPUT_VARIABLE ROOTSYS 
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    execute_process(
      COMMAND ${ROOT_CONFIG_EXECUTABLE} --incdir
      OUTPUT_VARIABLE ROOT_INCLUDE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    execute_process(
      COMMAND ${ROOT_CONFIG_EXECUTABLE} --libs
      OUTPUT_VARIABLE ROOT_LIBRARIES
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    execute_process(
      COMMAND ${ROOT_CONFIG_EXECUTABLE} --etcdir
      OUTPUT_VARIABLE ROOT_ETC_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    execute_process(
      COMMAND ${ROOT_CONFIG_EXECUTABLE} --libdir
      OUTPUT_VARIABLE ROOT_LIBRARY_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    # Make variables changeble to the advanced user
    mark_as_advanced(ROOT_CONFIG_EXECUTABLE)

    # Set the path to the valgrind suppression file
    set(ROOT_MEMORYCHECK_SUPPRESSIONS_FILE "${ROOT_ETC_DIR}/valgrind-root.supp" CACHE FILEPATH "Path to the valgrind suppression file for ROOT")
    mark_as_advanced(ROOT_MEMORYCHECK_SUPPRESSIONS_FILE)

    if(NOT ROOT_FIND_QUIETLY)
      message(STATUS "Found ROOT ${ROOT_VERSION} in ${ROOTSYS}")
    endif()

  endif("${ROOT_VERSION_PROCESSED}" VERSION_LESS "${ROOT_FIND_VERSION_PROCESSED}")
endif()

