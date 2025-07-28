# Check if has not commited changes
execute_process(COMMAND ${GIT_EXECUTABLE} update-index -q --refresh
    ERROR_QUIET)
execute_process(COMMAND ${GIT_EXECUTABLE} diff-index --name-only HEAD --
    OUTPUT_VARIABLE CHANGED_SOURCE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)

if(CHANGED_SOURCE)
  set(DIRTINESS "dirty")
  message(STATUS "Source tree is dirty")
else(CHANGED_SOURCE)
  set(DIRTINESS "clean")
  message(STATUS "Source tree is clean")
endif(CHANGED_SOURCE)

set(MUST_UPDATE_DIRTINESS ON)
if(EXISTS "${INCL_VERSIONING_HEADER}")
  file(READ "${INCL_VERSIONING_HEADER}" PREVIOUS_DIRTINESS
       LIMIT 1024)
  if(PREVIOUS_DIRTINESS MATCHES ${DIRTINESS})
    set(MUST_UPDATE_DIRTINESS OFF)
  endif(PREVIOUS_DIRTINESS MATCHES ${DIRTINESS})
endif(EXISTS "${INCL_VERSIONING_HEADER}")

if(MUST_UPDATE_DIRTINESS)
  message(STATUS "Updating versioning header ${INCL_VERSIONING_HEADER}")
  configure_file("${INCL_BINARY_DIR}/G4INCLVersion.hh.in"
                 "${INCL_VERSIONING_HEADER}"
                 @ONLY)
endif(MUST_UPDATE_DIRTINESS)
