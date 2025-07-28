if(CMAKE_VERSION VERSION_LESS 2.8.3)
  # Macro from the CMake wiki -- slightly modified
  MACRO(PARSE_ARGUMENTS prefix option_names arg_names)
    SET(DEFAULT_ARGS)

    SET(current_arg_name UNPARSED_ARGUMENTS)
    SET(current_arg_list)
    FOREACH(arg ${ARGN})
      SET(larg_names ${arg_names})
      LIST(FIND larg_names "${arg}" is_arg_name)
      IF (is_arg_name GREATER -1)
        SET(${prefix}_${current_arg_name} ${current_arg_list})
        SET(current_arg_name ${arg})
        SET(current_arg_list)
      ELSE (is_arg_name GREATER -1)
        SET(loption_names ${option_names})
        LIST(FIND loption_names "${arg}" is_option)
        IF (is_option GREATER -1)
          SET(${prefix}_${arg} TRUE)
        ELSE (is_option GREATER -1)
          SET(current_arg_list ${current_arg_list} ${arg})
        ENDIF (is_option GREATER -1)
      ENDIF (is_arg_name GREATER -1)
    ENDFOREACH(arg)
    SET(${prefix}_${current_arg_name} ${current_arg_list})
  ENDMACRO(PARSE_ARGUMENTS)
else(CMAKE_VERSION VERSION_LESS 2.8.3)
  include(CMakeParseArguments)
endif(CMAKE_VERSION VERSION_LESS 2.8.3)

function(parse_line LINE)

  set(OPTIONS "")
  set(ONE_VALUE_ARGS SEPARATORS COMMENT)
  if(CMAKE_VERSION VERSION_LESS 2.8.3)
    parse_arguments(PARSE_LINE "${OPTIONS}" "${ONE_VALUE_ARGS}" ${ARGN})
  else(CMAKE_VERSION VERSION_LESS 2.8.3)
    set(MULTI_VALUE_ARGS "")
    cmake_parse_arguments(PARSE_LINE "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN})
  endif(CMAKE_VERSION VERSION_LESS 2.8.3)

  # Default separators: space, tab
  if(NOT PARSE_LINE_SEPARATORS)
    set(PARSE_LINE_SEPARATORS " 	")
  endif(NOT PARSE_LINE_SEPARATORS)

  # Default comment character: #
  if(NOT PARSE_LINE_COMMENT)
    set(PARSE_LINE_COMMENT "#")
  endif(NOT PARSE_LINE_COMMENT)

  string(STRIP ${LINE} LINE_STRIPPED)

  string(SUBSTRING ${LINE_STRIPPED} 0 1 FIRST_CHARACTER)

  if(NOT FIRST_CHARACTER MATCHES "[${PARSE_LINE_COMMENT}]")

    list(GET PARSE_LINE_UNPARSED_ARGUMENTS -1 LAST_VAR)
    foreach(VAR ${PARSE_LINE_UNPARSED_ARGUMENTS})
      if(VAR STREQUAL LAST_VAR)
        set(TOKEN "${LINE_STRIPPED}")
        string(REGEX MATCH "[^${PARSE_LINE_SEPARATORS}].*" TOKEN "${LINE_STRIPPED}")
      else(VAR STREQUAL LAST_VAR)
        string(REGEX MATCH "[^${PARSE_LINE_SEPARATORS}]+" TOKEN "${LINE_STRIPPED}")
      endif(VAR STREQUAL LAST_VAR)

      if(NOT TOKEN)
        message(SEND_ERROR "Not enough tokens on the following line: ${LINE}")
        return()
      endif(NOT TOKEN)

      # Set the variable
      set(${VAR} "${TOKEN}" PARENT_SCOPE)

      # Remove the token from the line we are parsing
      # first escape square brackets...
      string(REPLACE "[" "\\[" TOKEN "${TOKEN}")
      string(REPLACE "]" "\\]" TOKEN "${TOKEN}")
      string(REGEX REPLACE "${TOKEN}[${PARSE_LINE_SEPARATORS}]+(.*)$" "\\1" LINE_STRIPPED "${LINE_STRIPPED}")
    endforeach(VAR "${PARSE_LINE_UNPARSED_ARGUMENTS}")

  else(NOT FIRST_CHARACTER MATCHES "[${PARSE_LINE_COMMENT}]")
    foreach(VAR ${PARSE_LINE_UNPARSED_ARGUMENTS})
      set(${VAR} "NOTFOUND" PARENT_SCOPE)
    endforeach(VAR ${PARSE_LINE_UNPARSED_ARGUMENTS})
  endif(NOT FIRST_CHARACTER MATCHES "[${PARSE_LINE_COMMENT}]")

endfunction(parse_line LINE)
