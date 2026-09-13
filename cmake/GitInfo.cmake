# Captures git identity at configure time for reproducibility metadata.
# Missing git or non-git trees yield "unknown" placeholders.

function(capsnet_get_git_info)
    set(CAPSNET_GIT_COMMIT "unknown" PARENT_SCOPE)
    set(CAPSNET_GIT_BRANCH "unknown" PARENT_SCOPE)
    set(CAPSNET_GIT_DIRTY "0" PARENT_SCOPE)

    find_package(Git QUIET)
    if(NOT GIT_FOUND)
        return()
    endif()

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE _commit
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _rc
    )
    if(_rc EQUAL 0)
        set(CAPSNET_GIT_COMMIT "${_commit}" PARENT_SCOPE)
    endif()

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE _branch
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _rc
    )
    if(_rc EQUAL 0)
        set(CAPSNET_GIT_BRANCH "${_branch}" PARENT_SCOPE)
    endif()

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" status --porcelain
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE _status
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _rc
    )
    if(_rc EQUAL 0 AND NOT "${_status}" STREQUAL "")
        set(CAPSNET_GIT_DIRTY "1" PARENT_SCOPE)
    endif()
endfunction()
