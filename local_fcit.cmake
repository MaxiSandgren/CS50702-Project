# Loaded by VAMP's project() call. Define the module after its targets exist.
function(add_course_fcit)
    if(NOT TARGET Python::SABIModule OR Python_VERSION VERSION_LESS 3.12)
        message(FATAL_ERROR "local_fcit needs CMake 3.26+ and Python 3.12+ to match VAMP's stable-ABI wheel.")
    endif()
    get_filename_component(course_dir "${CMAKE_PROJECT_vamp_INCLUDE}" DIRECTORY)
    nanobind_add_module(local_fcit NB_STATIC STABLE_ABI NOMINSIZE "${course_dir}/local_fcit.cc")
    target_link_libraries(local_fcit PRIVATE vamp_cpp)
    set_target_properties(local_fcit PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${course_dir}")
endfunction()

cmake_language(DEFER CALL add_course_fcit)
