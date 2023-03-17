cmake_minimum_required(VERSION 3.15)

# Set the C/C++ specified in the projects as requirements
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

project(micromesh_core VERSION 1.0.0)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(UNIX)
  add_definitions(-DLINUX)
endif()

# XXXX we might be able to force this by setting CMAKE_GENERATOR_PLATFORM ??
if(NOT (CMAKE_SIZEOF_VOID_P EQUAL 8))
    message( FATAL_ERROR "64-bit builds only : reconfigure with -A x64" )
endif()

set(CMAKE_DEBUG_POSTFIX "_d")
set(CMAKE_RELWITHDEBINFO_POSTFIX "_rd")

find_package(Threads REQUIRED)

#
# build
#

set(PUBLIC_HEADER_FILES
    include/micromesh/micromesh_api.h
    include/micromesh/micromesh_types.h
    include/micromesh/micromesh_gpu.h
    include/micromesh/micromesh_utils.h
    include/micromesh/micromesh_operations.h
    include/micromesh/micromesh_format_types.h
)
install(FILES ${PUBLIC_HEADER_FILES} DESTINATION "include")

set(INTERNAL_HEADER_FILES
    include/micromesh_internal/micromesh_context.h
    include/micromesh_internal/micromesh_containers.h
    include/micromesh_internal/micromesh_math.h
)

set(SOURCE_FILES
    src/context.cpp
    src/convert.cpp
    src/utils.cpp
    src/sanitize.cpp
    src/micromap.cpp
    src/mesh.cpp
)

source_group("public_include" FILES ${PUBLIC_HEADER_FILES})
source_group("internal_include" FILES ${INTERNAL_HEADER_FILES})
source_group("source" FILES ${SOURCE_FILES})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_BINARY_DIR}")

# static lib

add_library(micromesh_core STATIC ${SOURCE_FILES} ${PUBLIC_HEADER_FILES} ${INTERNAL_HEADER_FILES})
target_link_libraries(micromesh_core PUBLIC Threads::Threads) # So other projects link with the system thread library

set_target_properties(micromesh_core PROPERTIES FOLDER "micromesh_core"
                                     POSITION_INDEPENDENT_CODE ON)

target_include_directories(micromesh_core PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>)    

install(TARGETS micromesh_core DESTINATION "lib" EXPORT "micromeshCoreTargets")

# tests

set(MICROMESH_CORE_BUILD_TESTS OFF CACHE BOOL "Build the unit tests for micromesh core")
if(MICROMESH_CORE_BUILD_TESTS)
    set(TEST_FILES test/test_core.cpp)
    source_group("tests" FILES ${TEST_FILES})
    add_executable(micromesh_core_test ${TEST_FILES})
    target_link_libraries(micromesh_core_test micromesh_core)
    set_target_properties(micromesh_core_test PROPERTIES FOLDER "micromesh_core")
endif()

install(EXPORT "micromeshCoreTargets" FILE "micromeshCoreTargets.cmake" EXPORT_LINK_INTERFACE_LIBRARIES DESTINATION "cmake")

#
# package target
#

set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
include(CPack)
