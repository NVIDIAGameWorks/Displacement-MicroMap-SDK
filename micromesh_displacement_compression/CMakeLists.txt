cmake_minimum_required(VERSION 3.15)

# Set the C/C++ specified in the projects as requirements
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

project(micromesh_displacement_compression VERSION 1.0.0)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)


if (NOT TARGET micromesh_core AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../micromesh_core)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../micromesh_core  ${CMAKE_BINARY_DIR}/micromesh_core)
endif()


if(UNIX)
  add_definitions(-DLINUX)
endif()

# XXXX we might be able to force this by setting CMAKE_GENERATOR_PLATFORM ??
if(NOT (CMAKE_SIZEOF_VOID_P EQUAL 8))
    message( FATAL_ERROR "64-bit builds only : reconfigure with -A x64" )
endif()

set(CMAKE_DEBUG_POSTFIX "_d")
set(CMAKE_RELWITHDEBINFO_POSTFIX "_rd")

#
# build
#

set(PUBLIC_HEADER_FILES
    include/micromesh/micromesh_displacement_compression.h
)
install(FILES ${PUBLIC_HEADER_FILES} DESTINATION "include")

set(INTERNAL_HEADER_FILES
    src/cpp_compatibility.h
    src/displacement_block_codec.h
    src/displacement_configs.h
    src/displacement_diagnostic_utils.h
    src/displacement_mesh_codec.h
)

set(SOURCE_FILES
    src/cpp_compatibility.cpp
    src/displacement_block_codec.cpp
    src/displacement_compression.cpp
    src/displacement_configs.cpp
    src/displacement_mesh_codec.cpp
)

source_group("public_include" FILES ${PUBLIC_HEADER_FILES})
source_group("internal_include" FILES ${INTERNAL_HEADER_FILES})
source_group("source" FILES ${SOURCE_FILES})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_BINARY_DIR}")

# static lib

add_library(micromesh_displacement_compression STATIC ${SOURCE_FILES} ${PUBLIC_HEADER_FILES} ${INTERNAL_HEADER_FILES})
target_link_libraries(micromesh_displacement_compression micromesh_core)

set_target_properties(micromesh_displacement_compression PROPERTIES FOLDER "micromesh_displacement_compression"
                                                                    POSITION_INDEPENDENT_CODE ON)

target_include_directories(micromesh_displacement_compression PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>)    

install(TARGETS micromesh_displacement_compression DESTINATION "lib" EXPORT "micromeshDisplacementCompressionTargets")

# tests

set(MICROMESH_DISPLACEMENT_COMPRESSION_BUILD_TESTS OFF CACHE BOOL "Build the unit tests for micromesh_displacement_compression")
if(MICROMESH_DISPLACEMENT_COMPRESSION_BUILD_TESTS)
    set(TEST_FILES test/test_compression.cpp)
    source_group("tests" FILES ${TEST_FILES})
    add_executable(micromesh_displacement_compression_test ${TEST_FILES})
    target_link_libraries(micromesh_displacement_compression_test micromesh_displacement_compression micromesh_core)
    set_target_properties(micromesh_displacement_compression_test PROPERTIES FOLDER "micromesh_displacement_compression")
endif()

install(EXPORT "micromeshDisplacementCompressionTargets" FILE "micromeshDisplacementCompressionTargets.cmake" EXPORT_LINK_INTERFACE_LIBRARIES DESTINATION "cmake")

#
# package target
#

set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
include(CPack)
