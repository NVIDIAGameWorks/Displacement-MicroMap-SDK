cmake_minimum_required(VERSION 3.15)

# Set the C/C++ specified in the projects as requirements
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

project(micromesh_displacement_remeshing VERSION 1.0.0)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/vulkan.cmake)

find_path(GLM_INCLUDE_DIR
  NAMES glm/glm.hpp
  HINTS ${GLM_LOCATION} $ENV{GLM_LOCATION} ${VK_SDK_PATH}/include ${VULKANSDK_INCLUDE_DIR}
)
if(NOT GLM_INCLUDE_DIR)
  message(FATAL_ERROR "GLM not found. Aborting. Set `GLM_INCLUDE_DIR` or include `GLM` with the Vulkan SDK install")
endif()

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
    include/micromesh/micromesh_displacement_remeshing.h
)
install(FILES ${PUBLIC_HEADER_FILES} DESTINATION "include")

set(INTERNAL_HEADER_FILES

)

_add_package_VulkanSDK()

set(SOURCE_FILES
    src/remeshing_cpu.cpp
    src/remeshing_gpu.cpp
    src/remeshing_internal.h
)

source_group("public_include" FILES ${PUBLIC_HEADER_FILES})
source_group("internal_include" FILES ${INTERNAL_HEADER_FILES})
source_group("source" FILES ${SOURCE_FILES})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" "${GLM_INCLUDE_DIR}")

#----------------------------------
# Adding shader files
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
file(GLOB SHD_HDR  ${SHD_DIR}/*.h)
file(GLOB SHD_SRC  ${SHD_DIR}/*.rchit ${SHD_DIR}/*.rmiss ${SHD_DIR}/*.frag ${SHD_DIR}/*.rahit ${SHD_DIR}/*.rgen ${SHD_DIR}/*.vert ${SHD_DIR}/*.comp ${SHD_DIR}/*.glsl)

# Compiling shaders to Spir-V header
compile_glsl(
    SOURCE_FILES ${SHD_SRC}
    HEADER_FILES ${SHD_HDR}
    DST "${CMAKE_CURRENT_SOURCE_DIR}/_autogen"
    VULKAN_TARGET "vulkan1.2"
    HEADER ON 
    DEPENDENCY ${VULKAN_BUILD_DEPENDENCIES}
    FLAGS "-I${SHD_DIR}" -g
    )
#target_sources(${PROJECT_NAME} PRIVATE ${SHD_SRC} ${SHD_HDR})
source_group("shaders" FILES ${SHD_SRC} ${SHD_HDR})


#----------------------------------



# static lib

add_library(micromesh_displacement_remeshing STATIC ${SOURCE_FILES} ${PUBLIC_HEADER_FILES} ${INTERNAL_HEADER_FILES} ${SHD_SRC} ${SHD_HDR}) 
target_link_libraries(micromesh_displacement_remeshing micromesh_core)

set_target_properties(micromesh_displacement_remeshing PROPERTIES FOLDER "micromesh_displacement_remeshing")

target_include_directories(micromesh_displacement_remeshing PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>)    

install(TARGETS micromesh_displacement_remeshing DESTINATION "lib" EXPORT "micromeshDisplacementRemeshingTargets")

# tests

set(MICROMESH_DISPLACEMENT_REMESHING_BUILD_TESTS OFF CACHE BOOL "Build the unit tests for micromesh_displacement_remeshing")
if(MICROMESH_DISPLACEMENT_REMESHING_BUILD_TESTS)
    set(TEST_FILES test/test_remeshing.cpp)
    source_group("tests" FILES ${TEST_FILES})
    add_executable(micromesh_displacement_remeshing_test ${TEST_FILES})
    target_link_libraries(micromesh_displacement_remeshing_test micromesh_displacement_remeshing micromesh_core)
    set_target_properties(micromesh_displacement_remeshing_test PROPERTIES FOLDER "micromesh_displacement_remeshing")
endif()

install(EXPORT "micromeshDisplacementRemeshingTargets" FILE "micromeshDisplacementRemeshingTargets.cmake" EXPORT_LINK_INTERFACE_LIBRARIES DESTINATION "cmake")

#
# package target
#

set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
include(CPack)
