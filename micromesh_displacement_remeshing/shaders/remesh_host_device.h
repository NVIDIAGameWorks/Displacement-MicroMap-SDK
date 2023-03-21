/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/


#ifdef __cplusplus
#pragma once
#include "glm/glm.hpp"

#define MAIN void MAIN_NAME(const uvec3& gl_GlobalInvocationID, RemesherData* rmData)
#define RM_DATA_ARG RemesherData *rmData,
#define RM_DATA(x_) rmData->x_
#define RM_DATA_VAL rmData,
#define RM_CONSTANTS RM_DATA(constants)
#define TEXTURE(t, c) vec4(0.f)

namespace micromesh
{
#else
#define MAIN void main()
#define RM_DATA_ARG
#define RM_DATA(x_) x_
#define RM_DATA_VAL
#define RM_CONSTANTS constants
#define TEXTURE(t, c) texture(t, c)

#endif

#ifndef REMESHER_HOST_DEVICE_H
#define REMESHER_HOST_DEVICE_H

#define REMESHER_BLOCK_SIZE 256

#define REMESHER_TANGENT_INDEX 4

// clang-format off
#ifdef __cplusplus // GLSL Type
using uint = uint32_t;
using uvec2 = glm::uvec2;
using uvec3 = glm::uvec3;
using ivec2 = glm::ivec2;
using mat4 = glm::mat4;
#endif


#ifdef __cplusplus
#define REF(t_) t_ &
#else
#define REF(t_) inout t_
#endif


#define RM_H_NOT_FOUND ~0u
#define RM_H_NOT_SET ((~0u)-1)

#define RM_V_MARKED (0x1 << 0)
#define RM_V_ORPHAN (0x1 << 1)
#define RM_V_FIXED (0x1 << 2)
#define RM_V_COPIED (0x1 << 3)
#define RM_V_DOUBLE_MARKED (0x1 << 4)
#define RM_V_EDGE (0x1 << 5)
#define RM_V_UNKNOWN (0x1 << 6)
#define RM_V_DEBUG (0x1 << 7)

#ifdef __cplusplus
enum RemesherMode
{
#else
const uint
#endif
eDecimate = 0,
eOptimizeForDisplacement = 2,
eGenerateMicromeshInfo = 3
#ifdef __cplusplus
};
#else
;
#endif


#ifdef __cplusplus
// typedef to have the needed sizeof
// With optimized displacement direction storage
// Disabled for now
typedef struct { uint v[5]; } RemesherVertex;
#else
#define RemesherVertex uint
#endif


#ifdef __cplusplus
// typedef to have the needed sizeof
typedef struct { uint v[3]; } RemesherEdge;
#else
#define RemesherEdge uint
#endif

#ifdef __cplusplus
// typedef to have the needed sizeof
typedef struct { uint v[3]; } RemesherHashEntry;
#else
#define RemesherHashEntry uint
#endif
struct RemesherConstants
{
    ivec2 dispMapResolution;

    uint vertexCount;
    uint indexCount;
    uint hashMapSize;
    uint edgeListSize;
    float errorThreshold;
    float deduplicationThreshold;

    uint compactionPass;
    float curvatureImportance;
    uint remeshingMode;

    uint iterationIndex;

    uint clampedSubdLevel;

    uint backupPositions;

    uint activeVertices;

    uint isFinalCompaction;

    uint maxValence;
    float maxImportance;

    uint texcoordCount;
    uint texcoordIndex;

    float directionBoundsFactor;

};



#ifdef __cplusplus
typedef struct { uint v[9]; } RemesherTriangle;
#else
#define RemesherTriangle uint
#endif


#define RM_MICROMESH_DISP_BOUNDS_DATA_SIZE 2
#define RM_MICROMESH_DISP_DIR_DATA_SIZE 3
#ifdef __cplusplus
// typedef to have the needed sizeof
typedef struct { int v[RM_MICROMESH_DISP_DIR_DATA_SIZE]; } RemesherMicromeshDisplacementDir;
#else
#define RemesherMicromeshDisplacementDir float
#endif


#ifdef __cplusplus
// typedef to have the needed sizeof
typedef struct {
    uint16_t decimationFlags;
    uint16_t subdLevel;
    float    minDisplacement;
    float    maxDisplacement;
} RemesherMicromeshInfo;
#else
#define RemesherMicromeshInfo uint
#endif

#ifndef __cplusplus
// MUST be an exact replica of micromesh::RemesherErrorState in micromesh/micromesh_displacement_remeshing.h
#define eRemesherErrorNone                 0x0
#define eRemesherErrorVertexHashNotFound   0x1
#define eRemesherErrorEdgeHashNotFound     0x2
#define eRemesherErrorDebug                0x3
#define eRemesherErrorOutOfEdgeStorage     0x5
#define eRemesherErrorNoTriangleFound      0x6
#define eRemesherErrorNoVertexHistoryFound 0x7
#define eRemesherErrorInvalidConstantValue 0x8
#endif

struct RemesherMetadata
{

    uint collapsedEdgesCount;

    uint edgeCount;

    uint vertexCompactionValidEntries;
    uint vertexCompactionCurrentInvalidEntry;
    uint vertexCompactionCurrentValidEntry;

    uint indexCompactionValidEntries;
    uint indexCompactionCurrentInvalidEntry;
    uint indexCompactionCurrentValidEntry;


    uint uncompactedTriangleCount;
    uint uncompactedVertexCount;

    uint activeVertices;

};

// Otherwise, this struct is defined in micromesh_displacement_remeshing.h
#ifndef __cplusplus
// Copy of the struct defined in micromesh_displacement_remeshing.h
struct RemeshingCurrentState
{
    uint triangleCount;
    uint vertexCount;
    uint errorState;
    uint mergeCount;
    uvec4 debug;
};
#endif

#ifdef __cplusplus
enum RemesherBindings
{
#else
const uint
#endif
    // User-provided data, modified in place (copy of enum gpu::GpuRemeshingResource)

    // main modified buffers
    // -------------------------
    // must be pre-filled with appropriate data prior task begin
    // will be updated continuously during the process.
    // 3 x float per-vertex
    eGpuRemeshingMeshVertexPositionsBuffer=0,
    // 2 x float per-vertex
    eGpuRemeshingMeshVertexTexcoordsBuffer=1,
    // 2 x uint per-vertex
    eGpuRemeshingMeshVertexHashBuffer=2,
    // 3 x uint per-triangle
    eGpuRemeshingMeshTrianglesBuffer=3,
    // 1 x uint per-triangle (e.g. per-triangle component/material assignments etc.)
    // (optional `GpuRemeshing_config::useTriangleUserIDs`)
    eGpuRemeshingMeshTriangleUserIDsBuffer=4,

    // 1 x float per-vertex (optional `GpuRemeshing_config::useVertexCurvature`)
    eGpuRemeshingMeshVertexImportanceBuffer=5,

    // output buffers
    // -------------------------
    //
    // 1 x uint { uint16 subdivlevel, uint16 edgeflags} per-triangle
    // (optional `OpRemeshing_settings::generateDisplacementInfo`, only in eDecimate mode)
    eGpuRemeshingMeshTriangleSubdivisionInfoBuffer=6,
    // 3 x float per-vertex
    // (optional `OpRemeshing_settings::generateDisplacementInfo`, only in eDecimate mode)
    eGpuRemeshingMeshVertexDirectionsBuffer=7,
    // 2 x float per-vertex
    // (optional `OpRemeshing_settings::generateDisplacementInfo`, only in eDecimate mode)
    eGpuRemeshingMeshVertexDirectionBoundsBuffer=8,

    // intermediate buffers used during process
    // ----------------------------------------
    // 3 x uint per-vertex as below
    // RemeshingVertexMergeInfo {
    //  uint32_t vertexIndexA;
    //  uint32_t vertexIndexB;
    //  float    blendAtoB;
    // }
    eGpuRemeshingMeshVertexMergeBuffer=9,

    // 1 x uint per-vertex
    eGpuRemeshingDebugVertexBuffer=10,
    // 1 x uint per-triangle
    eGpuRemeshingDebugTriangleBuffer=11,

    // 1 RemeshingCurrentState struct, used for feedback
    eGpuRemeshingCurrentStateBuffer=12,


    // Scratch buffers, opaque to the user (see remesher::ScratchBuffers enum)
    eScratchIndexBuffer = 13,
    eScratchTriangles = 14,
    eScratchMetadata = 15,
    eScratchHashMap = 16,
    eScratchEdgeList = 17,
    eScratchVertices = 18,
    eScratchTrianglesDesc = 19,
    eScratchVertexAliases = 20,
    eScratchVertexOriginalPos = 21,
    eScratchActiveVertices = 23
#ifdef __cplusplus
, eBindingCount
};
#else
;
#endif

#ifdef __cplusplus
} // namespace micromesh
#endif


#endif // REMESHER_HOST_DEVICE_H