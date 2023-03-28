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
#include "remesh_host_device.h"
#include <atomic>
using namespace micromesh;
#endif


#ifndef __cplusplus
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

#ifndef REMESHER_OVERRIDE_BLOCK_SIZE
layout(local_size_x = REMESHER_BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;
#endif

layout(push_constant) uniform RemesherConstants_
{
    RemesherConstants constants;
};


// User-provided data, modified in place (copy of enum gpu::GpuRemeshingResource)

// main modified buffers
// -------------------------
// must be pre-filled with appropriate data prior task begin
// will be updated continuously during the process.
// 4 x float per-vertex - 4th is unused
layout(binding = eGpuRemeshingMeshVertexPositionsBuffer, set = 0) buffer GpuRemeshingMeshVertexPositionsBuffer
{
    vec4 vertexPositions[];
};

// texcoordCount x 2 x float per-vertex
layout(binding = eGpuRemeshingMeshVertexTexcoordsBuffer, set = 0) buffer GpuRemeshingMeshVertexTexcoordsBuffer
{
    vec2 vertexTexCoords[];
};

// 2 x uint per-vertex
layout(binding = eGpuRemeshingMeshVertexHashBuffer, set = 0) buffer GpuRemeshingMeshVertexHashBuffer
{
    uvec2 vertexHash[];
};


// 3 x uint per-triangle
layout(binding = eGpuRemeshingMeshTrianglesBuffer, set = 0) buffer GpuRemeshingMeshTrianglesBuffer
{
    uint triangles[];
};


// 1 x uint per-triangle (e.g. per-triangle component/material assignments etc.)
// (optional `GpuRemeshing_config::useTriangleUserIDs`)
layout(binding = eGpuRemeshingMeshTriangleUserIDsBuffer, set = 0) buffer GpuRemeshingMeshTriangleUserIDsBuffer
{
    uint trianglesUserID[];
};


// 1 x float16 per-vertex (optional `GpuRemeshing_config::useVertexCurvature`)
// Addressing using f16vec2 for alignment, need care when accessing x or y component (even or odd index)
layout(binding = eGpuRemeshingMeshVertexImportanceBuffer, set = 0) buffer GpuRemeshingMeshVertexImportanceBuffer
{
    f16vec2 vertexImportances[];
};

// output buffers
// -------------------------
//
// 1 x uint { uint16 subdivlevel, uint16 edgeflags} per-triangle
// (optional `OpRemeshing_settings::generateDisplacementInfo`, only in eDecimate mode)
layout(binding = eGpuRemeshingMeshTriangleSubdivisionInfoBuffer, set = 0) buffer GpuRemeshingMeshTriangleSubdivisionInfoBuffer
{
    uint triangleSubdivisionInfo[];
};

// 4 x float16 per-vertex - 4th is unused
// (optional `OpRemeshing_settings::generateDisplacementInfo`, only in eDecimate mode)
layout(binding = eGpuRemeshingMeshVertexDirectionsBuffer, set = 0) buffer GpuRemeshingMeshVertexDirectionsBuffer
{
    f16vec4 vertexDirections[];
};


// 2 x float per-vertex
// (optional `OpRemeshing_settings::generateDisplacementInfo`, only in eDecimate mode)
layout(binding = eGpuRemeshingMeshVertexDirectionBoundsBuffer, set = 0) buffer GpuRemeshingMeshVertexDirectionBoundsBuffer
{
    vec2 vertexDirectionBounds[];
};
layout(binding = eGpuRemeshingMeshVertexDirectionBoundsBuffer, set = 0) buffer GpuRemeshingMeshVertexDirectionBoundsBufferI
{
    uvec2 vertexDirectionBoundsU[];
};


// intermediate buffers used during process
// ----------------------------------------
// 3 x uint per-vertex as below
// RemeshingVertexMergeInfo {
//  uint32_t vertexIndexA;
//  uint32_t vertexIndexB;
//  float    blendAtoB;
// }
layout(binding = eGpuRemeshingMeshVertexMergeBuffer, set = 0) buffer GpuRemeshingMeshVertexMergeBuffer
{
    uint vertexMerges[];
};


// 1 x uint per-vertex
layout(binding = eGpuRemeshingDebugVertexBuffer, set = 0) buffer GpuRemeshingDebugVertexBuffer
{
    uint vertexDebug[];
};

// 1 x uint per-triangle
layout(binding = eGpuRemeshingDebugTriangleBuffer, set = 0) buffer GpuRemeshingDebugTriangleBuffer
{
    uint triangleDebug[];
};


// 1 RemeshingCurrentState struct, used for feedback
layout(binding = eGpuRemeshingCurrentStateBuffer, set = 0) buffer GpuRemeshingCurrentStateBuffer
{
    RemeshingCurrentState currentState;
};


// Scratch buffers, opaque to the user (see remesher::ScratchBuffers enum)
layout(binding = eScratchIndexBuffer, set = 0) buffer ScratchIndexBuffer
{
    uint scratchIndices[];
};

// Scratch buffers, opaque to the user (see remesher::ScratchBuffers enum)
layout(binding = eScratchIndexBuffer, set = 0) buffer ScratchIndexBuffer128
{
    uvec4 scratchIndices128[];
};


layout(binding = eScratchTriangles, set = 0) buffer ScratchTriangles
{
    uint scratchTriangles[];
};


layout(binding = eScratchMetadata, set = 0) buffer ScratchMetadata
{
    RemesherMetadata scratchMetadata;
};


layout(binding = eScratchHashMap, set = 0) buffer ScratchHashMap
{
    uint scratchHashMap[];
};

layout(binding = eScratchHashMap, set = 0) buffer ScratchHashMap128
{
    uvec4 scratchHashMap128[];
};


layout(binding = eScratchEdgeList, set = 0) buffer ScratchEdgeList
{
    uint scratchEdges[];
};


layout(binding = eScratchVertices, set = 0) buffer ScratchVertices
{
    uint scratchVertices[];
};

layout(binding = eScratchVertices, set = 0) buffer ScratchVerticesF
{
    float scratchVerticesF[];
};


layout(binding = eScratchTrianglesDesc, set = 0) buffer ScratchTrianglesDesc
{
    uint64_t scratchTriangleDescs[];
};


layout(binding = eScratchVertexAliases, set = 0) buffer ScratchVertexAliases
{
    uint scratchVertexAliases[];
};

layout(binding = eScratchVertexOriginalPos, set = 0) buffer ScratchVertexOriginalPos
{
    vec4 scratchVertexOriginalPos[];
};

layout(binding = eScratchActiveVertices, set = 0) buffer ScratchActiveVerticesBuffer
{
    uint scratchActiveVertices[];
};


#else
// CPU bindings - unused since CPU fallback is not currently supported
struct RemesherData
{
    const uint*           inputIndices;
    const uint*           inputVertices;
    uint*                 outputIndices;
    uvec2*                tempIndices;
    uint*                 outputVertices;
    RemesherMetadata      outputMetadata;
    std::atomic_uint*     edgesHash;
    std::atomic_uint*     edgesList;
    std::atomic_uint*     triangles;
    std::atomic_uint64_t* trianglesDesc;
    std::atomic_uint*     vertices;

    int*  verticesI = (int*)vertices;
    uint* tempVertices;

    uint*            vertexAliases;
    float*           originalVertexPos;
    std::atomic_int* micromeshInfo;
    std::atomic_int* micromeshInfoBackup;
    float*           micromeshDispDir;


    RemesherConstants constants;
};
#endif
