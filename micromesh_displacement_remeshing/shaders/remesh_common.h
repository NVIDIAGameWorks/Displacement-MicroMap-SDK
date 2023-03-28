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
#endif
#include "remesh_host_device.h"

#ifndef __cplusplus
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_ray_query : require
#extension GL_EXT_shader_atomic_float : enable
#endif
#include "remesh_bindings.h"


#define REMESHER_ACCESSORS_ALL
#define REMESHER_INTERNAL
#include "remesh_accessors.h"


#define M_PI 3.1415926535897932384626433832795


uint encodePreviousTriangle(uint triangleIndex, uint localVertexRef)
{
    const uint mask = 0x3;
    return ((triangleIndex & ~(mask << 30)) | ((localVertexRef & mask) << 30));
}

uvec2 decodePreviousTriangle(uint previousTriangle)
{
    const uint mask = 0x3;
    return uvec2((previousTriangle & ~(mask << 30)), previousTriangle >> 30);
}

#define getActiveVertex(arg0_) getActiveVertexImpl(RM_DATA_VAL arg0_)
uint getActiveVertexImpl(RM_DATA_ARG uint threadIndex)
{
    if(RM_DATA(scratchMetadata).activeVertices == 0)
    {
        if(threadIndex >= RM_CONSTANTS.vertexCount)
            return ~0u;
        return threadIndex;
    }

    if(threadIndex >= RM_DATA(scratchMetadata).activeVertices)
        return ~0u;
    return RM_DATA(scratchActiveVertices)[threadIndex];
}

#define hasEdge(arg0_, arg1_) hasEdgeImpl(RM_DATA_VAL arg0_, arg1_)
bool hasEdgeImpl(RM_DATA_ARG uint triIndex, uint edgeIndex)
{
    for(uint i = 0; i < 3; i++)
        if(rtGetEdgeIndex(triIndex, i) == edgeIndex)
            return true;
    return false;
}

uint subdivLeveGetSegmentCount(uint subdivLevel)
{
    return (1u << subdivLevel);
}

uint subdivLeveGetTriangleCount(uint subdivLevel)
{
    return (1u << (subdivLevel * 2));
}

uint subdivLevelGetVertexCount(uint subdivLevel)
{
    uint vertsPerEdgeCount = subdivLeveGetSegmentCount(subdivLevel) + 1;
    return (vertsPerEdgeCount * (vertsPerEdgeCount + 1)) / 2;
}


float mixCurvatures(float c1, float c2)
{
    float t = 0.f;
    if(c1 == 0.f && c2 != 0.f)
        t = c2;
    if(c1 != 0.f && c2 == 0.f)
        t = c1;
    if(c1 != 0.f && c2 != 0.f)
        t = max(c1, c2);

    return t;
}


#define cvGetOutputPosition(arg0_) cvGetOutputPositionImpl(RM_DATA_VAL arg0_)
vec3 cvGetOutputPositionImpl(RM_DATA_ARG uint index)
{
    return RM_DATA(vertexPositions)[index].xyz;
}


#define cvGetOutputTexCoord(arg0_) cvGetOutputTexCoordImpl(RM_DATA_VAL arg0_)
vec2 cvGetOutputTexCoordImpl(RM_DATA_ARG uint index)
{
    vec2 tex = RM_DATA(vertexTexCoords)[RM_CONSTANTS.texcoordCount * index + RM_CONSTANTS.texcoordIndex];
    return tex;
}

#define cvGetVertexDirection(arg0_) cvGetVertexDirectionImpl(RM_DATA_VAL arg0_)
vec3 cvGetVertexDirectionImpl(RM_DATA_ARG uint index)
{
    return vec3(RM_DATA(vertexDirections)[index].xyz);
}


#define normalizeVertexDirection(arg0_) normalizeVertexDirectionImpl(RM_DATA_VAL arg0_)
void normalizeVertexDirectionImpl(RM_DATA_ARG uint index)
{
    f16vec3 d                            = RM_DATA(vertexDirections)[index].xyz;
    RM_DATA(vertexDirections)[index].xyz = normalize(RM_DATA(vertexDirections)[index].xyz);
}

#define copyVertexDirection(arg0_, arg1_) copyVertexDirectionImpl(RM_DATA_VAL arg0_, arg1_)
void copyVertexDirectionImpl(RM_DATA_ARG uint dstIndex, uint srcIndex)
{
    RM_DATA(vertexDirections)[dstIndex] = RM_DATA(vertexDirections)[srcIndex];
}


#define isValid(arg0_) isValidImpl(RM_DATA_VAL arg0_)
bool isValidImpl(RM_DATA_ARG uvec3 triIndices)
{
    return !(triIndices.x == triIndices.y || triIndices.x == triIndices.z || triIndices.y == triIndices.z);
}

#define getOutputVertex(arg0_) getOutputVertexImpl(RM_DATA_VAL arg0_)
vec3 getOutputVertexImpl(RM_DATA_ARG uint index)
{
    return cvGetOutputPositionImpl(RM_DATA_VAL index);
}


#define getOriginalVertex(arg0_) getOriginalVertexImpl(RM_DATA_VAL arg0_)
vec3 getOriginalVertexImpl(RM_DATA_ARG uint index)
{
    return RM_DATA(scratchVertexOriginalPos)[index].xyz;
}

#define setOriginalVertex(arg0_, arg1_) setOriginalVertexImpl(RM_DATA_VAL arg0_, arg1_)
void setOriginalVertexImpl(RM_DATA_ARG uint index, vec3 pos)
{
    RM_DATA(scratchVertexOriginalPos)[index].xyz = pos;
}

#define getOriginalMaxEdgeLength(arg0_) getOriginalMaxEdgeLengthImpl(RM_DATA_VAL arg0_)
float getOriginalMaxEdgeLengthImpl(RM_DATA_ARG uint index)
{
    return RM_DATA(scratchVertexOriginalPos)[index].w;
}

#define setOriginalMaxEdgeLength(arg0_, arg1_) setOriginalMaxEdgeLengthImpl(RM_DATA_VAL arg0_, arg1_)
void setOriginalMaxEdgeLengthImpl(RM_DATA_ARG uint index, float maxEdgeLength)
{
    RM_DATA(scratchVertexOriginalPos)[index].w = maxEdgeLength;
}


#define getOutputPosNormal(arg0_, arg1_, arg2_) getOutputPosNormalImpl(RM_DATA_VAL arg0_, arg1_, arg2_)
void getOutputPosNormalImpl(RM_DATA_ARG uint index, REF(vec3) pos, REF(vec3) normal)
{
    pos    = cvGetOutputPositionImpl(RM_DATA_VAL index);
    normal = cvGetVertexDirectionImpl(RM_DATA_VAL index);
}

#define getOutputNormal(arg0_) getOutputNormalImpl(RM_DATA_VAL arg0_)
vec3 getOutputNormalImpl(RM_DATA_ARG uint index)
{
    return cvGetVertexDirectionImpl(RM_DATA_VAL index);
}


#define mergeOutputVertices(arg0_, arg1_, arg2_) mergeOutputVerticesImpl(RM_DATA_VAL arg0_, arg1_, arg2_)
void mergeOutputVerticesImpl(RM_DATA_ARG uint v0, uint v1, bool blockVolumePreservation)
{
    uint mergeSlot = atomicAdd(RM_DATA(currentState).mergeCount, 1);

    float w = 0.5f;

    if(blockVolumePreservation)
        w = -w;

    RM_DATA(vertexMerges)[3 * mergeSlot + 0] = v0;
    RM_DATA(vertexMerges)[3 * mergeSlot + 1] = v1;
    RM_DATA(vertexMerges)[3 * mergeSlot + 2] = floatBitsToUint(w);
}


uint wangHash(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

uint xorshift32(uint x64)
{
    x64 ^= x64 << 13;
    x64 ^= x64 >> 7;
    x64 ^= x64 << 17;
    return x64;
}


#define hashVertexIndex(arg0_) hashVertexIndexImpl(RM_DATA_VAL arg0_)
uint hashVertexIndexImpl(RM_DATA_ARG vec3 v)
{
    const uint offset = 0;
    uint       h;
    if(RM_CONSTANTS.deduplicationThreshold == 0.f)
    {
        h = wangHash(offset + floatBitsToUint(v.x));
        h = wangHash(offset + h + floatBitsToUint(v.y));
        h = wangHash(offset + h + floatBitsToUint(v.z));
    }
    else
    {
        v = floor(v / RM_CONSTANTS.deduplicationThreshold);
        h = wangHash(offset + uint(v.x));
        h = wangHash(offset + h + uint(v.y));
        h = wangHash(offset + h + uint(v.z));
    }
    return h % RM_CONSTANTS.hashMapSize;
}

#define hashVertexChecksum(arg0_) hashVertexChecksumImpl(RM_DATA_VAL arg0_)
uint hashVertexChecksumImpl(RM_DATA_ARG vec3 v)
{
    const uint offset = 0;
    uint       h;
    if(RM_CONSTANTS.deduplicationThreshold == 0.f)
    {
        h = xorshift32(offset + floatBitsToUint(v.x));
        h = xorshift32(offset + h + floatBitsToUint(v.y));
        h = xorshift32(offset + h + floatBitsToUint(v.z));
    }
    else
    {
        v = floor(v / RM_CONSTANTS.deduplicationThreshold);
        h = xorshift32(offset + uint(v.x));
        h = xorshift32(offset + h + uint(v.y));
        h = xorshift32(offset + h + uint(v.z));
    }
    return h % RM_CONSTANTS.hashMapSize;
}

#define rehashIndex(arg0_) rehashIndexImpl(RM_DATA_VAL arg0_)
uint rehashIndexImpl(RM_DATA_ARG uint h)
{
    uint newH = wangHash(h) % RM_CONSTANTS.hashMapSize;
    if(h == newH)
        return (h + 1) % RM_CONSTANTS.hashMapSize;
    return newH;
}

#define rehashEdgeIndex(arg0_, arg1_) rehashEdgeIndexImpl(RM_DATA_VAL arg0_, arg1_)
uint rehashEdgeIndexImpl(RM_DATA_ARG uint h, uvec2 edge)
{
    uint newH = wangHash(h + edge.x + edge.y) % RM_CONSTANTS.hashMapSize;
    if(h == newH)
        return (h + edge.x + edge.y) % RM_CONSTANTS.hashMapSize;
    return newH;
}

#define hashVertexAttributesIndex(arg0_) hashVertexAttributesIndexImpl(RM_DATA_VAL arg0_)
uint hashVertexAttributesIndexImpl(RM_DATA_ARG uint index)
{
    return RM_DATA(vertexHash)[index].x % RM_CONSTANTS.hashMapSize;
}

#define hashVertexAttributesChecksum(arg0_) hashVertexAttributesChecksumImpl(RM_DATA_VAL arg0_)
uint hashVertexAttributesChecksumImpl(RM_DATA_ARG uint index)
{
    return max(1u, RM_DATA(vertexHash)[index].y);
}

#define hashFullVertexIndex(arg0_) hashFullVertexIndexImpl(RM_DATA_VAL arg0_)
uint hashFullVertexIndexImpl(RM_DATA_ARG uint index)
{
    uint h = RM_DATA(vertexHash)[index].x;
    return h % RM_CONSTANTS.hashMapSize;
}

#define hashFullVertexChecksum(arg0_) hashFullVertexChecksumImpl(RM_DATA_VAL arg0_)
uint hashFullVertexChecksumImpl(RM_DATA_ARG uint index)
{
    uint h = RM_DATA(vertexHash)[index].y;
    return max(1u, h);
}

uint hashEdgeChecksum(uvec2 e)
{
    if(e.x < e.y)
        e = uvec2(e.x, e.y);
    else
        e = uvec2(e.y, e.x);

    uint h = xorshift32(e.x * 13543 + e.y * 7987 + 545);
    return max(1u, h);
}

#define hashEdgeIndex(arg0_) hashEdgeIndexImpl(RM_DATA_VAL arg0_)
uint hashEdgeIndexImpl(RM_DATA_ARG uvec2 e)
{
    if(e.x < e.y)
        e = uvec2(e.x, e.y);
    else
        e = uvec2(e.y, e.x);

    uint h = wangHash(e.x * 7525 + e.y * 213 + 78897);
    return h % RM_CONSTANTS.hashMapSize;
}


bool hasFlag(uint mask, uint flag)
{
    return (mask & flag) == flag;
}
