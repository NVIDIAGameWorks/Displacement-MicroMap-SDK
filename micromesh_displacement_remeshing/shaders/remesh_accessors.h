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

// Encode fp32 values into uint form suitable for atomicMin/Max operations
uint encodeMinMaxFp32(float val)
{
    uint bits = floatBitsToUint(val);
    bits ^= (int(bits) >> 31) | 0x80000000u;
    return bits;
}

// Decode fp32 values into uint form suitable for atomicMin/Max operations
float decodeMinMaxFp32(uint bits)
{
    bits ^= ~(int(bits) >> 31) | 0x80000000u;
    return uintBitsToFloat(bits);
}


#if defined(REMESHER_ACCESSORS_ALL) || defined(REMESHER_ACCESSORS_VERTICES)


#ifndef REMESHER_VERTEX_COUNT
#define REMESHER_VERTEX_COUNT RM_CONSTANTS.vertexCount
#endif

// Using explicit argument list arg0, arg1_ etc instead of variadic arguments since not supported by GLSL compiler

#define rvGetFlags(arg0_) rvGetFlagsImpl(RM_DATA_VAL arg0_)
uint rvGetFlagsImpl(RM_DATA_ARG uint index)
{
    return RM_DATA(scratchVertices)[index] & 0xFF;
}

#define rvSetFlags(arg0_, arg1_) rvSetFlagsImpl(RM_DATA_VAL arg0_, arg1_)
void rvSetFlagsImpl(RM_DATA_ARG uint index, uint flags)
{
    RM_DATA(scratchVertices)[index] = (RM_DATA(scratchVertices)[index] & ~0xFF) | flags & 0xFF;

    // Uncomment to output vertex flags for ad-hoc external visualization
    //RM_DATA(vertexDebug)[index] = (RM_DATA(vertexDebug)[index] & ~0xFF) | flags & 0xFF;
}


#define rvAtomicAddFlag(arg0_, arg1_) rvAtomicAddFlagImpl(RM_DATA_VAL arg0_, arg1_)
void rvAtomicAddFlagImpl(RM_DATA_ARG uint index, uint flags)
{
    atomicOr(RM_DATA(scratchVertices)[index], flags & 0xFF);
    
    // Uncomment to output vertex flags for ad-hoc external visualization
    //atomicOr(RM_DATA(vertexDebug)[index], flags & 0xFF);
    
}

#define rvAtomicRemoveFlag(arg0_, arg1_) rvAtomicRemoveFlagImpl(RM_DATA_VAL arg0_, arg1_)
void rvAtomicRemoveFlagImpl(RM_DATA_ARG uint index, uint flag)
{
    atomicAnd(RM_DATA(scratchVertices)[index], ~(flag & 0xFF));
    
    // Uncomment to output vertex flags for ad-hoc external visualization
    //atomicAnd(RM_DATA(vertexDebug)[index], ~(flag & 0xFF));
}

#define rvAtomicSetFlags(arg0_, arg1_) rvAtomicSetFlagsImpl(RM_DATA_VAL arg0_, arg1_)
void rvAtomicSetFlagsImpl(RM_DATA_ARG uint index, uint flags)
{
    RM_DATA(scratchVertices)[index] = (RM_DATA(scratchVertices)[index] & ~(0xFF)) | flags;
    // Uncomment to output vertex flags for ad-hoc external visualization
    //RM_DATA(vertexDebug)[index] = (RM_DATA(vertexDebug)[index] & ~(0xFF)) | flags;
}


#define rvGetLastTriangle(arg0_) rvGetLastTriangleImpl(RM_DATA_VAL arg0_)
uint rvGetLastTriangleImpl(RM_DATA_ARG uint index)
{
    return RM_DATA(scratchVertices)[1 * REMESHER_VERTEX_COUNT + index];
}

#define rvSetLastTriangle(arg0_, arg1_) rvSetLastTriangleImpl(RM_DATA_VAL arg0_, arg1_)
void rvSetLastTriangleImpl(RM_DATA_ARG uint index, uint lt)
{
    RM_DATA(scratchVertices)[1 * REMESHER_VERTEX_COUNT + index] = lt;
}

#define rvAtomicExchLastTriangle(arg0_, arg1_) rvAtomicExchLastTriangleImpl(RM_DATA_VAL arg0_, arg1_)
uint rvAtomicExchLastTriangleImpl(RM_DATA_ARG uint index, uint lt)
{
    return atomicExchange(RM_DATA(scratchVertices)[1 * REMESHER_VERTEX_COUNT + index], lt);
}

#define rvGetMergingWith(arg0_) rvGetMergingWithImpl(RM_DATA_VAL arg0_)
uint rvGetMergingWithImpl(RM_DATA_ARG uint index)
{
    return RM_DATA(scratchVertices)[2 * REMESHER_VERTEX_COUNT + index];
}

#define rvSetMergingWith(arg0_, arg1_) rvSetMergingWithImpl(RM_DATA_VAL arg0_, arg1_)
void rvSetMergingWithImpl(RM_DATA_ARG uint index, uint mergingWith)
{
    RM_DATA(scratchVertices)[2 * REMESHER_VERTEX_COUNT + index] = mergingWith;
}

#define rvGetDedupMerged(arg0_) rvGetDedupMergedImpl(RM_DATA_VAL arg0_)
uint rvGetDedupMergedImpl(RM_DATA_ARG uint index)
{
    return RM_DATA(scratchVertices)[3 * REMESHER_VERTEX_COUNT + index];
}

#define rvSetDedupMerged(arg0_, arg1_) rvSetDedupMergedImpl(RM_DATA_VAL arg0_, arg1_)
void rvSetDedupMergedImpl(RM_DATA_ARG uint index, uint dedupMerged)
{
    RM_DATA(scratchVertices)[3 * REMESHER_VERTEX_COUNT + index] = dedupMerged;
}

#define rvGetCurvature(arg0_) rvGetCurvatureImpl(RM_DATA_VAL arg0_)
float rvGetCurvatureImpl(RM_DATA_ARG uint index)
{
    return float(RM_DATA(vertexImportances)[index/2][index%2]);
}

#define rvSetCurvature(arg0_, arg1_) rvSetCurvatureImpl(RM_DATA_VAL arg0_, arg1_)
void rvSetCurvatureImpl(RM_DATA_ARG uint index, float c)
{
    RM_DATA(vertexImportances)[index/2][index%2] = float16_t(c);
}

#define rvGetAttribsHash(arg0_) rvGetAttribsHashImpl(RM_DATA_VAL arg0_)
uint rvGetAttribsHashImpl(RM_DATA_ARG uint index)
{
    return RM_DATA(scratchVertices)[index] >> 16;
}

#define rvSetAttribsHash(arg0_, arg1_) rvSetAttribsHashImpl(RM_DATA_VAL arg0_, arg1_)
void rvSetAttribsHashImpl(RM_DATA_ARG uint index, uint h)
{
    RM_DATA(scratchVertices)[index] = (RM_DATA(scratchVertices)[index] & 0xFFFF) | (h << 16);
}

#define rvGetHashIndex(arg0_) rvGetHashIndexImpl(RM_DATA_VAL arg0_)
uint rvGetHashIndexImpl(RM_DATA_ARG uint index)
{
    return RM_DATA(scratchVertices)[4 * REMESHER_VERTEX_COUNT + index];
}

#define rvSetHashIndex(arg0_, arg1_) rvSetHashIndexImpl(RM_DATA_VAL arg0_, arg1_)
void rvSetHashIndexImpl(RM_DATA_ARG uint index, uint h)
{
    RM_DATA(scratchVertices)[4 * REMESHER_VERTEX_COUNT + index] = h;
}

#ifdef REMESHER_INTERNAL
#define rvResetMinMaxDisplacement(arg0_) rvResetMinMaxDisplacementImpl(RM_DATA_VAL arg0_)
void rvResetMinMaxDisplacementImpl(RM_DATA_ARG uint index)
{
    RM_DATA(vertexDirectionBoundsU)[index].x = encodeMinMaxFp32(1e34f);
    RM_DATA(vertexDirectionBoundsU)[index].y = encodeMinMaxFp32(-1e34f);
}

#define rvFinalizeMinMaxDisplacement(arg0_) rvFinalizeMinMaxDisplacementImpl(RM_DATA_VAL arg0_)
void rvFinalizeMinMaxDisplacementImpl(RM_DATA_ARG uint index)
{
    uvec2 minMaxU = RM_DATA(vertexDirectionBoundsU)[index];
    vec2 minMaxF;
    minMaxF.x = decodeMinMaxFp32(minMaxU.x);
    minMaxF.y = decodeMinMaxFp32(minMaxU.y);

    if (RM_CONSTANTS.directionBoundsFactor != 1.f)
    {
        float midpoint = (minMaxF.x + minMaxF.y) / 2.f;
        float dist = (minMaxF.y - midpoint);
        dist *= RM_CONSTANTS.directionBoundsFactor;

        minMaxF.x = midpoint - dist;
        minMaxF.y = midpoint + dist;
    }
    RM_DATA(vertexDirectionBounds)[index].x = minMaxF.x;
    RM_DATA(vertexDirectionBounds)[index].y = minMaxF.y-minMaxF.x;
}


#define rvGetMinDisplacement(arg0_) rvGetMinDisplacementImpl(RM_DATA_VAL arg0_)
float rvGetMinDisplacementImpl(RM_DATA_ARG uint index)
{
    return decodeMinMaxFp32(RM_DATA(vertexDirectionBoundsU)[index].x);
}

#define rvGetMaxDisplacement(arg0_) rvGetMaxDisplacementImpl(RM_DATA_VAL arg0_)
float rvGetMaxDisplacementImpl(RM_DATA_ARG uint index)
{
    return decodeMinMaxFp32(RM_DATA(vertexDirectionBoundsU)[index].y);
}

#define rvAtomicMinDisplacement(arg0_, arg1_) rvAtomicMinDisplacementImpl(RM_DATA_VAL arg0_, arg1_)
void rvAtomicMinDisplacementImpl(RM_DATA_ARG uint index, float d)
{
    atomicMin(RM_DATA(vertexDirectionBoundsU)[index].x, encodeMinMaxFp32(d));
}

#define rvAtomicMaxDisplacement(arg0_, arg1_) rvAtomicMaxDisplacementImpl(RM_DATA_VAL arg0_, arg1_)
void rvAtomicMaxDisplacementImpl(RM_DATA_ARG uint index, float d)
{
    atomicMax(RM_DATA(vertexDirectionBoundsU)[index].y, encodeMinMaxFp32(d));
}


#endif

#endif


#if defined(REMESHER_ACCESSORS_ALL) || defined(REMESHER_ACCESSORS_EDGES)

#ifndef REMESHER_EDGE_COUNT
#define REMESHER_EDGE_COUNT (RM_CONSTANTS.edgeListSize)
#endif

#define reGetVertices(arg0_) reGetVerticesImpl(RM_DATA_VAL arg0_)
uvec2 reGetVerticesImpl(RM_DATA_ARG uint index)
{
    return uvec2(uint(RM_DATA(scratchEdges)[2 * index + 0]), uint(RM_DATA(scratchEdges)[2 * index + 1]));
}

#define reSetVertices(arg0_, arg1_) reSetVerticesImpl(RM_DATA_VAL arg0_, arg1_)
void reSetVerticesImpl(RM_DATA_ARG uint index, uvec2 v)
{
    RM_DATA(scratchEdges)[2 * index + 0] = v.x;
    RM_DATA(scratchEdges)[2 * index + 1] = v.y;
}

#define reAtomicMaxCost(arg0_, arg1_) reAtomicMaxCostImpl(RM_DATA_VAL arg0_, arg1_)
float reAtomicMaxCostImpl(RM_DATA_ARG uint index, float cost)
{
    return uintBitsToFloat(atomicMax(RM_DATA(scratchEdges)[2 * REMESHER_EDGE_COUNT + index], floatBitsToUint(cost)));
}

#define reGetCost(arg0_) reGetCostImpl(RM_DATA_VAL arg0_)
float reGetCostImpl(RM_DATA_ARG uint index)
{
    return uintBitsToFloat(RM_DATA(scratchEdges)[2 * REMESHER_EDGE_COUNT + index]);
}

#define reSetCost(arg0_, arg1_) reSetCostImpl(RM_DATA_VAL arg0_, arg1_)
void reSetCostImpl(RM_DATA_ARG uint index, float c)
{
    RM_DATA(scratchEdges)[2 * REMESHER_EDGE_COUNT + index] = floatBitsToUint(c);
}


#endif


#if defined(REMESHER_ACCESSORS_ALL) || defined(REMESHER_ACCESSORS_HASH_ENTRY)


#ifndef REMESHER_HASH_ENTRY_COUNT
#define REMESHER_HASH_ENTRY_COUNT RM_CONSTANTS.hashMapSize
#endif

#define rhGetChecksum(arg0_) rhGetChecksumImpl(RM_DATA_VAL arg0_)
uint rhGetChecksumImpl(RM_DATA_ARG uint index)
{
    return RM_DATA(scratchHashMap)[index];
}

#define rhSetChecksum(arg0_, arg1_) rhSetChecksumImpl(RM_DATA_VAL arg0_, arg1_)
void rhSetChecksumImpl(RM_DATA_ARG uint index, uint c)
{
    RM_DATA(scratchHashMap)[index] = c;
}

#define rhAtomicCompSwapChecksum(arg0_, arg1_, arg2_) rhAtomicCompSwapChecksumImpl(RM_DATA_VAL arg0_, arg1_, arg2_)
uint rhAtomicCompSwapChecksumImpl(RM_DATA_ARG uint index, uint compare, uint data)
{
    return atomicCompSwap(RM_DATA(scratchHashMap)[index], compare, data);
}

#define rhGetStoredIndex(arg0_) rhGetStoredIndexImpl(RM_DATA_VAL arg0_)
uint rhGetStoredIndexImpl(RM_DATA_ARG uint index)
{
    return RM_DATA(scratchHashMap)[2*REMESHER_HASH_ENTRY_COUNT + index];
}

#define rhSetStoredIndex(arg0_, arg1_) rhSetStoredIndexImpl(RM_DATA_VAL arg0_, arg1_)
void rhSetStoredIndexImpl(RM_DATA_ARG uint index, uint c)
{
    RM_DATA(scratchHashMap)[2*REMESHER_HASH_ENTRY_COUNT + index] = c;
}

#define rhGetRefCounter(arg0_) rhGetRefCounterImpl(RM_DATA_VAL arg0_)
uint rhGetRefCounterImpl(RM_DATA_ARG uint index)
{
    return RM_DATA(scratchHashMap)[REMESHER_HASH_ENTRY_COUNT + index] & 0xFFFF;
}

#define rhSetRefCounter(arg0_, arg1_) rhSetRefCounterImpl(RM_DATA_VAL arg0_, arg1_)
void rhSetRefCounterImpl(RM_DATA_ARG uint index, uint c)
{
    RM_DATA(scratchHashMap)
    [2 * REMESHER_HASH_ENTRY_COUNT + index] =
        (RM_DATA(scratchHashMap)[REMESHER_HASH_ENTRY_COUNT + index] & 0xFFFF0000) | c;
}

#define rhAtomicIncRefCounter(arg0_) rhAtomicIncRefCounterImpl(RM_DATA_VAL arg0_)
uint rhAtomicIncRefCounterImpl(RM_DATA_ARG uint index)
{
    return atomicAdd(RM_DATA(scratchHashMap)[REMESHER_HASH_ENTRY_COUNT + index], 1u);
}


#endif


#if defined(REMESHER_ACCESSORS_ALL) || defined(REMESHER_ACCESSORS_TRIANGLE)
//
#ifndef REMESHER_TRIANGLE_COUNT
#define REMESHER_TRIANGLE_COUNT (RM_CONSTANTS.indexCount / 3)
#endif
#define TRIANGLE_WORD_COUNT 9

#define rtSetPreviousTriangle(arg0_, arg1_, arg2_) rtSetPreviousTriangleImpl(RM_DATA_VAL arg0_, arg1_, arg2_)
void rtSetPreviousTriangleImpl(RM_DATA_ARG uint index, uint localTri, uint previousTriangle)
{
    RM_DATA(scratchTriangles)[TRIANGLE_WORD_COUNT * index + localTri] = previousTriangle;
}

#define rtGetPreviousTriangle(arg0_, arg1_) rtGetPreviousTriangleImpl(RM_DATA_VAL arg0_, arg1_)
uint rtGetPreviousTriangleImpl(RM_DATA_ARG uint index, uint localTri)
{
    return RM_DATA(scratchTriangles)[TRIANGLE_WORD_COUNT * index + localTri];
}

#define rtSetEdgeIndex(arg0_, arg1_, arg2_) rtSetEdgeIndexImpl(RM_DATA_VAL arg0_, arg1_, arg2_)
void rtSetEdgeIndexImpl(RM_DATA_ARG uint index, uint localTri, uint edgeIndex)
{
    RM_DATA(scratchTriangles)[TRIANGLE_WORD_COUNT * index + 3 + localTri] = edgeIndex;
}

#define rtGetEdgeIndex(arg0_, arg1_) rtGetEdgeIndexImpl(RM_DATA_VAL arg0_, arg1_)
uint rtGetEdgeIndexImpl(RM_DATA_ARG uint index, uint localTri)
{
    return RM_DATA(scratchTriangles)[TRIANGLE_WORD_COUNT * index + 3 + localTri];
}

#define rtGetEdgeIndices(arg0_) rtGetEdgeIndicesImpl(RM_DATA_VAL arg0_)
uvec3 rtGetEdgeIndicesImpl(RM_DATA_ARG uint index)
{
    uvec3 res;
    for(uint i = 0; i < 3; i++)
    {
        res[i] = RM_DATA(scratchTriangles)[TRIANGLE_WORD_COUNT * index + 3 + i];
    }
    return res;
}

#define rtSetEdgeIndices(arg0_, arg1_) rtSetEdgeIndicesImpl(RM_DATA_VAL arg0_, arg1_)
void rtSetEdgeIndicesImpl(RM_DATA_ARG uint index, uvec3 indices)
{
    for(uint i = 0; i < 3; i++)
    {
        RM_DATA(scratchTriangles)[TRIANGLE_WORD_COUNT * index + 3 + i] = indices[i];
    }
}

#define rtSetIsValid(arg0_, arg1_) rtSetIsValidImpl(RM_DATA_VAL arg0_, arg1_)
void rtSetIsValidImpl(RM_DATA_ARG uint index, bool isValid)
{
    uint v = RM_DATA(scratchTriangles)[TRIANGLE_WORD_COUNT * index + 6];
    if(isValid)
        v = v | (1 << 31);
    else
        v = v & ~(1 << 31);
    RM_DATA(scratchTriangles)[TRIANGLE_WORD_COUNT * index + 6] = v;
}

#define rtGetIsValid(arg0_) rtGetIsValidImpl(RM_DATA_VAL arg0_)
bool rtGetIsValidImpl(RM_DATA_ARG uint index)
{
    uint v = RM_DATA(scratchTriangles)[TRIANGLE_WORD_COUNT * index + 6];
    return (v >> 31) == 0x1;
}

#define rtResetAggregatedCounter(arg0_) rtResetAggregatedCounterImpl(RM_DATA_VAL arg0_)
void rtResetAggregatedCounterImpl(RM_DATA_ARG uint index)
{
    RM_DATA(scratchTriangles)
    [TRIANGLE_WORD_COUNT * index + 6] = RM_DATA(scratchTriangles)[TRIANGLE_WORD_COUNT * index + 6] & (1 << 31);
}

#define rtSetAggregatedCounter(arg0_, arg1_) rtSetAggregatedCounterImpl(RM_DATA_VAL arg0_, arg1_)
void rtSetAggregatedCounterImpl(RM_DATA_ARG uint index, uint value)
{
    RM_DATA(scratchTriangles)
    [TRIANGLE_WORD_COUNT * index + 6] = RM_DATA(scratchTriangles)[TRIANGLE_WORD_COUNT * index + 6] | (value & ~(1 << 31));
}

#define rtAtomicAddAggregatedCounter(arg0_, arg1_) rtAtomicAddAggregatedCounterImpl(RM_DATA_VAL arg0_, arg1_)
uint rtAtomicAddAggregatedCounterImpl(RM_DATA_ARG uint index, uint value)
{
    return atomicAdd(RM_DATA(scratchTriangles)[TRIANGLE_WORD_COUNT * index + 6], value) & ~(1 << 31);
}

#define rtGetAggregatedCounter(arg0_) rtGetAggregatedCounterImpl(RM_DATA_VAL arg0_)
uint rtGetAggregatedCounterImpl(RM_DATA_ARG uint index)
{
    return RM_DATA(scratchTriangles)[TRIANGLE_WORD_COUNT * index + 6] & ~(1 << 31);
}

#define rtCopy(arg0_, arg1_) rtCopyImpl(RM_DATA_VAL arg0_, arg1_)
void rtCopyImpl(RM_DATA_ARG uint dst, uint src)
{
    for(uint i = 0; i < TRIANGLE_WORD_COUNT; i++)
    {
        RM_DATA(scratchTriangles)
        [TRIANGLE_WORD_COUNT * dst + i] = RM_DATA(scratchTriangles)[TRIANGLE_WORD_COUNT * src + i];
    }
}

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F  // max value
#endif
#define rtAtomicMinMaxDisplacement(arg0_, arg1_) rtAtomicMinMaxDisplacementImpl(RM_DATA_VAL arg0_, arg1_)
void rtAtomicMinMaxDisplacementImpl(RM_DATA_ARG uint index, vec2 disp)
{
    atomicMin(RM_DATA(scratchTriangles)[TRIANGLE_WORD_COUNT * index + 7 + 0], encodeMinMaxFp32(disp.x));
    atomicMax(RM_DATA(scratchTriangles)[TRIANGLE_WORD_COUNT * index + 7 + 1], encodeMinMaxFp32(disp.y));
}

#define rtGetMinDisplacement(arg0_) rtGetMinDisplacementImpl(RM_DATA_VAL arg0_)
float rtGetMinDisplacementImpl(RM_DATA_ARG uint index)
{
    float d = decodeMinMaxFp32(RM_DATA(scratchTriangles)[TRIANGLE_WORD_COUNT * index + 7 + 0]);
    return d;
}

#define rtGetMaxDisplacement(arg0_) rtGetMaxDisplacementImpl(RM_DATA_VAL arg0_)
float rtGetMaxDisplacementImpl(RM_DATA_ARG uint index)
{
    float d = decodeMinMaxFp32(RM_DATA(scratchTriangles)[TRIANGLE_WORD_COUNT * index + 7 + 1]);
    return d;
}

#define rtResetDisplacementBounds(arg0_) rtResetDisplacementBoundsImpl(RM_DATA_VAL arg0_)
void rtResetDisplacementBoundsImpl(RM_DATA_ARG uint index)
{
    RM_DATA(scratchTriangles)[TRIANGLE_WORD_COUNT * index + 7 + 0] = encodeMinMaxFp32(FLT_MAX);
    RM_DATA(scratchTriangles)[TRIANGLE_WORD_COUNT * index + 7 + 1] = encodeMinMaxFp32(-FLT_MAX);
}



#endif


#if defined(REMESHER_ACCESSORS_ALL) || defined(REMESHER_ACCESSORS_MICROMESH_INFO)
//
#ifndef REMESHER_TRIANGLE_COUNT
#define REMESHER_TRIANGLE_COUNT (RM_CONSTANTS.indexCount / 3)
#endif
#ifndef REMESHER_ITERATION_INDEX
#define REMESHER_ITERATION_INDEX (RM_CONSTANTS.iterationIndex)
#endif

uint getUint8(uint v, uint index)
{
    return (v >> 8 * index) & 0xFF;
}


#define setUint8(_target, _index, _v) \
{\
uint bitMask = (0xFF << (8 * _index));\
uint inserted = (_v & 0xFF) << (8 * _index);\
atomicAnd(_target, ~bitMask); \
atomicOr(_target, inserted); \
}


// Target even iteration number is the final storage  
#define RMM_TARGET_SUBD_ODD_ITERATION 1
#define RMM_TARGET_SUBD_EVEN_ITERATION 0

#define RMM_CURRENT_SUBD_ODD_ITERATION 0
#define RMM_CURRENT_SUBD_EVEN_ITERATION 1

#define RMM_DECIMATION_FLAGS 2
#define RMM_OPEN_EDGE_FLAGS 3

#define rmmGetTargetSubdivisionLevel(arg0_) rmmGetTargetSubdivisionLevelImpl(RM_DATA_VAL arg0_)
uint rmmGetTargetSubdivisionLevelImpl(RM_DATA_ARG uint index)
{
    if(REMESHER_ITERATION_INDEX % 2 == 1)
        return getUint8(RM_DATA(triangleSubdivisionInfo)[index], RMM_TARGET_SUBD_ODD_ITERATION);
    else
        return getUint8(RM_DATA(triangleSubdivisionInfo)[index], RMM_TARGET_SUBD_EVEN_ITERATION);
}

#define rmmGetCurrentSubdivisionLevel(arg0_) rmmGetCurrentSubdivisionLevelImpl(RM_DATA_VAL arg0_)
uint rmmGetCurrentSubdivisionLevelImpl(RM_DATA_ARG uint index)
{
    if(REMESHER_ITERATION_INDEX % 2 == 0)
        return getUint8(RM_DATA(triangleSubdivisionInfo)[index], RMM_CURRENT_SUBD_EVEN_ITERATION);
    else
        return getUint8(RM_DATA(triangleSubdivisionInfo)[index], RMM_CURRENT_SUBD_ODD_ITERATION);
}

#define rmmSetTargetSubdivisionLevel(arg0_, arg1_) rmmSetTargetSubdivisionLevelImpl(RM_DATA_VAL arg0_, arg1_)
void rmmSetTargetSubdivisionLevelImpl(RM_DATA_ARG uint index, uint level)
{
    if (REMESHER_ITERATION_INDEX % 2 == 1)
    {
        setUint8(RM_DATA(triangleSubdivisionInfo)[index], RMM_TARGET_SUBD_ODD_ITERATION, level);
    }
    else
    {
        setUint8(RM_DATA(triangleSubdivisionInfo)[index], RMM_TARGET_SUBD_EVEN_ITERATION, level);
    }
}

#define rmmSetCurrentSubdivisionLevel(arg0_, arg1_) rmmSetCurrentSubdivisionLevelImpl(RM_DATA_VAL arg0_, arg1_)
void rmmSetCurrentSubdivisionLevelImpl(RM_DATA_ARG uint index, uint level)
{
    if(REMESHER_ITERATION_INDEX % 2 == 0)
        {setUint8(RM_DATA(triangleSubdivisionInfo)[index], RMM_CURRENT_SUBD_EVEN_ITERATION, level);}
    else
        {setUint8(RM_DATA(triangleSubdivisionInfo)[index], RMM_CURRENT_SUBD_ODD_ITERATION, level);}
}

#define rmmGetDecimationFlags(arg0_) rmmGetDecimationFlagsImpl(RM_DATA_VAL arg0_)
uint rmmGetDecimationFlagsImpl(RM_DATA_ARG uint index)
{
    return getUint8(RM_DATA(triangleSubdivisionInfo)[index], RMM_DECIMATION_FLAGS);
}
#define rmmSetDecimationFlags(arg0_, arg1_) rmmSetDecimationFlagsImpl(RM_DATA_VAL arg0_, arg1_)
void rmmSetDecimationFlagsImpl(RM_DATA_ARG uint index, uint flags)
{
    setUint8(RM_DATA(triangleSubdivisionInfo)[index], RMM_DECIMATION_FLAGS, flags);
}

#define rmmCleanup(arg0_) rmmCleanupImpl(RM_DATA_VAL arg0_)
void rmmCleanupImpl(RM_DATA_ARG uint index)
{
    setUint8(RM_DATA(triangleSubdivisionInfo)[index], RMM_OPEN_EDGE_FLAGS, 0);
    setUint8(RM_DATA(triangleSubdivisionInfo)[index], RMM_TARGET_SUBD_ODD_ITERATION, 0);
}

#define rmmCopy(arg0_, arg1_) rmmCopyImpl(RM_DATA_VAL arg0_, arg1_)
void rmmCopyImpl(RM_DATA_ARG uint dst, uint src)
{
    RM_DATA(triangleSubdivisionInfo)[dst] = RM_DATA(triangleSubdivisionInfo)[src];
}

#endif


#if defined(REMESHER_ACCESSORS_ALL) || defined(REMESHER_ACCESSORS_MICROMESH_DISP_DIR)
//
#ifndef REMESHER_VERTEX_COUNT
#define REMESHER_VERTEX_COUNT (RM_CONSTANTS.vertexCount)
#endif
#define rddSetDirection(arg0_, arg1_) rddSetDirectionImpl(RM_DATA_VAL arg0_, arg1_)
void rddSetDirectionImpl(RM_DATA_ARG uint index, vec3 d)
{
    RM_DATA(vertexDirections)[index] = f16vec4(d, 1.f);
}
#define rddGetDirection(arg0_) rddGetDirectionImpl(RM_DATA_VAL arg0_)
vec3 rddGetDirectionImpl(RM_DATA_ARG uint index)
{
    return vec3(RM_DATA(vertexDirections)[index].xyz);
}






#define siGetDedupIndex(arg0_) siGetDedupIndexImpl(RM_DATA_VAL arg0_)
uint siGetDedupIndexImpl(RM_DATA_ARG uint index)
{
   return RM_DATA(scratchIndices)[index];
}

#define siSetDedupIndex(arg0_, arg1_) siSetDedupIndexImpl(RM_DATA_VAL arg0_, arg1_)
void siSetDedupIndexImpl(RM_DATA_ARG uint index, uint value)
{
   RM_DATA(scratchIndices)[index] = value;
}

#define siGetHashIndex(arg0_) siGetHashIndexImpl(RM_DATA_VAL arg0_)
uint siGetHashIndexImpl(RM_DATA_ARG uint index)
{
   return RM_DATA(scratchIndices)[REMESHER_TRIANGLE_COUNT*3 + index];
}

#define siSetHashIndex(arg0_, arg1_) siSetHashIndexImpl(RM_DATA_VAL arg0_, arg1_)
void siSetHashIndexImpl(RM_DATA_ARG uint index, uint value)
{
   RM_DATA(scratchIndices)[REMESHER_TRIANGLE_COUNT*3 + index] = value;
}


#define siGetDedupTriangle(arg0_) siGetDedupTriangleImpl(RM_DATA_VAL arg0_)
uvec3 siGetDedupTriangleImpl(RM_DATA_ARG uint index)
{
    uint mod4 = index%4;
    uint div4 = index/4;
    uvec3 res;

    switch(mod4)
    {
    case 0:
        return RM_DATA(scratchIndices128)[div4*3].xyz;
    case 1:
        res.x = siGetDedupIndex(RM_DATA_ARG 3*index+0);
        res.yz = RM_DATA(scratchIndices128)[div4*3+1].xy;
        return res;
    case 2:
        res.xy = RM_DATA(scratchIndices128)[div4*3+1].zw;
        res.z = siGetDedupIndex(RM_DATA_ARG 3*index+2);
        return res;
    case 3:
        return RM_DATA(scratchIndices128)[div4*3+2].yzw;    
    }
}


#endif
