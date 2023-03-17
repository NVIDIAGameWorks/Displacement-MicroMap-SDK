//
// Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#pragma once

#include <cassert>
#include <cmath>

#include "micromesh_types.h"
#include "micromesh_api.h"

namespace micromesh
{
// returns a stringified name for the enum. For instance,
// getErrorName(Result::eInvalidFrequency) returns "eInvalidFrequency".
MICROMESH_API const char* MICROMESH_CALL micromeshResultGetName(Result result);

MICROMESH_API Result MICROMESH_CALL       micromeshFormatGetInfo(Format format, FormatInfo* formatInfo);
MICROMESH_API MicromapType MICROMESH_CALL micromeshFormatGetMicromapType(Format format);
MICROMESH_API Format MICROMESH_CALL       micromeshFormatGetMinMaxFormat(Format format);

// defaults to ePerVertex for uncompressed/packed formats
MICROMESH_API Frequency MICROMESH_CALL micromeshFormatGetCompressedFrequency(Format format);
// defaults to eUnknown for uncompresed/packed formats
MICROMESH_API StandardLayoutType MICROMESH_CALL micromeshFormatGetCompressedStandardLayout(Format format);

MICROMESH_API Result MICROMESH_CALL micromeshBlockFormatDispC1GetInfo(BlockFormatDispC1 format, FormatInfo* formatInfo);
MICROMESH_API Result MICROMESH_CALL micromeshBlockFormatOpaC1GetInfo(BlockFormatOpaC1 format, FormatInfo* formatInfo);

MICROMESH_API const char* MICROMESH_CALL micromeshGetFormatString(Format format);
MICROMESH_API const char* MICROMESH_CALL micromeshBlockFormatDispC1GetString(BlockFormatDispC1 format);
MICROMESH_API const char* MICROMESH_CALL micromeshBlockFormatOpaC1GetString(BlockFormatOpaC1 format);

// setup stanard layouts
MICROMESH_API Result MICROMESH_CALL micromeshLayoutInitStandard(MicromapLayout* layout, StandardLayoutType standard);

// get type if applicable, only works if layout was created through micromeshLayoutInitStandard
MICROMESH_API StandardLayoutType MICROMESH_CALL micromeshLayoutGetStandardType(const MicromapLayout* layout);

// generates the micro-triangles of the UV mesh for a provided layout
// triangles and vertices are in layout order.
// arrays must be properly sized (see subdivLevelGetCount)
// if `edgeFlag` is non zero, then the arrays will contain
// unused vertices or degenerated triangles
MICROMESH_API Result MICROMESH_CALL micromeshLayoutBuildUVMesh(const MicromapLayout* layout,
                                                               ArrayInfo_uint16_2*   uvVertices,
                                                               ArrayInfo_uint32_3*   triangleIndices,
                                                               uint32_t              subdivLevel,
                                                               uint32_t              edgeFlag);

// computes values.count based on triangleSubdivLevels
// can also fill in computeTriangleValueIndexOffsets
MICROMESH_API Result MICROMESH_CALL micromeshMicromapSetupValues(Micromap* micromap, bool computeTriangleValueIndexOffsets);

// computes values.count based on triangleSubdivLevels
// can also fill in computeTriangleValueIndexOffsets
MICROMESH_API Result MICROMESH_CALL micromeshMicromapPackedSetupValues(MicromapPacked* micromap, bool computeTriangleValueByteOffsets);

// computes values.count based on triangleSubdivLevels and
// triangleBlockFormats
// can also fill in computeTriangleValueByteOffsets
MICROMESH_API Result MICROMESH_CALL micromeshMicromapCompressedSetupValues(MicromapCompressed* micromap,
                                                                           bool computeTriangleValueByteOffsets,
                                                                           const MessageCallbackInfo* messageCallbackInfo);


// snaps uv based on edgeDecimationFlag
MICROMESH_API BaryUV_uint16 MICROMESH_CALL micromeshUVGetEdgeDecimated(BaryUV_uint16 coord, uint32_t subdivLevel, uint32_t edgeDecimationFlag);

// returns perturbated UV based on swizzle
MICROMESH_API BaryUV_uint16 MICROMESH_CALL micromeshUVGetSwizzled(BaryUV_uint16 uv, uint32_t subdivLevel, TriangleSwizzle swizzle);

// returns values from a single triangle
MICROMESH_API ArrayInfo MICROMESH_CALL micromeshMicromapGetTriangleArray(const Micromap* micromap, uint32_t triangleIndex);

//////////////////////////////////////////////////////////////////////////

// arbitrary sub triangle within base triangle
struct SubTriangleExtraction
{
    // which base triangle to extract from
    uint32_t triangleIndex;

    // triangle must be well behaved and match subdivision
    // one horizontal and one vertical edge with proper length
    BaryUV_uint16 subVertices[3];
    // subdivision level of sub triangle
    uint32_t subSubdiv;
    // typically matches micromap.values.byteStride
    // ignored if we are using "packed"
    uint32_t valueCopyBytes;
};

// no error checks assumed to be called at high-frequncy within function that error checks
// values must _not_ be packed nor compressed
MICROMESH_API void MICROMESH_CALL micromeshSubTriangleExtraction(const Micromap*              micromap,
                                                                 const SubTriangleExtraction* input,
                                                                 ArrayInfo*                   ouput);

MICROMESH_API void MICROMESH_CALL micromeshSubTriangleExtractionPacked(const MicromapPacked*        micromap,
                                                                       const SubTriangleExtraction* extract,
                                                                       ArrayInfo*                   output);

//////////////////////////////////////////////////////////////////////////

// a block triangle represents one compressed block within a base triangle
// its layout depends on the block compression format
struct BlockTriangle
{
    // three UV coordinates of this block triangle
    // relative to original base triangle (bary::Triangle)
    BaryUV_uint16 vertices[3];
    // flipped winding 0/1
    uint8_t flipped;
    // u and v sign relative to first vertex
    // bit 0: set if verticesUV[1].u > verticesUV[0].u
    // bit 1: set if verticesUV[2].v > verticesUV[0].v
    uint8_t signBits;
    // 3 x 2 bits that specify which local edge (0,1,2) maps to what
    // base edge (0,1,2) The value 3 means the local edge does not lie
    // on any base edge
    uint8_t baseEdgeIndices;
    uint8_t _reserved;
    // where this block's data starts relative to
    // original base triangle valuesOffset (which is in bytes for compressed data)
    uint32_t blockByteOffset;
};

MICROMESH_API uint32_t MICROMESH_CALL micromeshBlockFormatDispC1GetBlockCount(BlockFormatDispC1 blockFormat, uint32_t baseSubdivLevel);

MICROMESH_API void MICROMESH_CALL micromeshBlockTriangleSplitDispC1(const BlockTriangle* inTri, BlockTriangle* outTris, uint32_t outStride);

MICROMESH_API Result MICROMESH_CALL micromeshBlockFormatDispC1GetBlockTriangles(BlockFormatDispC1 blockFormat,
                                                                                uint32_t          baseSubdivLevel,
                                                                                uint32_t          blockTrisCount,
                                                                                BlockTriangle*    blockTris);

// simple packing of uncompressed unorm11 values into eR11_unorm_lvl3_pack512
// use above functions to prepare the block triangles
// `blockFormat` must be `BlockFormatDispC1::eR11_unorm_lvl3_pack512`
// `baseData::format` must be `Format::eR11_unorm_pack16` or `Format::eR11_unorm_packed_align32`
// `blocksData` is base address where all blocks start
MICROMESH_API Result MICROMESH_CALL micromeshBlockFormatDispC1FillBlocks(BlockFormatDispC1     blockFormat,
                                                                         uint32_t              blockTrisCount,
                                                                         const BlockTriangle*  blockTris,
                                                                         const MicromapLayout* baseLayout,
                                                                         uint32_t              baseSubdivLevel,
                                                                         const ArrayInfo*      baseData,
                                                                         size_t                blocksDataSize,
                                                                         void*                 blocksData);

//////////////////////////////////////////////////////////////////////////

MICROMESH_API uint32_t MICROMESH_CALL micromeshBirdUVToLinearTriangle(uint32_t u, uint32_t v, uint32_t isUpperTriangle, uint32_t subdivLevel);

MICROMESH_API uint32_t MICROMESH_CALL micromeshBirdUVToLinearVertex(uint32_t u, uint32_t v, uint32_t subdivLevel);


//////////////////////////////////////////////////////////////////////////
// special packed format computations

inline uint32_t packedCountR11UnormPackedAlign32(uint32_t numValues)
{
    return (numValues + 31) / 32;
}
// `bitOffset` starts from data address provided, no range cap
// `bitWidth` must be <= 32
inline void packedBitWrite(void* data, uint32_t bitOffset, uint32_t bitWidth, uint32_t inValue)
{
    uint32_t* outU32 = reinterpret_cast<uint32_t*>(data);

    const uint32_t bitMask = uint32_t((1ull << bitWidth) - 1);
    const uint32_t bitIdx  = bitOffset;
    const uint32_t idx     = bitIdx / 32;
    const uint32_t shift   = bitIdx % 32;

    union
    {
        uint32_t u32s[2];
        uint64_t u64;
    } value;

    union
    {
        uint32_t u32s[2];
        uint64_t u64;
    } mask;

    mask.u64  = uint64_t(bitMask) << shift;
    value.u64 = uint64_t(inValue & bitMask) << shift;

    if(shift + bitWidth <= 32)
    {
        outU32[idx] &= ~mask.u32s[0];
        outU32[idx] |= value.u32s[0];
    }
    else
    {
        outU32[idx] &= ~mask.u32s[0];
        outU32[idx] |= value.u32s[0];
        outU32[idx + 1] &= ~mask.u32s[1];
        outU32[idx + 1] |= value.u32s[1];
    }
}

// `bitOffset` starts from data address provided, no range cap
// `bitWidth` must be <= 32
inline uint32_t packedBitRead(const void* data, uint32_t bitOffset, uint32_t bitWidth)
{
    const uint32_t* outU32 = reinterpret_cast<const uint32_t*>(data);

    const uint32_t bitMask = uint32_t((1ull << bitWidth) - 1);
    const uint32_t bitIdx  = bitOffset;
    const uint32_t idx     = bitIdx / 32;
    const uint32_t shift   = bitIdx % 32;

    if(shift + bitWidth <= 32)
    {
        return (outU32[idx] >> shift) & bitMask;
    }
    else
    {
        union
        {
            uint32_t u32s[2];
            uint64_t u64;
        };
        u32s[0] = outU32[idx];
        u32s[1] = outU32[idx + 1];
        return uint32_t(u64 >> shift) & bitMask;
    }
}

inline void packedWriteR11UnormPackedAlign32(void* data, uint32_t valueIdx, uint32_t inValue)
{
    packedBitWrite(data, valueIdx * 11, 11, inValue);
}

inline uint16_t packedReadR11UnormPackedAlign32(const void* data, uint32_t valueIdx)
{
    return uint16_t(packedBitRead(data, valueIdx * 11, 11));
}

//////////////////////////////////////////////////////////////////////////

inline uint32_t blockFormatDispC1GetSubdivLevel(BlockFormatDispC1 format)
{
    switch(format)
    {
    case BlockFormatDispC1::eR11_unorm_lvl3_pack512:
        return 3;
    case BlockFormatDispC1::eR11_unorm_lvl4_pack1024:
        return 4;
    case BlockFormatDispC1::eR11_unorm_lvl5_pack1024:
        return 5;
    default:
        return 0;
    }
}

inline uint32_t blockFormatDispC1GetByteSize(BlockFormatDispC1 format)
{
    switch(format)
    {
    case BlockFormatDispC1::eR11_unorm_lvl3_pack512:
        return 64;
    case BlockFormatDispC1::eR11_unorm_lvl4_pack1024:
        return 128;
    case BlockFormatDispC1::eR11_unorm_lvl5_pack1024:
        return 128;
    default:
        return 0;
    }
}

//////////////////////////////////////////////////////////////////////////

inline bool micromapLayoutIsValid(const MicromapLayout& a)
{
    return a.pfnGetMicroTriangleIndex && a.pfnGetMicroVertexIndex;
}

inline bool micromapLayoutIsEqual(const MicromapLayout& a, const MicromapLayout& b)
{
    return a.pfnGetMicroTriangleIndex == b.pfnGetMicroTriangleIndex
           && a.pfnGetMicroVertexIndex == b.pfnGetMicroVertexIndex && a.userData == b.userData;
}

//////////////////////////////////////////////////////////////////////////

// aid setup of ArrayInfo
// these functions would be typically used by users of the api to setup
// the ArrayInfo that needs to be passed to the api.

inline void arraySetData(ArrayInfo& info, const void* data, uint64_t count, uint32_t byteStride = 0)
{
    info.data       = const_cast<void*>(data);
    info.count      = count;
    info.byteStride = byteStride ? byteStride : info.byteStride;
}

inline void arraySetData(ArrayInfo& info, void* data, uint64_t count, uint32_t byteStride = 0)
{
    info.data       = data;
    info.count      = count;
    info.byteStride = byteStride ? byteStride : info.byteStride;
}

template <typename T>
inline void arraySetDataVec(ArrayInfo& info, T& vector)
{
    assert(sizeof(typename T::value_type) == info.byteStride);
    info.data  = reinterpret_cast<void*>(vector.data());
    info.count = vector.size();
}

template <typename T>
inline void arraySetDataVec(ArrayInfo& info, const T& vector)
{
    assert(sizeof(typename T::value_type) == info.byteStride);
    info.data  = const_cast<void*>(reinterpret_cast<const void*>(vector.data()));
    info.count = vector.size();
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

// These array utils are typically used within the micromesh components.
// As user of the api, there is no need to use ArrayInfo other than passing
// your data-structures.

inline bool arrayIsValid(const ArrayInfo& info)
{
    return (info.format != Format::eUndefined && info.byteStride != 0 && info.data);
}

inline bool arrayIsEmpty(const ArrayInfo& info)
{
    return !(info.count && info.data);
}

inline bool arrayIsEqual(const ArrayInfo& a, const ArrayInfo& b)
{
    return a.data == b.data && a.count == b.count && a.format == b.format && a.byteStride == b.byteStride;
}

template <typename T>
inline void arraySetV(ArrayInfo& info, uint64_t idx, T value)
{
    assert(idx < info.count);
    *reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(info.data) + (idx * info.byteStride)) = value;
}

template <class T>
inline T arrayGetV(const ArrayInfo& info, uint64_t idx)
{
    assert(idx < info.count);
    return *reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(info.data) + (idx * info.byteStride));
}

template <class T>
inline T* arrayGet(ArrayInfo& info, uint64_t idx)
{
    assert(idx < info.count);
    return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(info.data) + (idx * info.byteStride));
}

template <class T>
inline const T* arrayGet(const ArrayInfo& info, uint64_t idx)
{
    assert(idx < info.count);
    return reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(info.data) + (idx * info.byteStride));
}

template <typename T>
inline const T* arrayPointerGet(const ArrayInfo& info, const T* ptr, uint64_t idx)
{
    // pointer must come from array
    assert(uint64_t(ptr) >= uint64_t(info.data) && (((uint64_t(ptr) - uint64_t(info.data)) / info.byteStride) + idx) <= info.count);
    return reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(ptr) + info.byteStride * idx);
}

template <typename T>
inline T arrayPointerGetV(const ArrayInfo& info, const T* ptr, uint64_t idx)
{
    // pointer must come from array
    assert(uint64_t(ptr) >= uint64_t(info.data) && (((uint64_t(ptr) - uint64_t(info.data)) / info.byteStride) + idx) <= info.count);
    return *reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(ptr) + info.byteStride * idx);
}

template <typename Tarray, typename T>
inline void arrayFill(Tarray& info, T tval)
{
    for(uint64_t i = 0; i < info.count; i++)
    {
        *reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(info.data) + info.byteStride * i) = tval;
    }
}

template <typename Ta>
inline bool arrayTypedIsValid(const Ta& info)
{
    return arrayIsValid(info) && info.format == info.s_format;
}

template <typename Ta>
inline void arrayTypedSetV(Ta& info, uint64_t idx, typename Ta::value_type value)
{
    assert(idx < info.count);
    *reinterpret_cast<typename Ta::value_type*>(reinterpret_cast<uint8_t*>(info.data) + (idx * info.byteStride)) = value;
}

template <typename Ta>
inline typename Ta::value_type arrayTypedGetV(const Ta& info, uint64_t idx)
{
    assert(idx < info.count);
    return *reinterpret_cast<const typename Ta::value_type*>(reinterpret_cast<const uint8_t*>(info.data) + (idx * info.byteStride));
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

inline uint32_t meshGetTriangleMapping(const ArrayInfo_uint32& meshTriangleMappings, uint64_t idx)
{
    return arrayIsEmpty(meshTriangleMappings) ? uint32_t(idx) : arrayGetV<uint32_t>(meshTriangleMappings, idx);
}

inline bool meshIsTriangleDegenerate(Vector_uint32_3 indices)
{
    return (indices.x == indices.y || indices.y == indices.z || indices.z == indices.x);
}

// ensures that the barycentric interpolation sum
// `vertex[indices.x] * bary.w + vertex[indices.y] * bary.u + vertex[indices.z] * bary.v`
// is re-ordered based on the indices values. So that barycentric values on edges of
// adjacent triangles with identical vertex indices, get summed in the same order
// (the unused vertex will always have a weight of zero).
inline void meshReorderStableInterpolation(Vector_uint32_3& indices, BaryWUV_float& bary)
{
    if(indices.x > indices.y)
    {
        float b    = bary.w;
        bary.w     = bary.u;
        bary.u     = b;
        uint32_t i = indices.x;
        indices.x  = indices.y;
        indices.y  = i;
    }

    if(indices.y > indices.z)
    {
        float b    = bary.u;
        bary.u     = bary.v;
        bary.v     = b;
        uint32_t i = indices.y;
        indices.y  = indices.z;
        indices.z  = i;
    }

    if(indices.x > indices.y)
    {
        float b    = bary.w;
        bary.w     = bary.u;
        bary.u     = b;
        uint32_t i = indices.x;
        indices.x  = indices.y;
        indices.y  = i;
    }
}

//////////////////////////////////////////////////////////////////////////

inline bool meshtopoIsValid(const MeshTopology& topo)
{
    return arrayTypedIsValid(topo.edgeTriangleConnections) && arrayTypedIsValid(topo.edgeVertices)
           && topo.edgeVertices.count == topo.edgeTriangleRanges.count && arrayTypedIsValid(topo.vertexTriangleConnections)
           && arrayTypedIsValid(topo.vertexEdgeConnections) && arrayTypedIsValid(topo.triangleEdges)
           && arrayTypedIsValid(topo.triangleVertices) && topo.triangleEdges.count == topo.triangleVertices.count;
}

inline bool micromapIsValid(const Micromap& map)
{
    return arrayTypedIsValid(map.triangleSubdivLevels) && arrayTypedIsValid(map.triangleValueIndexOffsets)
           && map.triangleSubdivLevels.count == map.triangleValueIndexOffsets.count && arrayIsValid(map.values)
           && map.layout.pfnGetMicroTriangleIndex && map.layout.pfnGetMicroVertexIndex
           && (map.frequency == Frequency::ePerMicroTriangle || map.frequency == Frequency::ePerMicroVertex);
}

inline bool micromapIsValid(const MicromapCompressed& map)
{
    return arrayTypedIsValid(map.triangleSubdivLevels) && arrayTypedIsValid(map.triangleValueByteOffsets)
           && arrayTypedIsValid(map.triangleBlockFormats) && arrayIsValid(map.values)
           && map.triangleSubdivLevels.count == map.triangleValueByteOffsets.count
           && map.triangleSubdivLevels.count == map.triangleBlockFormats.count;
}

inline bool micromapIsValid(const MicromapPacked& map)
{
    return arrayTypedIsValid(map.triangleSubdivLevels) && arrayTypedIsValid(map.triangleValueByteOffsets)
           && map.triangleSubdivLevels.count == map.triangleValueByteOffsets.count && arrayIsValid(map.values)
           && map.layout.pfnGetMicroTriangleIndex && map.layout.pfnGetMicroVertexIndex
           && (map.frequency == Frequency::ePerMicroTriangle || map.frequency == Frequency::ePerMicroVertex)
           && (map.values.format == Format::eR11_unorm_packed_align32) && (map.values.byteStride == 1);
}

// only arrays
inline bool micromapAreArraysValid(const Micromap& map)
{
    return arrayTypedIsValid(map.triangleSubdivLevels) && arrayTypedIsValid(map.triangleValueIndexOffsets)
           && map.triangleSubdivLevels.count == map.triangleValueIndexOffsets.count && arrayIsValid(map.values);
}

template <class T>
inline const T* micromapGetTriangleValue(const Micromap& micromap, uint64_t triIdx, uint32_t valueIdx)
{
    uint32_t valueOffset = *arrayGet<uint32_t>(micromap.triangleValueIndexOffsets, triIdx);
    return reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(micromap.values.data)
                                      + (valueOffset + valueIdx) * micromap.values.byteStride);
}

template <class T>
inline T* micromapGetTriangleValue(Micromap& micromap, uint64_t triIdx, uint32_t valueIdx)
{
    uint32_t valueOffset = *arrayGet<uint32_t>(micromap.triangleValueIndexOffsets, triIdx);
    return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(micromap.values.data)
                                + (valueOffset + valueIdx) * micromap.values.byteStride);
}

//////////////////////////////////////////////////////////////////////////

inline uint32_t topoEdgeGetOtherVertex(Vector_uint32_2 edge, uint32_t vertexIdx)
{
    return (edge.x == vertexIdx) ? edge.y : edge.x;
}

inline uint32_t topoTriangleFindVertex(Vector_uint32_3 triangle, uint32_t vertexIdx)
{
    if(triangle.x == vertexIdx)
        return 0;
    else if(triangle.y == vertexIdx)
        return 1;
    else if(triangle.z == vertexIdx)
        return 2;
    else
        return INVALID_INDEX;
}

//////////////////////////////////////////////////////////////////////////

inline uint32_t subdivLevelGetSegmentCount(uint32_t subdivLevel)
{
    return (1u << subdivLevel);
}

inline uint32_t subdivLevelGetTriangleCount(uint32_t subdivLevel)
{
    return (1u << (subdivLevel * 2));
}

inline uint32_t subdivLevelGetTriangleCount(uint32_t subdivLevel, uint32_t edgeFlag)
{
    if(subdivLevel == 0)
        return 1;

    uint32_t segmentCount  = subdivLevelGetSegmentCount(subdivLevel);
    uint32_t triangleCount = (1u << (subdivLevel * 2));
    if(edgeFlag & 1)
        triangleCount -= segmentCount / 2;
    if(edgeFlag & 2)
        triangleCount -= segmentCount / 2;
    if(edgeFlag & 4)
        triangleCount -= segmentCount / 2;

    return triangleCount;
}

inline uint32_t subdivLevelGetVertexCount(uint32_t subdivLevel)
{
    uint32_t vertsPerEdgeCount = subdivLevelGetSegmentCount(subdivLevel) + 1;
    return (vertsPerEdgeCount * (vertsPerEdgeCount + 1)) / 2;
}

inline uint32_t subdivLevelGetVertexCount(uint32_t subdivLevel, uint32_t edgeFlag)
{
    if(subdivLevel == 0)
        return 3;

    uint32_t segmentCount      = subdivLevelGetSegmentCount(subdivLevel);
    uint32_t vertsPerEdgeCount = segmentCount + 1;
    uint32_t vertexCount       = (vertsPerEdgeCount * (vertsPerEdgeCount + 1)) / 2;
    if(edgeFlag & 1)
        vertexCount -= segmentCount / 2;
    if(edgeFlag & 2)
        vertexCount -= segmentCount / 2;
    if(edgeFlag & 4)
        vertexCount -= segmentCount / 2;

    return vertexCount;
}

inline uint32_t subdivLevelGetCount(uint32_t subdivLevel, Frequency freq)
{
    if(freq == Frequency::ePerMicroTriangle)
        return (1u << (subdivLevel * 2));
    else
    {
        return subdivLevelGetVertexCount(subdivLevel);
    }
}

//////////////////////////////////////////////////////////////////////////

inline BaryUV_uint16 makeBaryUV_uint16(uint32_t u, uint32_t v)
{
    return {uint16_t(u), uint16_t(v)};
}

inline BaryWUV_uint16 makeBaryWUV_uint16(uint32_t w, uint32_t u, uint32_t v)
{
    return {uint16_t(w), uint16_t(u), uint16_t(v)};
}

inline BaryWUV_float baryUVtoWUV_float(BaryUV_float coord)
{
    return {1.0f - coord.u - coord.v, coord.u, coord.v};
}

inline BaryWUV_float baryUVtoWUV_float(BaryUV_uint16 coord, uint32_t subdivLevel)
{
    uint32_t maxcoord = 1 << subdivLevel;
    float    mul      = 1.0f / float(maxcoord);
    return {float(maxcoord - coord.u - coord.v) * mul, float(coord.u) * mul, float(coord.v) * mul};
}

inline BaryWUV_uint16 baryUVtoWUV_uint(BaryUV_uint16 coord, uint32_t subdivLevel)
{
    uint32_t maxcoord = 1 << subdivLevel;
    return {uint16_t(maxcoord - coord.u - coord.v), coord.u, coord.v};
}

inline BaryWUV_float baryWUVtoWUV_float(BaryWUV_uint16 coord, uint32_t subdivLevel)
{
    uint32_t maxcoord = 1 << subdivLevel;
    float    mul      = 1.0f / float(maxcoord);
    return {float(coord.w) * mul, float(coord.u) * mul, float(coord.v) * mul};
}

inline bool baryUVisEqual(BaryUV_uint16 a, BaryUV_uint16 b)
{
    return a.u == b.u && a.v == b.v;
}

inline bool baryWUVisOnEdge(BaryWUV_uint16 b)
{
    return (b.w == 0 || b.u == 0 || b.v == 0);
}

inline bool baryWUVisOnEdge(BaryWUV_float b)
{
    return (b.w == 0 || b.u == 0 || b.v == 0);
}

template <typename T>
inline T baryWUVGetInterpolated(BaryWUV_float bary, T v0, T v1, T v2)
{
    return v0 * bary.w + v1 * bary.u + v2 * bary.v;
}

// generic vector, barycentric weights stored in .x .y .z
template <typename Tvector, typename T>
inline T baryVectorGetInterpolated(const Tvector bary, T v0, T v1, T v2)
{
    return v0 * bary.x + v1 * bary.y + v2 * bary.z;
}

//////////////////////////////////////////////////////////////////////////

inline uint32_t umajorUVtoLinear(uint32_t u, uint32_t v, uint32_t subdivLevel)
{
    uint32_t vertsPerEdgeCount = subdivLevelGetSegmentCount(subdivLevel) + 1;
    uint32_t trinum            = (u * (u + 1)) / 2;
    return u * (vertsPerEdgeCount + 1) - trinum + v;
}

inline uint32_t umajorUVtoLinear(BaryUV_uint16 coord, uint32_t subdivLevel)
{
    return umajorUVtoLinear(coord.u, coord.v, subdivLevel);
}

inline BaryUV_uint16 umajorLinearToUV(uint32_t index, uint32_t subdivLevel)
{
    uint32_t vertsPerEdgeCount = subdivLevelGetSegmentCount(subdivLevel) + 1;

    uint32_t u = uint32_t(floor(
        double(-sqrt((2 * vertsPerEdgeCount + 1) * (2 * vertsPerEdgeCount + 1) - 8 * index) + 2 * vertsPerEdgeCount + 1) / 2.0));
    uint32_t v = index - u * (2 * vertsPerEdgeCount - u + 1) / 2;
    return {uint16_t(u), uint16_t(v)};
}

//////////////////////////////////////////////////////////////////////////

inline BaryUV_uint16 blockTriangleLocalToBaseUV(const BlockTriangle* info, BaryUV_uint16 locaUV)
{
    int32_t anchor[2] = {info->vertices[0].u, info->vertices[0].v};
    int32_t signs[2] = {info->vertices[1].u > info->vertices[0].u ? 1 : -1, info->vertices[2].v > info->vertices[0].v ? 1 : -1};
    int32_t local[2] = {locaUV.u, locaUV.v};
    local[0] *= signs[0];
    local[1] *= signs[1];

    BaryUV_uint16 baseUV;
    baseUV.u = uint16_t(anchor[0] + local[0] + (signs[0] != signs[1] ? -local[1] : 0));
    baseUV.v = uint16_t(anchor[1] + local[1]);
    return baseUV;
}

// may return invalid / out of bounds coords, use baryUVisValid
inline BaryUV_uint16 blockTriangleBaseToLocalUV(const BlockTriangle* info, BaryUV_uint16 baseUV)
{
    int32_t base[2]   = {baseUV.u, baseUV.v};
    int32_t anchor[2] = {info->vertices[0].u, info->vertices[0].v};
    int32_t signs[2] = {info->vertices[1].u > info->vertices[0].u ? 1 : -1, info->vertices[2].v > info->vertices[0].v ? 1 : -1};
    int32_t local[2] = {};
    local[1]         = base[1] - anchor[1];
    local[0]         = base[0] - anchor[0] - (signs[0] != signs[1] ? -local[1] : 0);

    local[0] *= signs[0];
    local[1] *= signs[1];

    BaryUV_uint16 locaUV;
    locaUV.u = uint16_t(local[0]);
    locaUV.v = uint16_t(local[1]);
    return locaUV;
}

//////////////////////////////////////////////////////////////////////////

class MeshTopologyUtil
{
  private:
    static inline const uint32_t* topoGetConnected(const ArrayInfo_range32& ranges,
                                                   const ArrayInfo_uint32&  connections,
                                                   uint32_t                 idx,
                                                   uint32_t&                outCount)
    {
        const Range32* range = arrayGet<Range32>(ranges, idx);
        outCount             = range->count;
        return outCount ? arrayGet<uint32_t>((const ArrayInfo&)connections, range->first) : nullptr;
    }

    static inline ArrayInfo_uint32 topoGetConnectedArray(const ArrayInfo_range32& ranges, const ArrayInfo_uint32& connections, uint32_t idx)
    {
        const Range32* range      = arrayGet<Range32>(ranges, idx);
        const uint32_t* firstAddr = range->count ? arrayGet<uint32_t>((const ArrayInfo&)connections, range->first) : nullptr;
        return ArrayInfo_uint32(firstAddr, range->count, connections.byteStride);
    }

  public:
    MeshTopologyUtil(const MeshTopology& topo)
        : m_topo(topo)
    {
    }
    const MeshTopology& m_topo;

    static const uint32_t TRIANGLE_VERTEX_COUNT = 3u;
    static const uint32_t TRIANGLE_EDGE_COUNT   = 3u;
    static const uint32_t EDGE_VERTEX_COUNT     = 2u;

    inline Vector_uint32_3 getTriangleVertices(uint32_t triIdx) const
    {
        const uint32_t* data = arrayGet<uint32_t>((const ArrayInfo&)m_topo.triangleVertices, triIdx);
        return {data[0], data[1], data[2]};
    }

    inline Vector_uint32_3 getTriangleEdges(uint32_t triIdx) const
    {
        const uint32_t* data = arrayGet<uint32_t>((const ArrayInfo&)m_topo.triangleEdges, triIdx);
        return {data[0], data[1], data[2]};
    }

    // use return of getVertexTriangles in getVertexTriangle
    inline const uint32_t* getVertexTriangles(uint32_t vertexIdx, uint32_t& outCount) const
    {
        return topoGetConnected(m_topo.vertexTriangleRanges, m_topo.vertexTriangleConnections, vertexIdx, outCount);
    }

    inline ArrayInfo_uint32 getVertexTrianglesArray(uint32_t vertexIdx)
    {
        return topoGetConnectedArray(m_topo.vertexTriangleRanges, m_topo.vertexTriangleConnections, vertexIdx);
    }

    inline const uint32_t getVertexTriangle(const uint32_t* vertexTriangles, uint32_t idx) const
    {
        return *arrayPointerGet((const ArrayInfo&)m_topo.vertexTriangleConnections, vertexTriangles, idx);
    }

    // use return of getVertexEdges in getVertexEdge
    inline const uint32_t* getVertexEdges(uint32_t vertexIdx, uint32_t& outCount) const
    {
        return topoGetConnected(m_topo.vertexEdgeRanges, m_topo.vertexEdgeConnections, vertexIdx, outCount);
    }

    inline ArrayInfo_uint32 getVertexEdgesArray(uint32_t vertexIdx)
    {
        return topoGetConnectedArray(m_topo.vertexEdgeRanges, m_topo.vertexEdgeConnections, vertexIdx);
    }

    inline const uint32_t getVertexEdge(const uint32_t* vertexEdges, uint32_t idx) const
    {
        return *arrayPointerGet((const ArrayInfo&)m_topo.vertexEdgeConnections, vertexEdges, idx);
    }

    // use return of getEdgeTriangles in getEdgeTriangle
    inline const uint32_t* getEdgeTriangles(uint32_t edgeIdx, uint32_t& outCount) const
    {
        return topoGetConnected(m_topo.edgeTriangleRanges, m_topo.edgeTriangleConnections, edgeIdx, outCount);
    }

    inline ArrayInfo_uint32 getEdgeTrianglesArray(uint32_t edgeIdx)
    {
        return topoGetConnectedArray(m_topo.edgeTriangleRanges, m_topo.edgeTriangleConnections, edgeIdx);
    }

    inline const uint32_t getEdgeTriangle(const uint32_t* edgeTriangles, uint32_t idx) const
    {
        return *arrayPointerGet((const ArrayInfo&)m_topo.edgeTriangleConnections, edgeTriangles, idx);
    }

    // in the winding order of the first triangle it was used with
    inline Vector_uint32_2 getEdgeVertices(uint32_t edgeIdx) const
    {
        return *arrayGet<Vector_uint32_2>((const ArrayInfo&)m_topo.edgeVertices, edgeIdx);
    }

    // can return INVALID_INDEX
    inline uint32_t findEdge(uint32_t vertexA, uint32_t vertexB) const
    {
        uint32_t        count;
        const uint32_t* vEdges = getVertexEdges(vertexA, count);
        if(!vEdges || !count)
        {
            return INVALID_INDEX;
        }

        for(uint32_t i = 0; i < count; i++)
        {
            uint32_t edgeIndex = getVertexEdge(vEdges, i);
            if(topoEdgeGetOtherVertex(getEdgeVertices(edgeIndex), vertexA) == vertexB)
            {
                return edgeIndex;
            }
        }

        return INVALID_INDEX;
    }
};

}  // namespace micromesh
