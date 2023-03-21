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

#include <cstddef>
#include <cstdint>
#include <cassert>

namespace micromesh
{
//////////////////////////////////////////////

// (1 << 15) still fits in uint16_t
#define MICROMESH_MAX_SUBDIV_LEVEL 15

struct BaryUV_uint16
{
    uint16_t u{};
    uint16_t v{};

    inline uint16_t& operator[](size_t idx)
    {
        assert(idx < 2);
        return (&u)[idx];
    }
    inline const uint16_t& operator[](size_t idx) const
    {
        assert(idx < 2);
        return (&u)[idx];
    }
};

struct BaryWUV_uint16
{
    uint16_t w{};
    uint16_t u{};
    uint16_t v{};

    inline uint16_t& operator[](size_t idx)
    {
        assert(idx < 3);
        return (&w)[idx];
    }
    inline const uint16_t& operator[](size_t idx) const
    {
        assert(idx < 3);
        return (&w)[idx];
    }
};

struct BaryUV_float
{
    float u{};
    float v{};

    inline float& operator[](size_t idx)
    {
        assert(idx < 2);
        return (&u)[idx];
    }
    inline const float& operator[](size_t idx) const
    {
        assert(idx < 2);
        return (&u)[idx];
    }
};

struct BaryWUV_float
{
    float w{};
    float u{};
    float v{};

    inline float& operator[](size_t idx)
    {
        assert(idx < 3);
        return (&w)[idx];
    }
    inline const float& operator[](size_t idx) const
    {
        assert(idx < 3);
        return (&w)[idx];
    }
};

//////////////////////////////////////////////

struct MicroVertexInfo
{
    // base triangle of this micro vertex
    uint32_t triangleIndex;
    // uv coordinate within the base triangle
    BaryUV_uint16 vertexUV;
};

//////////////////////////////////////////////

struct Vector_float_2
{
    float x{};
    float y{};

    inline float& operator[](size_t idx)
    {
        assert(idx < 2);
        return (&x)[idx];
    }
    inline const float& operator[](size_t idx) const
    {
        assert(idx < 2);
        return (&x)[idx];
    }
};

struct Vector_float_3
{
    float x{};
    float y{};
    float z{};

    inline float& operator[](size_t idx)
    {
        assert(idx < 3);
        return (&x)[idx];
    }
    inline const float& operator[](size_t idx) const
    {
        assert(idx < 3);
        return (&x)[idx];
    }
};

struct Vector_float_4
{
    float x{};
    float y{};
    float z{};
    float w{};

    inline float& operator[](size_t idx)
    {
        assert(idx < 4);
        return (&x)[idx];
    }
    inline const float& operator[](size_t idx) const
    {
        assert(idx < 4);
        return (&x)[idx];
    }
};

struct Matrix_float_4x4
{
    Vector_float_4 columns[4]{};

    inline Vector_float_4& operator[](size_t idx)
    {
        assert(idx < 4);
        return columns[idx];
    }
    inline const Vector_float_4& operator[](size_t idx) const
    {
        assert(idx < 4);
        return columns[idx];
    }
};

struct Vector_uint16_2
{
    uint16_t x{};
    uint16_t y{};

    inline uint16_t& operator[](size_t idx)
    {
        assert(idx < 2);
        return (&x)[idx];
    }
    inline const uint16_t& operator[](size_t idx) const
    {
        assert(idx < 2);
        return (&x)[idx];
    }
};

struct Vector_uint32_2
{
    uint32_t x{};
    uint32_t y{};

    inline uint32_t& operator[](size_t idx)
    {
        assert(idx < 2);
        return (&x)[idx];
    }
    inline const uint32_t& operator[](size_t idx) const
    {
        assert(idx < 2);
        return (&x)[idx];
    }
};

struct Vector_uint32_3
{
    uint32_t x{};
    uint32_t y{};
    uint32_t z{};

    inline uint32_t& operator[](size_t idx)
    {
        assert(idx < 3);
        return (&x)[idx];
    }
    inline const uint32_t& operator[](size_t idx) const
    {
        assert(idx < 3);
        return (&x)[idx];
    }
};

struct Vector_uint32_4
{
    uint32_t x{};
    uint32_t y{};
    uint32_t z{};
    uint32_t w{};

    inline uint32_t& operator[](size_t idx)
    {
        assert(idx < 4);
        return (&x)[idx];
    }
    inline const uint32_t& operator[](size_t idx) const
    {
        assert(idx < 4);
        return (&x)[idx];
    }
};

//////////////////////////////////////////////

enum class Result : uint32_t
{
    eSuccess = 0,
    eFailure,
    // a function is re-entrant and must be called multiple times to complete
    eContinue,
    // value frequency was wrong
    eInvalidFrequency,
    // format was wrong
    eInvalidFormat,
    // blockformat was wrong
    eInvalidBlockFormat,
    // count or offset + count or index creates out of range
    // also applies to provided array counts
    eInvalidRange,
    // a given parameter is invalid (numerical or enum)
    eInvalidValue,
    // the contents of MicromapLayout were not valid
    eInvalidLayout,
    // a context is used out of order for a started operation sequence
    // Operations with multiple steps need to be processed in-order
    eInvalidOperationOrder,
    // two displaced triangles had gaps along their common edge before they
    // were encoded
    eMismatchingInputEdgeValues,
    // two displaced triangles had gaps along their common edge after they
    // were encoded
    eMismatchingOutputEdgeValues,
    // requested version is not available
    eUnsupportedVersion,
    // unsupported shader language
    eUnsupportedShaderCodeType
};

//////////////////////////////////////////////

// This enumeration lists all potentially supported data formats for the
// micromesh SDK. Individual functions may only support subsets of these
// formats; see their documentation for more information.
//
// The enum values here match VkFormat's enum values, when there are
// corresponding formats. For instance, Format::eRG16_sfloat has the same
// value as VK_FORMAT_R16G16_SFLOAT, and Format::eDispC1_r11_unorm_block
// corresponds to a VkFormat allocated by Vulkan extension 397.
//
// We encourage you to use the micromesh SDK with your own file formats,
// and to extend and modify the micromesh SDK for your use cases. If you
// have a new format we have two recommended ways to add it to this enum:
// if the format may potentially have Vulkan support and if requesting
// a Vulkan extension is possible, request an extension ID from Khronos.
// Alternatively, use the following method without a Vulkan extension ID to
// avoid collisions with future extensions with high probability: generate a
// random 28-bit int, add 2^29 == 536870912, and use the result as the new enum
// ID. For example, if your random 29-bit number was 123456789, you would add
//   eYourNewFormatName = 660327701, // 123456789 + 536870912
enum class Format : uint32_t
{
    eUndefined     = 0,
    eR8_unorm      = 9,
    eR8_snorm      = 10,
    eR8_uint       = 13,
    eR8_sint       = 14,
    eRG8_unorm     = 16,
    eRG8_snorm     = 17,
    eRG8_uint      = 20,
    eRG8_sint      = 21,
    eRGB8_unorm    = 23,
    eRGB8_snorm    = 24,
    eRGB8_uint     = 27,
    eRGB8_sint     = 28,
    eRGBA8_unorm   = 37,
    eRGBA8_snorm   = 38,
    eRGBA8_uint    = 41,
    eRGBA8_sint    = 42,
    eR16_unorm     = 70,
    eR16_snorm     = 71,
    eR16_uint      = 74,
    eR16_sint      = 75,
    eR16_sfloat    = 76,
    eRG16_unorm    = 77,
    eRG16_snorm    = 78,
    eRG16_uint     = 81,
    eRG16_sint     = 82,
    eRG16_sfloat   = 83,
    eRGB16_unorm   = 84,
    eRGB16_snorm   = 85,
    eRGB16_uint    = 88,
    eRGB16_sint    = 89,
    eRGB16_sfloat  = 90,
    eRGBA16_unorm  = 91,
    eRGBA16_snorm  = 92,
    eRGBA16_uint   = 95,
    eRGBA16_sint   = 96,
    eRGBA16_sfloat = 97,
    eR32_uint      = 98,
    eR32_sint      = 99,
    eR32_sfloat    = 100,
    eRG32_uint     = 101,
    eRG32_sint     = 102,
    eRG32_sfloat   = 103,
    eRGB32_uint    = 104,
    eRGB32_sint    = 105,
    eRGB32_sfloat  = 106,
    eRGBA32_uint   = 107,
    eRGBA32_sint   = 108,
    eRGBA32_sfloat = 109,
    eR64_uint      = 110,
    eR64_sint      = 111,
    eR64_sfloat    = 112,
    eRG64_uint     = 113,
    eRG64_sint     = 114,
    eRG64_sfloat   = 115,
    eRGB64_uint    = 116,
    eRGB64_sint    = 117,
    eRGB64_sfloat  = 118,
    eRGBA64_uint   = 119,
    eRGBA64_sint   = 120,
    eRGBA64_sfloat = 121,

    // opacity encoding (based on VK NV extension reservation 397)

    // block-compressed, variable format used
    // for uncompressed 1 or 2 bit data stored in 8-bit
    // valueByteSize = 1
    // byteStride must be 1
    eOpaC1_rx_uint_block = 1000396000,

    // displacement encoding  (based on VK NV extension reservation 398)

    // block-compressed, variable format used
    // for compressed data stored in blocks of 512 or 1024 bits
    // valueByteSize = 1
    // byteStride must be 1
    eDispC1_r11_unorm_block = 1000397000,

    // for uncompressed data 1 x 11 bit stored in 16-bit
    eR11_unorm_pack16 = 1000397001,

    // variable packed format
    // for uncompressed data 1 x 11 bit densely packed sequence of 32bit values.
    // Each triangle starts at a 32-bit boundary.
    // valueByteSize = 1
    // minmax as eR11_unorm_pack16
    eR11_unorm_packed_align32 = 1000397002,
};

enum class BlockFormatOpaC1 : uint16_t
{
    eInvalid = 0,
    // 8 x R1_UINT in single 8-bit
    eR1_uint_x8 = 1,
    // 4 x R2_UINT in single 8-bit
    eR2_uint_x4 = 2,
};

enum class BlockFormatDispC1 : uint16_t
{
    eInvalid                 = 0,
    eR11_unorm_lvl3_pack512  = 1,
    eR11_unorm_lvl4_pack1024 = 2,
    eR11_unorm_lvl5_pack1024 = 3,
};

enum class ChannelType : uint32_t
{
    eUndefined,
    eUnorm,
    eSnorm,
    eUint,
    eSint,
    eSfloat,
};

// not class to keep uint32
enum ChannelOffset : uint32_t
{
    eChannelR,
    eChannelG,
    eChannelB,
    eChannelA,
};

struct FormatInfo
{
    ChannelType channelType;
    // can be 0 if variable / compressed
    uint32_t channelBitCount;
    // 1 to 4
    uint32_t channelCount;
    // default size
    uint32_t byteSize;
    // if format is compressed or packed, then it's not trivial to access
    bool isCompressedOrPacked;
    // valueCount is typically 1 but
    // can be 0 or greater if specially packed / compressed
    // some operations may not support such packings
    uint32_t valueCount;
};

//////////////////////////////////////////////

struct ArrayInfo
{
    void*    data   = nullptr;
    uint64_t count  = 0ull;
    Format   format = Format::eUndefined;
    // compresed data must use byteStride == 1
    uint32_t byteStride = 0u;
};

struct Range32
{
    uint32_t first = 0u;
    uint32_t count = 0u;
};

struct Range64
{
    uint64_t first = 0ull;
    uint64_t count = 0ull;
};

// a few special structs for readability of common single format inputs
// stride can be altered by user

#define MICROMESH_ARRAY_INFO_TYPED(_n, _f, _t)                                                                         \
    struct _n : public ArrayInfo                                                                                       \
    {                                                                                                                  \
        static const Format s_format = _f;                                                                             \
        typedef _t          value_type;                                                                                \
                                                                                                                       \
        _n()                                                                                                           \
        {                                                                                                              \
            format     = _f;                                                                                           \
            byteStride = uint32_t(sizeof(_t));                                                                         \
        }                                                                                                              \
        _n(const void* _data, uint64_t _count, uint32_t _byteStride = 0)                                               \
        {                                                                                                              \
            format     = _f;                                                                                           \
            byteStride = _byteStride ? _byteStride : uint32_t(sizeof(_t));                                             \
            count      = _count;                                                                                       \
            data       = const_cast<void*>(_data);                                                                     \
        }                                                                                                              \
        _n(void* _data, uint64_t _count, uint32_t _byteStride = 0)                                                     \
        {                                                                                                              \
            format     = _f;                                                                                           \
            byteStride = _byteStride ? _byteStride : uint32_t(sizeof(_t));                                             \
            count      = _count;                                                                                       \
            data       = _data;                                                                                        \
        }                                                                                                              \
    };                                                                                                                 \
    static_assert(sizeof(_n) == sizeof(ArrayInfo), "ArrayInfo sizeof mismatch " #_n);

MICROMESH_ARRAY_INFO_TYPED(ArrayInfo_uint8, Format::eR8_uint, uint8_t);
MICROMESH_ARRAY_INFO_TYPED(ArrayInfo_uint16, Format::eR16_uint, uint16_t);
MICROMESH_ARRAY_INFO_TYPED(ArrayInfo_uint16_2, Format::eRG16_uint, Vector_uint16_2);

MICROMESH_ARRAY_INFO_TYPED(ArrayInfo_uint32, Format::eR32_uint, uint32_t);
MICROMESH_ARRAY_INFO_TYPED(ArrayInfo_uint32_2, Format::eRG32_uint, Vector_uint32_2);
MICROMESH_ARRAY_INFO_TYPED(ArrayInfo_uint32_3, Format::eRGB32_uint, Vector_uint32_3);
MICROMESH_ARRAY_INFO_TYPED(ArrayInfo_uint32_4, Format::eRGBA32_uint, Vector_uint32_4);

MICROMESH_ARRAY_INFO_TYPED(ArrayInfo_float, Format::eR32_sfloat, float);
MICROMESH_ARRAY_INFO_TYPED(ArrayInfo_float_2, Format::eRG32_sfloat, Vector_float_2);
MICROMESH_ARRAY_INFO_TYPED(ArrayInfo_float_3, Format::eRGB32_sfloat, Vector_float_3);
MICROMESH_ARRAY_INFO_TYPED(ArrayInfo_float_4, Format::eRGBA32_sfloat, Vector_float_4);

MICROMESH_ARRAY_INFO_TYPED(ArrayInfo_range32, Format::eRG32_uint, Range32);
MICROMESH_ARRAY_INFO_TYPED(ArrayInfo_range64, Format::eRG64_uint, Range64);

//////////////////////////////////////////////

enum class Frequency : uint32_t
{
    ePerMicroVertex,
    ePerMicroTriangle,
};

enum class StandardLayoutType : uint32_t
{
    // not a standard layout type, custom functions used
    eUnknown,

    // values are stored left to right
    // top to bottom for the following
    // triangle WUV
    //
    // W _ V
    // | /
    // U
    eUmajor,

    // special spatial curve that is the result of applying this splitting recursively
    // (depth-first)
    // triangle WUV
    //
    // W___x___V
    // |0 /|3 /
    // | /1| /
    // x___x
    // |2 /
    // | /
    // U
    eBirdCurve,
};

// mapping from uv-coordinate of the barycentric micromesh grid to the value storage index
typedef uint32_t (*PFN_getMicroVertexIndex)(uint32_t u, uint32_t v, uint32_t subdivLevel, void* userData);

// mapping from uv-coordinate of first vertex in triangle, for upper triangle the coordinate is the corner
// of the quad that the upper triangle opposes.
//
// x___x
// |0\1|  number inside triangle reflects isUpperTriangle
// uv__x
// The vertices within a micro triangle itself are expected to be ordered to have the horizontal
// edge first, and using the illustration above from left to right (increasing u coordinate)
typedef uint32_t (*PFN_getMicroTriangleIndex)(uint32_t u, uint32_t v, uint32_t isUpperTriangle, uint32_t subdivLevel, void* userData);

// a valid layout must have both function pointers set
struct MicromapLayout
{
    void*                     userData                 = nullptr;
    PFN_getMicroVertexIndex   pfnGetMicroVertexIndex   = nullptr;
    PFN_getMicroTriangleIndex pfnGetMicroTriangleIndex = nullptr;
};

struct MicromapValue
{
    // RGBA has 4 channels
    union
    {
        float    value_float[4];
        uint32_t value_uint32[4];
        int32_t  value_int32[4];
        uint16_t value_uint16[4];
        int16_t  value_int16[4];
        uint8_t  value_uint8[4];
        int8_t   value_int8[4];
    };

    MicromapValue() { value_uint32[0] = value_uint32[1] = value_uint32[2] = value_uint32[3] = 0; }
};

struct MicromapValueFloatExpansion
{
    // especially quantized data may represent a different value range
    // but even float could be stored as [0,1] but representing something else
    //
    // RGBA has 4 channels
    float bias[4];
    float scale[4];

    MicromapValueFloatExpansion()
    {
        bias[0] = bias[1] = bias[2] = bias[3] = 0.0f;
        scale[0] = scale[1] = scale[2] = scale[3] = 1.0f;
    }
};

// Micromap references uncompressed data
// trivial to index and values.count reflects the
// total number of values stored for all triangles.
struct Micromap
{
    // values for all triangles
    ArrayInfo values;
    // applicable to unorm/snorm/float
    MicromapValueFloatExpansion valueFloatExpansion;
    // lowest subdivision level used in any triangle
    uint32_t minSubdivLevel{};
    // highest subdivision level used in any triangle
    uint32_t maxSubdivLevel{};
    // per-triangle subdivision level
    ArrayInfo_uint16 triangleSubdivLevels;
    // per-triangle offset into values array in indices.
    // marks the begin of where the triangle's data
    // is stored
    ArrayInfo_uint32 triangleValueIndexOffsets;

    // storage ordering of values in the barycentric grid
    // as well as whether the values are per-microvertex or per-microtriangle
    MicromapLayout layout;
    // at what frequency values are stored
    Frequency frequency{};
};

// Note:
// While both compressed and packed use triangleValueByteOffsets,
// it is compatible with triangleValueIndexOffsets, as the byteStride
// for compressed or packed formats must be 1 byte.
// All structs match in content up to
// triangleValueIndexOffsets / triangleValueByteOffsets.

// MicromapPacked references specially packed data, without trivial indexing
// but still relatively simple to decode, compared to a compressed format
// currently only eR11_unorm_packed_align32
// values.count will reflect the number of bytes, not the number
// of values stored.
struct MicromapPacked
{
    // packed values for all triangles
    // byteStride must be 1
    ArrayInfo values;
    // applicable to unorm/snorm/float
    MicromapValueFloatExpansion valueFloatExpansion;
    // lowest subdivision level used in any triangle
    uint32_t minSubdivLevel{};
    // highest subdivision level used in any triangle
    uint32_t maxSubdivLevel{};
    // per-triangle subdivision level
    ArrayInfo_uint16 triangleSubdivLevels;
    // per-triangle offset into values array in bytes.
    // marks the begin of where the triangle's data
    // is stored.
    ArrayInfo_uint32 triangleValueByteOffsets;

    // storage ordering of values in the barycentric grid
    // as well as whether the values are per-microvertex or per-microtriangle
    MicromapLayout layout;
    // at what frequency values are stored
    Frequency frequency{};
};

// MicromapCompressed references block-compressed data
// each triangle can use a different block-compression format
// which is provided.
// Block compression formats do store values in a fixed ordering
// and also define the frequency of them.
// values.count will reflect the number of bytes, not the number
// of values stored.
struct MicromapCompressed
{
    // compressed values for all triangles
    // byteStride must be 1
    ArrayInfo values;
    // applicable to unorm/snorm/float
    MicromapValueFloatExpansion valueFloatExpansion;
    // lowest subdivision level used in any triangle
    uint32_t minSubdivLevel{};
    // highest subdivision level used in any triangle
    uint32_t maxSubdivLevel{};
    // per-triangle subdivision level
    ArrayInfo_uint16 triangleSubdivLevels;
    // per-triangle offset into values array in bytes.
    // marks the begin of where the triangle's data
    // is stored.
    ArrayInfo_uint32 triangleValueByteOffsets;
    // per-triangle block format
    ArrayInfo_uint16 triangleBlockFormats;

    // no layout/frequency needed as it is implicit by compressed format
};

// not directly used by the sdk
// but useful utility struct
enum class MicromapType : uint32_t
{
    eUncompressed,
    ePacked,
    eCompressed,
    eInvalid,
};

struct MicromapGeneric
{
    // the type depends on the format, see `micromeshFormatGetMicromapType`
    MicromapType type = micromesh::MicromapType::eInvalid;
    union
    {
        micromesh::Micromap           uncompressed;
        micromesh::MicromapPacked     packed;
        micromesh::MicromapCompressed compressed;
    };

    MicromapGeneric() { uncompressed = micromesh::Micromap(); }
};

// compressed data can provide histogram information how
// certain blockFormats are used
struct BlockFormatUsage
{
    // how many triangles used this combination of subdivLevel and blockFormat
    uint32_t count{};
    uint32_t subdivLevel{};  // intentional u32 here
    uint32_t blockFormat{};  // intentional u32 here
};

struct MicromapBlockFormatUsage
{
    uint32_t          entriesCount = 0;
    BlockFormatUsage* entries      = nullptr;
};

// has applied instancing through
struct MeshBlockFormatUsage
{
    uint32_t          entriesCount = 0;
    BlockFormatUsage* entries      = nullptr;
};

//////////////////////////////////////////////

static const uint32_t INVALID_INDEX = ~0;
// INVALID_INDEX-1 to reserve invalid index
static const uint64_t MAX_UINT32_COUNT = 0xFFFFFFFEull;


struct DirectionBounds
{
    float bias;
    float scale;
};

// MeshProperties struct is never used by the api directly,
// but various operations make use of individual members.
// It exists for documentation purposes only.
// Triangle- and vertex-counts must stay within
// uint32_t limit.
struct MeshProperties
{
    // per-vertex positions
    ArrayInfo_float_3 meshVertexPositions;
    // per-vertex shading normals
    ArrayInfo_float_3 meshVertexNormals;
    // per-vertex texture coordinates
    ArrayInfo_float_2 meshVertexTexcoords;

    // per-vertex displacement directions
    ArrayInfo_float_3 meshVertexDirections;
    // per-vertex displacement direction bounds (bias,scale)
    ArrayInfo_float_2 meshVertexDirectionBounds;

    // per-triangle vertex indices
    ArrayInfo_uint32_3 meshTriangleVertices;

    // per-triangle vertex indices, which represent the topology
    // May mismatch from the generic meshTriangleVertices,
    // as for example separate texture coordinates could cause different
    // vertex indices for the meshTriangleVertices, but are expected to be
    // the same indices for meshTopologyTriangleVertices;
    ArrayInfo_uint32_3 meshTopologyTriangleVertices;
};

//////////////////////////////////////////////

// MeshSubdivProperties struct is never used by the api directly,
// but various operations make use of individual members.
// It exists for documentation purposes only
// Triangle- and vertex-counts must stay within
// uint32_t limit.
struct MeshSubdivProperties
{
    // highest subdivision level used in any triangle
    uint32_t maxSubdivLevel{};

    // per-mesh-triangle subdivision levels
    ArrayInfo_uint16 meshTriangleSubdivLevels;

    // per-mesh-triangle mapping index to a micromap triangle
    // if empty means 1:1 mapping between mesh-triangle and micromap-triangle
    // can use INVALID_MAPPING_INDEX to ignore a triangle
    // otherwise mapping index must be < meshTriangleMappings.count
    ArrayInfo_uint32 meshTriangleMappings;

    // per-mesh-triangle edge decimation flag.
    // Set nth bit if nth edge has neighboring triangle with
    // a subdivision level that is one level lower.
    // if empty means 0 for all triangles
    ArrayInfo_uint8 meshTrianglePrimitiveFlags;
};

//////////////////////////////////////////////

// MeshTopology is used to provide information about the
// topological relationship between vertices, triangles and
// edges. For micromesh data it is frequently used to ensure
// watertight values along mesh edges. Because each micromesh
// triangle will store per-microvertex data along shared edges
// individually, these mismatches can occur if the values are
// not processed properly using this connectivity.
// No assumptions are made about the ordering of connections.
// Triangle- and vertex-counts must stay within
// uint32_t limit.
struct MeshTopology
{
    // maximum number of edges a single vertex is used by
    uint32_t maxVertexEdgeValence = 0;
    // maximum number of triangles a single vertex is used by
    uint32_t maxVertexTriangleValence = 0;
    // maximum number of triangles an edge is used by
    uint32_t maxEdgeTriangleValence = 0;
    // non-zero if an edge has higher valence than 2 or
    // if the edge winding is inconsistent
    // (does not cover vertices that connect triangles
    //  that do not share an-edge)
    bool isNonManifold = false;

    // per-triangle vertex indices
    ArrayInfo_uint32_3 triangleVertices;
    // per-triangle edge indices
    // edge ordering {v0,v1}, {v1,v2}, {v2,v0}
    // can have INVALID_INDEX indices for degenerated triangles
    ArrayInfo_uint32_3 triangleEdges;

    // per-vertex range of triangles used by this vertex into triangle connections
    ArrayInfo_range32 vertexTriangleRanges;
    // index of triangle that a vertex is used with
    ArrayInfo_uint32 vertexTriangleConnections;
    // per-vertex range of edges used by this vertex into edge connections
    ArrayInfo_range32 vertexEdgeRanges;
    // index of edge that a vertex is used with
    ArrayInfo_uint32 vertexEdgeConnections;

    // per-edge vertex indices
    ArrayInfo_uint32_2 edgeVertices;
    // per-edge range of triangles used by this edge into triangle connections
    ArrayInfo_range32 edgeTriangleRanges;
    // index of triangle that a edge is used with
    ArrayInfo_uint32 edgeTriangleConnections;
};

//////////////////////////////////////////////

// Many functions support message callbacks, which can provide
// more detailed error information.

// Describes the importance of a message, from low to high.
enum class MessageSeverity : uint32_t
{
    eInfo    = 0,
    eWarning = 1,
    eError   = 2
};

// A micromeshMessageCallback function takes a message severity level, a
// null-terminated description of the message (usually with useful information
// for debugging; no newline at end), and a custom user pointer set when
// calling micromeshOpContextSetMessageCallback().
// Here's an example of a typical message callback:
// ```
// void MyCallback(MessageSeverity severity, const char* message, void* mydata)
// {
//     // This example ignores info messages, and only prints warnings and errors.
//     if(severity == MessageSeverity::eWarning)
//     {
//         printf("WARNING: %s\n", message);
//     }
//     else if (severity == MessageSeverity::eError)
//     {
//         printf("ERROR: %s\n", message);
//     }
// }
// ```
typedef void (*PFN_micromeshMessageCallback)(MessageSeverity severity, const char* message, uint32_t threadIndex, const void* userData);

// Passed to micromeshOpContextSetMessageCallback() to set the OpContext's
// message handler. This can also be used to print messages from functions that
// don't take an OpContext.
struct MessageCallbackInfo
{
    // If pfnCallback is null, no messages will be logged.
    PFN_micromeshMessageCallback pfnCallback = nullptr;
    // Arbitrary pointer.
    const void* userData = nullptr;
};

//////////////////////////////////////////////
// Operations related
//
// All major operations are done using a context.
// A context can work only at one operation step at a time.
// If an operation requires multiple steps to complete, these must be called
// in-order using the same context that started the operation.
// Such a sequence can be aborted with micromeshOpContextAbort.
// Contexts cannot be used concurrently unless specifically mentioned.
// See `micromesh_operations.h` for details.
typedef struct OpContext_s* OpContext;

// an internal data structure that handles thread-safe
// vertex deduplication based on hash & checksum pairs.
// See `micromesh_operations.h` for details.
typedef struct VertexDedup_s* VertexDedup;

struct TriangleSwizzle
{
    // triangle WUV can be shifted to
    // 0: WUV
    // 1: UVW
    // 2: VWU
    uint8_t shift : 2;
    // after shift perform flip
    // WUV -> WVU
    uint8_t flipVertexUV : 1;
};

}  // namespace micromesh
