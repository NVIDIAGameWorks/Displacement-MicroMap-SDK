//
// Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#include <cstring>
#include <limits>
#include <micromesh/micromesh_utils.h>
#include <micromesh_internal/micromesh_context.h>
#include <micromesh_internal/micromesh_math.h>

namespace micromesh
{
static_assert(offsetof(Micromap, values) == offsetof(MicromapPacked, values),
              "common member offset mismatch Micromap/MicromapPacked");
static_assert(offsetof(Micromap, valueFloatExpansion) == offsetof(MicromapPacked, valueFloatExpansion),
              "common member offset mismatch Micromap/MicromapPacked");
static_assert(offsetof(Micromap, minSubdivLevel) == offsetof(MicromapPacked, minSubdivLevel),
              "common member offset mismatch Micromap/MicromapPacked");
static_assert(offsetof(Micromap, maxSubdivLevel) == offsetof(MicromapPacked, maxSubdivLevel),
              "common member offset mismatch Micromap/MicromapPacked");
static_assert(offsetof(Micromap, triangleSubdivLevels) == offsetof(MicromapPacked, triangleSubdivLevels),
              "common member offset mismatch Micromap/MicromapPacked");
static_assert(offsetof(Micromap, triangleValueIndexOffsets) == offsetof(MicromapPacked, triangleValueByteOffsets),
              "common member offset mismatch Micromap/MicromapPacked");
static_assert(offsetof(Micromap, layout) == offsetof(MicromapPacked, layout),
              "common member offset mismatch Micromap/MicromapPacked");
static_assert(offsetof(Micromap, frequency) == offsetof(MicromapPacked, frequency),
              "common member offset mismatch Micromap/MicromapPacked");


static_assert(offsetof(Micromap, values) == offsetof(MicromapCompressed, values),
              "common member offset mismatch Micromap/MicromapCompressed");
static_assert(offsetof(Micromap, valueFloatExpansion) == offsetof(MicromapCompressed, valueFloatExpansion),
              "common member offset mismatch Micromap/MicromapCompressed");
static_assert(offsetof(Micromap, minSubdivLevel) == offsetof(MicromapCompressed, minSubdivLevel),
              "common member offset mismatch Micromap/MicromapCompressed");
static_assert(offsetof(Micromap, maxSubdivLevel) == offsetof(MicromapCompressed, maxSubdivLevel),
              "common member offset mismatch Micromap/MicromapCompressed");
static_assert(offsetof(Micromap, triangleSubdivLevels) == offsetof(MicromapCompressed, triangleSubdivLevels),
              "common member offset mismatch Micromap/MicromapCompressed");
static_assert(offsetof(Micromap, triangleValueIndexOffsets) == offsetof(MicromapCompressed, triangleValueByteOffsets),
              "common member offset mismatch Micromap/MicromapCompressed");

MICROMESH_API const char* MICROMESH_CALL micromeshResultGetName(Result result)
{
    switch(result)
    {
    case Result::eSuccess:
        return "eSuccess";
    case Result::eFailure:
        return "eFailure";
    case Result::eContinue:
        return "eContinue";
    case Result::eInvalidFrequency:
        return "eInvalidFrequency";
    case Result::eInvalidFormat:
        return "eInvalidFormat";
    case Result::eInvalidBlockFormat:
        return "eInvalidBlockFormat";
    case Result::eInvalidRange:
        return "eInvalidRange";
    case Result::eInvalidValue:
        return "eInvalidValue";
    case Result::eInvalidLayout:
        return "eInvalidLayout";
    case Result::eInvalidOperationOrder:
        return "eInvalidOperationOrder";
    case Result::eMismatchingInputEdgeValues:
        return "eMismatchingInputEdgeValues";
    case Result::eMismatchingOutputEdgeValues:
        return "eMismatchingOutputEdgeValues";
    case Result::eUnsupportedVersion:
        return "eUnsupportedVersion";
    case Result::eUnsupportedShaderCodeType:
        return "eUnsupportedShaderCodeType";
    }
    return "Unknown result";
}

MICROMESH_API Result MICROMESH_CALL micromeshFormatGetInfo(Format format, FormatInfo* info)
{
    /*
for typestr in types:gmatch("[%w_]+") do
    local pre, channelType = typestr:match("(%w+)_(%w+)")
    local channelBits  = tonumber(pre:match("(%d+)"))
    local channelCount = tonumber(string.len(pre:gsub("(%d+)", "")))

    local str = "case Format::"..typestr..": *info = {ChannelType::"..channelType..","..channelBits..","..channelCount..","..(channelCount * (channelBits/8))..", STATE_FALSE, 1}; return Result::eSuccess;"

    print(str)
end
*/

    switch(format)
    {
    case Format::eR8_unorm:
        *info = {ChannelType::eUnorm, 8, 1, 1, false, 1};
        return Result::eSuccess;
    case Format::eR8_snorm:
        *info = {ChannelType::eSnorm, 8, 1, 1, false, 1};
        return Result::eSuccess;
    case Format::eR8_uint:
        *info = {ChannelType::eUint, 8, 1, 1, false, 1};
        return Result::eSuccess;
    case Format::eR8_sint:
        *info = {ChannelType::eSint, 8, 1, 1, false, 1};
        return Result::eSuccess;
    case Format::eRG8_unorm:
        *info = {ChannelType::eUnorm, 8, 2, 2, false, 1};
        return Result::eSuccess;
    case Format::eRG8_snorm:
        *info = {ChannelType::eSnorm, 8, 2, 2, false, 1};
        return Result::eSuccess;
    case Format::eRG8_uint:
        *info = {ChannelType::eUint, 8, 2, 2, false, 1};
        return Result::eSuccess;
    case Format::eRG8_sint:
        *info = {ChannelType::eSint, 8, 2, 2, false, 1};
        return Result::eSuccess;
    case Format::eRGB8_unorm:
        *info = {ChannelType::eUnorm, 8, 3, 3, false, 1};
        return Result::eSuccess;
    case Format::eRGB8_snorm:
        *info = {ChannelType::eSnorm, 8, 3, 3, false, 1};
        return Result::eSuccess;
    case Format::eRGB8_uint:
        *info = {ChannelType::eUint, 8, 3, 3, false, 1};
        return Result::eSuccess;
    case Format::eRGB8_sint:
        *info = {ChannelType::eSint, 8, 3, 3, false, 1};
        return Result::eSuccess;
    case Format::eRGBA8_unorm:
        *info = {ChannelType::eUnorm, 8, 4, 4, false, 1};
        return Result::eSuccess;
    case Format::eRGBA8_snorm:
        *info = {ChannelType::eSnorm, 8, 4, 4, false, 1};
        return Result::eSuccess;
    case Format::eRGBA8_uint:
        *info = {ChannelType::eUint, 8, 4, 4, false, 1};
        return Result::eSuccess;
    case Format::eRGBA8_sint:
        *info = {ChannelType::eSint, 8, 4, 4, false, 1};
        return Result::eSuccess;
    case Format::eR16_unorm:
        *info = {ChannelType::eUnorm, 16, 1, 2, false, 1};
        return Result::eSuccess;
    case Format::eR16_snorm:
        *info = {ChannelType::eSnorm, 16, 1, 2, false, 1};
        return Result::eSuccess;
    case Format::eR16_uint:
        *info = {ChannelType::eUint, 16, 1, 2, false, 1};
        return Result::eSuccess;
    case Format::eR16_sint:
        *info = {ChannelType::eSint, 16, 1, 2, false, 1};
        return Result::eSuccess;
    case Format::eR16_sfloat:
        *info = {ChannelType::eSfloat, 16, 1, 2, false, 1};
        return Result::eSuccess;
    case Format::eRG16_unorm:
        *info = {ChannelType::eUnorm, 16, 2, 4, false, 1};
        return Result::eSuccess;
    case Format::eRG16_snorm:
        *info = {ChannelType::eSnorm, 16, 2, 4, false, 1};
        return Result::eSuccess;
    case Format::eRG16_uint:
        *info = {ChannelType::eUint, 16, 2, 4, false, 1};
        return Result::eSuccess;
    case Format::eRG16_sint:
        *info = {ChannelType::eSint, 16, 2, 4, false, 1};
        return Result::eSuccess;
    case Format::eRG16_sfloat:
        *info = {ChannelType::eSfloat, 16, 2, 4, false, 1};
        return Result::eSuccess;
    case Format::eRGB16_unorm:
        *info = {ChannelType::eUnorm, 16, 3, 6, false, 1};
        return Result::eSuccess;
    case Format::eRGB16_snorm:
        *info = {ChannelType::eSnorm, 16, 3, 6, false, 1};
        return Result::eSuccess;
    case Format::eRGB16_uint:
        *info = {ChannelType::eUint, 16, 3, 6, false, 1};
        return Result::eSuccess;
    case Format::eRGB16_sint:
        *info = {ChannelType::eSint, 16, 3, 6, false, 1};
        return Result::eSuccess;
    case Format::eRGB16_sfloat:
        *info = {ChannelType::eSfloat, 16, 3, 6, false, 1};
        return Result::eSuccess;
    case Format::eRGBA16_unorm:
        *info = {ChannelType::eUnorm, 16, 4, 8, false, 1};
        return Result::eSuccess;
    case Format::eRGBA16_snorm:
        *info = {ChannelType::eSnorm, 16, 4, 8, false, 1};
        return Result::eSuccess;
    case Format::eRGBA16_uint:
        *info = {ChannelType::eUint, 16, 4, 8, false, 1};
        return Result::eSuccess;
    case Format::eRGBA16_sint:
        *info = {ChannelType::eSint, 16, 4, 8, false, 1};
        return Result::eSuccess;
    case Format::eRGBA16_sfloat:
        *info = {ChannelType::eSfloat, 16, 4, 8, false, 1};
        return Result::eSuccess;
    case Format::eR32_uint:
        *info = {ChannelType::eUint, 32, 1, 4, false, 1};
        return Result::eSuccess;
    case Format::eR32_sint:
        *info = {ChannelType::eSint, 32, 1, 4, false, 1};
        return Result::eSuccess;
    case Format::eR32_sfloat:
        *info = {ChannelType::eSfloat, 32, 1, 4, false, 1};
        return Result::eSuccess;
    case Format::eRG32_uint:
        *info = {ChannelType::eUint, 32, 2, 8, false, 1};
        return Result::eSuccess;
    case Format::eRG32_sint:
        *info = {ChannelType::eSint, 32, 2, 8, false, 1};
        return Result::eSuccess;
    case Format::eRG32_sfloat:
        *info = {ChannelType::eSfloat, 32, 2, 8, false, 1};
        return Result::eSuccess;
    case Format::eRGB32_uint:
        *info = {ChannelType::eUint, 32, 3, 12, false, 1};
        return Result::eSuccess;
    case Format::eRGB32_sint:
        *info = {ChannelType::eSint, 32, 3, 12, false, 1};
        return Result::eSuccess;
    case Format::eRGB32_sfloat:
        *info = {ChannelType::eSfloat, 32, 3, 12, false, 1};
        return Result::eSuccess;
    case Format::eRGBA32_uint:
        *info = {ChannelType::eUint, 32, 4, 16, false, 1};
        return Result::eSuccess;
    case Format::eRGBA32_sint:
        *info = {ChannelType::eSint, 32, 4, 16, false, 1};
        return Result::eSuccess;
    case Format::eRGBA32_sfloat:
        *info = {ChannelType::eSfloat, 32, 4, 16, false, 1};
        return Result::eSuccess;
    case Format::eR64_uint:
        *info = {ChannelType::eUint, 64, 1, 8, false, 1};
        return Result::eSuccess;
    case Format::eR64_sint:
        *info = {ChannelType::eSint, 64, 1, 8, false, 1};
        return Result::eSuccess;
    case Format::eR64_sfloat:
        *info = {ChannelType::eSfloat, 64, 1, 8, false, 1};
        return Result::eSuccess;
    case Format::eRG64_uint:
        *info = {ChannelType::eUint, 64, 2, 16, false, 1};
        return Result::eSuccess;
    case Format::eRG64_sint:
        *info = {ChannelType::eSint, 64, 2, 16, false, 1};
        return Result::eSuccess;
    case Format::eRG64_sfloat:
        *info = {ChannelType::eSfloat, 64, 2, 16, false, 1};
        return Result::eSuccess;
    case Format::eRGB64_uint:
        *info = {ChannelType::eUint, 64, 3, 24, false, 1};
        return Result::eSuccess;
    case Format::eRGB64_sint:
        *info = {ChannelType::eSint, 64, 3, 24, false, 1};
        return Result::eSuccess;
    case Format::eRGB64_sfloat:
        *info = {ChannelType::eSfloat, 64, 3, 24, false, 1};
        return Result::eSuccess;
    case Format::eRGBA64_uint:
        *info = {ChannelType::eUint, 64, 4, 32, false, 1};
        return Result::eSuccess;
    case Format::eRGBA64_sint:
        *info = {ChannelType::eSint, 64, 4, 32, false, 1};
        return Result::eSuccess;
    case Format::eRGBA64_sfloat:
        *info = {ChannelType::eSfloat, 64, 4, 32, false, 1};
        return Result::eSuccess;
    case Format::eOpaC1_rx_uint_block:
        *info = {ChannelType::eUint, 0, 1, 1, true, 0};
        return Result::eSuccess;
    case Format::eDispC1_r11_unorm_block:
        // base compressed format must use byteStride 1
        *info = {ChannelType::eUnorm, 11, 1, 1, true, 0};
        return Result::eSuccess;
    case Format::eR11_unorm_pack16:
        *info = {ChannelType::eUnorm, 11, 1, 2, false, 1};
        return Result::eSuccess;
    case Format::eR11_unorm_packed_align32:
        *info = {ChannelType::eUnorm, 11, 1, 1, true, 0};
        return Result::eSuccess;
    default:
        return Result::eInvalidFormat;
    }
}

MICROMESH_API MicromapType MICROMESH_CALL micromeshFormatGetMicromapType(Format format)
{
    switch(format)
    {
    case Format::eUndefined:
        return MicromapType::eInvalid;
    case Format::eDispC1_r11_unorm_block:
    case Format::eOpaC1_rx_uint_block:
        return MicromapType::eCompressed;
    case Format::eR11_unorm_packed_align32:
        return MicromapType::ePacked;
    default:
        return MicromapType::eUncompressed;
    }
}

MICROMESH_API Format MICROMESH_CALL micromeshFormatGetMinMaxFormat(Format format)
{
    switch(format)
    {
    case Format::eDispC1_r11_unorm_block:
        return Format::eR8_uint;
    case Format::eOpaC1_rx_uint_block:
        return Format::eR11_unorm_pack16;
    case Format::eR11_unorm_packed_align32:
        return Format::eR11_unorm_pack16;
    default:
        return format;
    }
}

MICROMESH_API Frequency MICROMESH_CALL micromeshFormatGetCompressedFrequency(Format format)
{
    switch(format)
    {
    case Format::eDispC1_r11_unorm_block:
        return Frequency::ePerMicroVertex;
    case Format::eOpaC1_rx_uint_block:
        return Frequency::ePerMicroTriangle;
    default:
        return Frequency::ePerMicroVertex;
    }
}

MICROMESH_API StandardLayoutType MICROMESH_CALL micromeshFormatGetCompressedStandardLayout(Format format)
{
    switch(format)
    {
    case Format::eDispC1_r11_unorm_block:
        return StandardLayoutType::eBirdCurve;
    case Format::eOpaC1_rx_uint_block:
        return StandardLayoutType::eBirdCurve;
    default:
        return StandardLayoutType::eUnknown;
    }
}

MICROMESH_API Result MICROMESH_CALL micromeshBlockFormatDispC1GetInfo(BlockFormatDispC1 format, FormatInfo* info)
{
    switch(format)
    {
    case BlockFormatDispC1::eR11_unorm_lvl3_pack512:
        *info = {ChannelType::eUnorm, 11, 1, 512 / 8, true, 45};
        return Result::eSuccess;
    case BlockFormatDispC1::eR11_unorm_lvl4_pack1024:
        *info = {ChannelType::eUnorm, 11, 1, 1024 / 8, true, 153};
        return Result::eSuccess;
    case BlockFormatDispC1::eR11_unorm_lvl5_pack1024:
        *info = {ChannelType::eUnorm, 11, 1, 1024 / 8, true, 561};
        return Result::eSuccess;
    default:
        return Result::eInvalidFormat;
    }
}

MICROMESH_API Result MICROMESH_CALL micromeshBlockFormatOpaC1GetInfo(BlockFormatOpaC1 format, FormatInfo* info)
{
    switch(format)
    {
    case BlockFormatOpaC1::eR1_uint_x8:
        *info = {ChannelType::eUint, 1, 1, 1, true, 8};
        return Result::eSuccess;
    case BlockFormatOpaC1::eR2_uint_x4:
        *info = {ChannelType::eUint, 2, 1, 1, true, 4};
        return Result::eSuccess;
    default:
        return Result::eInvalidFormat;
    }
}

MICROMESH_API const char* MICROMESH_CALL micromeshGetFormatString(Format format)
{
    switch(format)
    {
    case Format::eUndefined:
        return "eUndefined";
    case Format::eR8_unorm:
        return "eR8_unorm";
    case Format::eR8_snorm:
        return "eR8_snorm";
    case Format::eR8_uint:
        return "eR8_uint";
    case Format::eR8_sint:
        return "eR8_sint";
    case Format::eRG8_unorm:
        return "eRG8_unorm";
    case Format::eRG8_snorm:
        return "eRG8_snorm";
    case Format::eRG8_uint:
        return "eRG8_uint";
    case Format::eRG8_sint:
        return "eRG8_sint";
    case Format::eRGB8_unorm:
        return "eRGB8_unorm";
    case Format::eRGB8_snorm:
        return "eRGB8_snorm";
    case Format::eRGB8_uint:
        return "eRGB8_uint";
    case Format::eRGB8_sint:
        return "eRGB8_sint";
    case Format::eRGBA8_unorm:
        return "eRGBA8_unorm";
    case Format::eRGBA8_snorm:
        return "eRGBA8_snorm";
    case Format::eRGBA8_uint:
        return "eRGBA8_uint";
    case Format::eRGBA8_sint:
        return "eRGBA8_sint";
    case Format::eR16_unorm:
        return "eR16_unorm";
    case Format::eR16_snorm:
        return "eR16_snorm";
    case Format::eR16_uint:
        return "eR16_uint";
    case Format::eR16_sint:
        return "eR16_sint";
    case Format::eR16_sfloat:
        return "eR16_sfloat";
    case Format::eRG16_unorm:
        return "eRG16_unorm";
    case Format::eRG16_snorm:
        return "eRG16_snorm";
    case Format::eRG16_uint:
        return "eRG16_uint";
    case Format::eRG16_sint:
        return "eRG16_sint";
    case Format::eRG16_sfloat:
        return "eRG16_sfloat";
    case Format::eRGB16_unorm:
        return "eRGB16_unorm";
    case Format::eRGB16_snorm:
        return "eRGB16_snorm";
    case Format::eRGB16_uint:
        return "eRGB16_uint";
    case Format::eRGB16_sint:
        return "eRGB16_sint";
    case Format::eRGB16_sfloat:
        return "eRGB16_sfloat";
    case Format::eRGBA16_unorm:
        return "eRGBA16_unorm";
    case Format::eRGBA16_snorm:
        return "eRGBA16_snorm";
    case Format::eRGBA16_uint:
        return "eRGBA16_uint";
    case Format::eRGBA16_sint:
        return "eRGBA16_sint";
    case Format::eRGBA16_sfloat:
        return "eRGBA16_sfloat";
    case Format::eR32_uint:
        return "eR32_uint";
    case Format::eR32_sint:
        return "eR32_sint";
    case Format::eR32_sfloat:
        return "eR32_sfloat";
    case Format::eRG32_uint:
        return "eRG32_uint";
    case Format::eRG32_sint:
        return "eRG32_sint";
    case Format::eRG32_sfloat:
        return "eRG32_sfloat";
    case Format::eRGB32_uint:
        return "eRGB32_uint";
    case Format::eRGB32_sint:
        return "eRGB32_sint";
    case Format::eRGB32_sfloat:
        return "eRGB32_sfloat";
    case Format::eRGBA32_uint:
        return "eRGBA32_uint";
    case Format::eRGBA32_sint:
        return "eRGBA32_sint";
    case Format::eRGBA32_sfloat:
        return "eRGBA32_sfloat";
    case Format::eOpaC1_rx_uint_block:
        return "eOpaC1_rx_uint_block";
    case Format::eDispC1_r11_unorm_block:
        return "eDispC1_r11_unorm_block";
    case Format::eR11_unorm_pack16:
        return "eR11_unorm_pack16";
    case Format::eR11_unorm_packed_align32:
        return "eR11_unorm_packed_align32";
    default:
        return "Invalid";
    }
}

MICROMESH_API const char* MICROMESH_CALL micromeshBlockFormatDispC1GetString(BlockFormatDispC1 format)
{
    switch(format)
    {
    case BlockFormatDispC1::eInvalid:
        return "eInvalid";
    case BlockFormatDispC1::eR11_unorm_lvl3_pack512:
        return "eR11_unorm_lvl3_pack512";
    case BlockFormatDispC1::eR11_unorm_lvl4_pack1024:
        return "eR11_unorm_lvl4_pack1024";
    case BlockFormatDispC1::eR11_unorm_lvl5_pack1024:
        return "eR11_unorm_lvl5_pack1024";
    default:
        return "Invalid";
    }
}

MICROMESH_API const char* MICROMESH_CALL micromeshBlockFormatOpaC1GetString(BlockFormatOpaC1 format)
{
    switch(format)
    {
    case BlockFormatOpaC1::eInvalid:
        return "eInvalid";
    case BlockFormatOpaC1::eR1_uint_x8:
        return "eR1_uint8";
    case BlockFormatOpaC1::eR2_uint_x4:
        return "eR2_uint4";
    default:
        return "Invalid";
    }
}

//////////////////////////////////////////////////////////////////////////

MICROMESH_API BaryUV_uint16 MICROMESH_CALL micromeshUVGetSwizzled(BaryUV_uint16 uv, uint32_t subdivLevel, TriangleSwizzle swizzle)
{
    BaryWUV_uint16 oldWUV = baryUVtoWUV_uint(uv, subdivLevel);
    BaryWUV_uint16 newWUV = oldWUV;

    if(swizzle.shift == 1)
    {
        newWUV = {oldWUV.u, oldWUV.v, oldWUV.w};
    }
    else if(swizzle.shift == 2)
    {
        newWUV = {oldWUV.v, oldWUV.w, oldWUV.u};
    }

    if(swizzle.flipVertexUV)
    {
        uint16_t tmp = newWUV.v;
        newWUV.v     = newWUV.u;
        newWUV.u     = tmp;
    }

    return {newWUV.u, newWUV.v};
}

//////////////////////////////////////////////////////////////////////////

MICROMESH_API BaryUV_uint16 MICROMESH_CALL micromeshUVGetEdgeDecimated(BaryUV_uint16 coord, uint32_t subdivLevel, uint32_t edgeDecimationFlag)
{
    uint32_t baryMax = 1 << subdivLevel;
    uint32_t coord_w = baryMax - coord.u - coord.v;

    if(subdivLevel == 0 || edgeDecimationFlag == 0)
        return coord;

    if(edgeDecimationFlag & 1 && coord.v == 0)
    {
        if(coord_w < baryMax / 2)
            return makeBaryUV_uint16((coord.u + 1) & ~1, 0);
        else
            return makeBaryUV_uint16((coord.u) & ~1, 0);
    }
    if(edgeDecimationFlag & 2 && coord_w == 0)
    {
        if(coord.u < baryMax / 2)
            return makeBaryUV_uint16((coord.u) & ~1, (coord.v + 1) & ~1);
        else
            return makeBaryUV_uint16((coord.u + 1) & ~1, (coord.v) & ~1);
    }
    if(edgeDecimationFlag & 4 && coord.u == 0)
    {
        if(coord.v < baryMax / 2)
            return makeBaryUV_uint16(0, (coord.v) & ~1);
        else
            return makeBaryUV_uint16(0, (coord.v + 1) & ~1);
    }
    return coord;
}

//////////////////////////////////////////////////////////////////////////

static inline Vector_uint32_3 processTriangle(const MicromapLayout* layout, BaryUV_uint16 a, BaryUV_uint16 b, BaryUV_uint16 c, uint32_t subivLevel, uint32_t edgeFlag)
{
    if(edgeFlag)
    {
        a = micromeshUVGetEdgeDecimated(a, subivLevel, edgeFlag);
        b = micromeshUVGetEdgeDecimated(b, subivLevel, edgeFlag);
        c = micromeshUVGetEdgeDecimated(c, subivLevel, edgeFlag);
    }

    Vector_uint32_3 indices;
    indices.x = layout->pfnGetMicroVertexIndex(a.u, a.v, subivLevel, layout->userData);
    indices.y = layout->pfnGetMicroVertexIndex(b.u, b.v, subivLevel, layout->userData);
    indices.z = layout->pfnGetMicroVertexIndex(c.u, c.v, subivLevel, layout->userData);

    return indices;
}

MICROMESH_API Result MICROMESH_CALL micromeshLayoutBuildUVMesh(const MicromapLayout* layout,
                                                               ArrayInfo_uint16_2*   uvVertices,
                                                               ArrayInfo_uint32_3*   triangleIndices,
                                                               uint32_t              subdivLevel,
                                                               uint32_t              edgeFlag)
{
    uint32_t numVertices  = subdivLevelGetVertexCount(subdivLevel);
    uint32_t numTriangles = subdivLevelGetTriangleCount(subdivLevel);

    if(uvVertices->count != numVertices || triangleIndices->count != numTriangles)
        return Result::eInvalidRange;

    uint32_t numSegmentsPerEdge = subdivLevelGetSegmentCount(subdivLevel);
    uint32_t numVtxPerEdge      = subdivLevelGetSegmentCount(subdivLevel) + 1;

    for(uint32_t u = 0; u < numVtxPerEdge; u++)
    {
        for(uint32_t v = 0; v < numVtxPerEdge - u; v++)
        {
            uint32_t  idx       = layout->pfnGetMicroVertexIndex(u, v, subdivLevel, layout->userData);
            uint16_t* vertexUVs = arrayGet<uint16_t>(*uvVertices, idx);
            vertexUVs[0]        = uint16_t(u);
            vertexUVs[1]        = uint16_t(v);
        }
    }

    for(uint32_t u = 0; u < numSegmentsPerEdge; u++)
    {
        for(uint32_t v = 0; v < numSegmentsPerEdge - u; v++)
        {
            {
                uint32_t        idx = layout->pfnGetMicroTriangleIndex(u, v, 0, subdivLevel, layout->userData);
                Vector_uint32_3 tri = processTriangle(layout, makeBaryUV_uint16(u, v), makeBaryUV_uint16(u + 1u, v),
                                                      makeBaryUV_uint16(u, v + 1u), subdivLevel, edgeFlag);
                uint32_t*       triIndices = arrayGet<uint32_t>(*triangleIndices, idx);
                triIndices[0]              = tri.x;
                triIndices[1]              = tri.y;
                triIndices[2]              = tri.z;
            }
            if(v != numSegmentsPerEdge - u - 1)
            {
                uint32_t idx = layout->pfnGetMicroTriangleIndex(u, v, 1, subdivLevel, layout->userData);
                // warning the order here was tuned for bird-curve, horizontal edge first, in theory need a different way of doing this
                Vector_uint32_3 tri = processTriangle(layout, makeBaryUV_uint16(u + 1u, v + 1u), makeBaryUV_uint16(u, v + 1u),
                                                      makeBaryUV_uint16(u + 1u, v), subdivLevel, edgeFlag);
                uint32_t*       triIndices = arrayGet<uint32_t>(*triangleIndices, idx);
                triIndices[0]              = tri.x;
                triIndices[1]              = tri.y;
                triIndices[2]              = tri.z;
            }
        }
    }

    return Result::eSuccess;
}


// Interleave even bits from x with odd bits from y
static inline uint32_t bird_interleaveBits(uint32_t x, uint32_t y)
{
    x = (x | (x << 8)) & 0x00ff00ff;
    x = (x | (x << 4)) & 0x0f0f0f0f;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;

    y = (y | (y << 8)) & 0x00ff00ff;
    y = (y | (y << 4)) & 0x0f0f0f0f;
    y = (y | (y << 2)) & 0x33333333;
    y = (y | (y << 1)) & 0x55555555;

    return x | (y << 1);
}

// Calculate exclusive prefix or (log(n) XOR's and SHF's)
static inline uint32_t bird_prefixEor(uint32_t x)
{
    x ^= x >> 1;
    x ^= x >> 2;
    x ^= x >> 4;
    x ^= x >> 8;

    return x;
}

// Compute 2 16-bit prefix XORs in a 32-bit register
static inline uint32_t bird_prefixEor2(uint32_t x)
{
    x ^= (x >> 1) & 0x7fff7fff;
    x ^= (x >> 2) & 0x3fff3fff;
    x ^= (x >> 4) & 0x0fff0fff;
    x ^= (x >> 8) & 0x00ff00ff;

    return x;
}

// Interleave 16 even bits from x with 16 odd bits from y
static inline uint32_t bird_interleaveBits2(uint32_t x, uint32_t y)
{
    x = (x & 0xffff) | (y << 16);
    x = ((x >> 8) & 0x0000ff00) | ((x << 8) & 0x00ff0000) | (x & 0xff0000ff);
    x = ((x >> 4) & 0x00f000f0) | ((x << 4) & 0x0f000f00) | (x & 0xf00ff00f);
    x = ((x >> 2) & 0x0c0c0c0c) | ((x << 2) & 0x30303030) | (x & 0xc3c3c3c3);
    x = ((x >> 1) & 0x22222222) | ((x << 1) & 0x44444444) | (x & 0x99999999);

    return x;
}

// Compute index of a single triplet of compression coefficients from triangle's barycentric coordinates
// Assumes u, v and w have only 16 valid bits in the lsbs (good for subdivision depths up to 64K segments per edge)
// Triplets are ordered along the bird curve
static inline uint32_t bird_getTripletIndex(uint32_t u, uint32_t v, uint32_t w, uint32_t level)
{
    const uint32_t coordMask = ((1U << level) - 1);

    uint32_t b0 = ~(u ^ w) & coordMask;
    uint32_t t  = (u ^ v) & b0;  //  (equiv: (~u & v & ~w) | (u & ~v & w))
    uint32_t c  = (((u & v & w) | (~u & ~v & ~w)) & coordMask) << 16;
    uint32_t f  = bird_prefixEor2(t | c) ^ u;
    uint32_t b1 = (f & ~b0) | t;  // equiv: (~u & v & ~w) | (u & ~v & w) | (f0 & u & ~w) | (f0 & ~u & w))

    uint32_t dist = bird_interleaveBits2(b0, b1);  // 13 instructions

    // Adjust computed distance accounting for "skipped" triangles on the bird curve

    f >>= 16;
    b0 <<= 1;
    return (dist + (b0 & ~f) - (b0 & f)) >> 3;
}

static inline uint32_t bird_getVertexLevel(BaryWUV_uint16 coord, uint32_t subdivLevel)
{
    uint32_t maxCoord = 1 << subdivLevel;

    if(coord.w == maxCoord || coord.u == maxCoord || coord.v == maxCoord)
    {
        return 0;
    }

    uint32_t shift    = 0;
    uint32_t minCoord = coord.w | coord.u | coord.v;
    for(shift = 0; shift < subdivLevel; shift++)
    {
        if(minCoord & (1 << shift))
        {
            break;
        }
    }

    return subdivLevel - shift;
}

static inline uint32_t bird_getVertexLevelCoordIndex(BaryWUV_uint16 coord, uint32_t subdivLevel)
{
    if(subdivLevel == 0)
    {
        if(coord.w)
        {
            return 0;
        }
        else if(coord.u)
        {
            return 1;
        }
        else
        {
            return 2;
        }
    }

    // we need to descend appropriately until subdivLevel is reached
    BaryWUV_uint16 quadref = makeBaryWUV_uint16(coord.w & ~1, coord.u & ~1, coord.v & ~1);
    BaryWUV_uint16 rest    = makeBaryWUV_uint16(coord.w & 1, coord.u & 1, coord.v & 1);
    // edge 0 = AC split
    // edge 1 = CB split
    // edge 2 = BA split
    uint32_t edge  = rest.u == 0 ? 0 : ((rest.v == 1) ? 1 : 2);
    uint32_t index = bird_getTripletIndex(quadref.u, quadref.v, quadref.w, subdivLevel) * 3;
    index += edge;

    return index;
}

static uint32_t bird_getMicroVertexIndex(uint32_t u, uint32_t v, uint32_t subdivLevel, void* user)
{
    BaryWUV_uint16 coord = baryUVtoWUV_uint(makeBaryUV_uint16(u, v), subdivLevel);

    // find out on which subdiv level our vertex sits on
    uint32_t level = bird_getVertexLevel(coord, subdivLevel);
    // adjust coord into level
    BaryWUV_uint16 base = makeBaryWUV_uint16(coord.w >> (subdivLevel - level), coord.u >> (subdivLevel - level),
                                             coord.v >> (subdivLevel - level));
    // get index relative within level
    uint32_t index = bird_getVertexLevelCoordIndex(base, level);
    if(level)
    {
        // append previous levels' vertices
        index += subdivLevelGetVertexCount(level - 1);
    }
    return index;
}

static uint32_t bird_getMicroTriangleIndex(uint32_t u, uint32_t v, uint32_t isUpperTriangle, uint32_t subdivLevel, void* user)
{
    // uvw.vw map to uv here
    uint32_t iu, iv, iw;

    iu = u;
    iv = v;
    iw = ~(iu + iv);
    if(isUpperTriangle)
        --iw;

    uint32_t b0 = ~(iu ^ iw);
    uint32_t t  = (iu ^ iv) & b0;
    uint32_t f  = bird_prefixEor(t);
    uint32_t b1 = ((f ^ iu) & ~b0) | t;

    return bird_interleaveBits(b0, b1);
}

MICROMESH_API uint32_t MICROMESH_CALL micromeshBirdUVToLinearTriangle(uint32_t u, uint32_t v, uint32_t isUpperTriangle, uint32_t subdivLevel)
{
    return bird_getMicroTriangleIndex(u, v, isUpperTriangle, subdivLevel, nullptr);
}

MICROMESH_API uint32_t MICROMESH_CALL micromeshBirdUVToLinearVertex(uint32_t u, uint32_t v, uint32_t subdivLevel)
{
    return bird_getMicroVertexIndex(u, v, subdivLevel, nullptr);
}

// mapping from uv-coordinate of the barycentric micromesh grid to the value storage index
static uint32_t umajor_getMicroVertexIndex(uint32_t u, uint32_t v, uint32_t subdivLevel, void* user)
{
    uint32_t vtxPerEdge = (1 << subdivLevel) + 1;
    uint32_t x          = v;
    uint32_t y          = u;
    uint32_t trinum     = (y * (y + 1)) / 2;
    return y * (vtxPerEdge + 1) - trinum + x;
}

static uint32_t umajor_getMicroTriangleIndex(uint32_t u, uint32_t v, uint32_t isUpperTriangle, uint32_t subdivLevel, void* user)
{
    uint32_t triPerEdge = (1 << subdivLevel) * 2;
    uint32_t x          = v;
    uint32_t y          = u;
    uint32_t trinum     = y * y;
    return y * (triPerEdge)-trinum + x * 2 + (isUpperTriangle ? 1 : 0);
}

// setup stanard layouts
MICROMESH_API Result MICROMESH_CALL micromeshLayoutInitStandard(MicromapLayout* layout, StandardLayoutType standard)
{
    switch(standard)
    {
    case StandardLayoutType::eUmajor:
        layout->userData                 = nullptr;
        layout->pfnGetMicroVertexIndex   = umajor_getMicroVertexIndex;
        layout->pfnGetMicroTriangleIndex = umajor_getMicroTriangleIndex;
        return Result::eSuccess;
    case StandardLayoutType::eBirdCurve:
        layout->userData                 = nullptr;
        layout->pfnGetMicroVertexIndex   = bird_getMicroVertexIndex;
        layout->pfnGetMicroTriangleIndex = bird_getMicroTriangleIndex;
        return Result::eSuccess;
    default:
        return Result::eInvalidValue;
    }
}

// get type if applicable
MICROMESH_API StandardLayoutType MICROMESH_CALL micromeshLayoutGetStandardType(const MicromapLayout* layout)
{
    MicromapLayout test;
    micromeshLayoutInitStandard(&test, StandardLayoutType::eUmajor);
    if(micromapLayoutIsEqual(test, *layout))
        return StandardLayoutType::eUmajor;

    micromeshLayoutInitStandard(&test, StandardLayoutType::eBirdCurve);
    if(micromapLayoutIsEqual(test, *layout))
        return StandardLayoutType::eBirdCurve;

    return StandardLayoutType::eUnknown;
}

MICROMESH_API Result MICROMESH_CALL micromeshMicromapCompressedSetupValues(MicromapCompressed* micromap,
                                                                           bool                computeValueByteOffsets,
                                                                           const MessageCallbackInfo* messageCallbackInfo)
{
    uint32_t count      = 0;
    Format   format     = micromap->values.format;
    uint32_t byteStride = micromap->values.byteStride;

    if(format != Format::eDispC1_r11_unorm_block && format != Format::eOpaC1_rx_uint_block && format != Format::eR11_unorm_packed_align32)
    {
        MLOGE(messageCallbackInfo,
              "micromap->values.format (%s) must be one of eDispC1_r11_unorm_block, eOpaC1_rx_uint_block, or "
              "eR11_unorm_packed_align32.",
              micromeshGetFormatString(format));
        return Result::eInvalidFormat;
    }

    if(byteStride != 1)
    {
        MLOGE(messageCallbackInfo, "micromap->values.byteStride (%u) must be 1.", byteStride);
        return Result::eInvalidValue;
    }

    for(uint64_t i = 0; i < micromap->triangleSubdivLevels.count; i++)
    {
        uint32_t subdivLevel = arrayGetV<uint16_t>(micromap->triangleSubdivLevels, i);
        uint32_t blockFormat = arrayGetV<uint16_t>(micromap->triangleBlockFormats, i);

        uint64_t offset = count;

        if(format == Format::eOpaC1_rx_uint_block)
        {
            uint32_t blockCount = BlockFormatOpaC1(blockFormat) == BlockFormatOpaC1::eR1_uint_x8 ? 8 : 4;

            uint32_t valueCount = subdivLevelGetTriangleCount(subdivLevel);
            count += (valueCount + blockCount - 1) / blockCount;
        }
        else
        {
            uint32_t blockSubdiv = blockFormat + 3 - 1;
            uint32_t valueCount  = blockFormat >= uint32_t(BlockFormatDispC1::eR11_unorm_lvl4_pack1024) ? 128 : 64;
            if(valueCount == 128)
            {
                // alignment
                uint32_t alignMask = valueCount - 1;
                count              = (count + alignMask) & ~(alignMask);
                offset             = count;
            }
            count += subdivLevelGetTriangleCount((subdivLevel < 3 ? 3 : subdivLevel) - blockSubdiv) * valueCount;
        }

        if(computeValueByteOffsets)
        {
            if(offset > std::numeric_limits<uint32_t>::max())
            {
                MLOGE(messageCallbackInfo,
                      "This compressed micromap would require more than 2^32-1 bytes of decompressed values. "
                      "Specifically, the byte offset (%zu) required for micromap triangle %zu would not fit within a "
                      "32-bit element of micromap->triangleValueByteOffsets.",
                      offset, i);
                return Result::eInvalidValue;
            }
            arraySetV<uint32_t>(micromap->triangleValueByteOffsets, i, uint32_t(offset));
        }
    }

    micromap->values.count = count;

    return Result::eSuccess;
}

MICROMESH_API Result MICROMESH_CALL micromeshMicromapPackedSetupValues(MicromapPacked* micromap, bool computeTriangleValueByteOffsets)
{
    uint32_t count      = 0;
    Format   format     = micromap->values.format;
    uint32_t byteStride = micromap->values.byteStride;

    // don't handle compressed
    if(format != Format::eR11_unorm_packed_align32)
        return Result::eInvalidFormat;

    // check that proper byteStride is used for this special format
    if(format == Format::eR11_unorm_packed_align32 && byteStride != 1)
        return Result::eInvalidValue;

    for(uint64_t i = 0; i < micromap->triangleSubdivLevels.count; i++)
    {
        if(computeTriangleValueByteOffsets)
        {
            arraySetV<uint32_t>(micromap->triangleValueByteOffsets, i, count);
        }

        uint32_t valueCount = subdivLevelGetCount(arrayGetV<uint16_t>(micromap->triangleSubdivLevels, i), micromap->frequency);
        count += packedCountBytesR11UnormPackedAlign32(valueCount);
    }

    micromap->values.count = count;

    return Result::eSuccess;
}

MICROMESH_API Result MICROMESH_CALL micromeshMicromapSetupValues(Micromap* micromap, bool computeTriangleValueIndexOffsets)
{
    uint32_t count      = 0;
    Format   format     = micromap->values.format;
    uint32_t byteStride = micromap->values.byteStride;

    // don't handle compressed / packed
    if(format == Format::eDispC1_r11_unorm_block || format == Format::eOpaC1_rx_uint_block || format == Format::eR11_unorm_packed_align32)
        return Result::eInvalidFormat;

    for(uint64_t i = 0; i < micromap->triangleSubdivLevels.count; i++)
    {
        if(computeTriangleValueIndexOffsets)
        {
            arraySetV<uint32_t>(micromap->triangleValueIndexOffsets, i, count);
        }

        uint32_t valueCount = subdivLevelGetCount(arrayGetV<uint16_t>(micromap->triangleSubdivLevels, i), micromap->frequency);
        count += valueCount;
    }

    micromap->values.count = count;

    return Result::eSuccess;
}

MICROMESH_API ArrayInfo MICROMESH_CALL micromeshMicromapGetTriangleArray(const Micromap* micromap, uint32_t triangleIndex)
{
    ArrayInfo info;
    assert(triangleIndex < micromap->triangleSubdivLevels.count);

    info = micromap->values;
    // shift pointer and setup count
    info.data = reinterpret_cast<uint8_t*>(info.data)
                + info.byteStride * arrayGetV<uint32_t>(micromap->triangleValueIndexOffsets, triangleIndex);
    info.count = subdivLevelGetCount(arrayGetV<uint16_t>(micromap->triangleSubdivLevels, triangleIndex), micromap->frequency);

    return info;
}

template <class Top>
static inline void doSubTriangleExtraction(const MicromapLayout*        layout,
                                           Frequency                    frequency,
                                           const ArrayInfo*             input,
                                           const SubTriangleExtraction* extract,
                                           uint32_t                     baseSubdiv,
                                           uint32_t                     triangleOffset,
                                           ArrayInfo*                   output)
{
    uint32_t subSubdiv  = extract->subSubdiv;
    uint32_t valueCount = subdivLevelGetCount(subSubdiv, frequency);

    assert(output->format == input->format);
    assert(output->count != valueCount);

    Vector_int32_2 subvertices[3] = {{extract->subVertices[0].u, extract->subVertices[0].v},
                                     {extract->subVertices[1].u, extract->subVertices[1].v},
                                     {extract->subVertices[2].u, extract->subVertices[2].v}};

    Vector_int32_2 deltaU        = subvertices[1] - subvertices[0];
    Vector_int32_2 deltaV        = subvertices[2] - subvertices[0];
    int32_t        subEdge       = int32_t(subdivLevelGetSegmentCount(subSubdiv));
    int32_t        subVtxPerEdge = subEdge + 1;

    assert((abs(deltaU.x) == subEdge || deltaU.x == 0) && (abs(deltaU.y) == subEdge || deltaU.y == 0));
    assert((abs(deltaV.x) == subEdge || deltaV.x == 0) && (abs(deltaV.y) == subEdge || deltaV.y == 0));

    deltaU = deltaU / int32_t(subEdge);
    deltaV = deltaV / int32_t(subEdge);

    if(frequency == Frequency::ePerMicroVertex)
    {
        for(int32_t u = 0; u < subVtxPerEdge; u++)
        {
            for(int32_t v = 0; v < subVtxPerEdge - u; v++)
            {
                Vector_int32_2 base = subvertices[0] + (deltaU * u) + (deltaV * v);

                uint32_t baseIdx = layout->pfnGetMicroVertexIndex(base.x, base.y, baseSubdiv, layout->userData);
                uint32_t subIdx  = layout->pfnGetMicroVertexIndex(u, v, subSubdiv, layout->userData);

                Top::copy(output, subIdx, input, baseIdx, triangleOffset, extract);
            }
        }
    }
    else
    {
        for(int32_t u = 0; u < subEdge; u++)
        {
            for(int32_t v = 0; v < subEdge - u; v++)
            {
                // For triangles the coordinates of the uv needs to be the quad corner.
                // Due to sub triangle winding/orientation changes this is not as simple as
                // with vertices before. We need to find the right base quad corner.
                // We use integer points that are within the target triangle and snap
                // them back to find the quad corner they belong to in the base triangle.
                //
                // V
                // |\
                                // | \
                // |__\       lower and upper
                // |\ u|\     triangle with respect
                // |l\ | \    to x
                // |__\|__\
                // x       U
                //
                // Not we only have to do the coordinate space adjustments for
                // the base triangle. The target triangle is using the local
                // u v layout just regularly.

                Vector_int32_2 base4 = (subvertices[0] + (deltaU * u) + (deltaV * v)) * 4;

                {
                    // The target triangle is the "lower quad".
                    // We apply quarter deltas to be get a point within it.
                    // Then snap back from our quadrupeled resolution
                    Vector_int32_2 tri4 = (base4 + deltaU + deltaV);
                    Vector_int32_2 snap = (tri4 / 4);
                    // Still need to figure out if uppoer or lower triangle in base.
                    // Test one of the u (here x) or v coords if they are in the second half.
                    uint32_t snap4Rest = (tri4.x & (~3));
                    uint32_t isUpper   = snap4Rest > 2;

                    uint32_t baseIdx = layout->pfnGetMicroTriangleIndex(snap.x, snap.y, baseSubdiv, 0, layout->userData);
                    uint32_t subIdx  = layout->pfnGetMicroTriangleIndex(u, v, subSubdiv, 0, layout->userData);

                    Top::copy(output, subIdx, input, baseIdx, triangleOffset, extract);
                }
                if(v != subEdge - u - 1)
                {
                    // The target triangle is the "upper quad".
                    // We apply 3 quarter deltas to be get a point within it.
                    // Then snap back from our quadrupeled resolution
                    Vector_int32_2 tri4 = (base4 + deltaU * 3 + deltaV * 3) / 4;
                    Vector_int32_2 snap = (tri4 / 4);
                    // Still need to figure out if uppoer or lower triangle in base.
                    // Test one of the u (here x) or v coords if they are in the second half.
                    uint32_t snap4Rest = (tri4.x & (~3));
                    uint32_t isUpper   = snap4Rest > 2;

                    uint32_t baseIdx = layout->pfnGetMicroTriangleIndex(snap.x, snap.y, baseSubdiv, isUpper, layout->userData);
                    uint32_t subIdx = layout->pfnGetMicroTriangleIndex(u, v, subSubdiv, 1, layout->userData);

                    Top::copy(output, subIdx, input, baseIdx, triangleOffset, extract);
                }
            }
        }
    }
}

MICROMESH_API void MICROMESH_CALL micromeshSubTriangleExtraction(const Micromap*              micromap,
                                                                 const SubTriangleExtraction* extract,
                                                                 ArrayInfo*                   output)
{
    uint32_t triangleOffset = arrayGetV<uint32_t>(micromap->triangleValueIndexOffsets, extract->triangleIndex);
    uint32_t baseSubdiv     = arrayGetV<uint16_t>(micromap->triangleSubdivLevels, extract->triangleIndex);

    struct copyOp
    {
        static void copy(ArrayInfo* output, uint32_t subIdx, const ArrayInfo* input, uint32_t baseIdx, uint32_t triangleOffset, const SubTriangleExtraction* extract)
        {
            std::memcpy(arrayGet<void>(*output, subIdx), arrayGet<void>(*input, baseIdx + triangleOffset), extract->valueCopyBytes);
        }
    };

    doSubTriangleExtraction<copyOp>(&micromap->layout, micromap->frequency, &micromap->values, extract, baseSubdiv,
                                    triangleOffset, output);
}

MICROMESH_API void MICROMESH_CALL micromeshSubTriangleExtractionPacked(const MicromapPacked*        micromap,
                                                                       const SubTriangleExtraction* extract,
                                                                       ArrayInfo*                   output)
{
    uint32_t triangleOffset = arrayGetV<uint32_t>(micromap->triangleValueByteOffsets, extract->triangleIndex);
    uint32_t baseSubdiv     = arrayGetV<uint16_t>(micromap->triangleSubdivLevels, extract->triangleIndex);

    struct copyOpPacked
    {
        static void copy(ArrayInfo*                   output,
                         uint32_t                     subIdx,
                         const ArrayInfo*             inputValues,
                         uint32_t                     baseIdx,
                         uint32_t                     triangleOffset,
                         const SubTriangleExtraction* extract)
        {
            uint16_t u16 = packedReadR11UnormPackedAlign32(arrayGet<void>(*inputValues, triangleOffset), baseIdx);
            packedWriteR11UnormPackedAlign32(output->data, subIdx, u16);
        }
    };

    doSubTriangleExtraction<copyOpPacked>(&micromap->layout, micromap->frequency, &micromap->values, extract,
                                          baseSubdiv, triangleOffset, output);
}

MICROMESH_API uint32_t MICROMESH_CALL micromeshBlockFormatDispC1GetBlockCount(BlockFormatDispC1 blockFormat, uint32_t baseSubdivLevel)
{
    uint32_t blockSubdivLevel = blockFormatDispC1GetSubdivLevel(blockFormat);
    if(blockSubdivLevel == ~0)
        return 0;


    uint32_t splitSubdiv = baseSubdivLevel > blockSubdivLevel ? baseSubdivLevel - blockSubdivLevel : 0;
    return 1 << (splitSubdiv * 2);
}

struct BaryUV_uint32
{
    uint32_t u;
    uint32_t v;
};

static inline void setVertices(BlockTriangle* tri, BaryUV_uint32 w, BaryUV_uint32 u, BaryUV_uint32 v)
{
    tri->vertices[0].u = uint16_t(w.u);
    tri->vertices[0].v = uint16_t(w.v);
    tri->vertices[1].u = uint16_t(u.u);
    tri->vertices[1].v = uint16_t(u.v);
    tri->vertices[2].u = uint16_t(v.u);
    tri->vertices[2].v = uint16_t(v.v);
    tri->signBits      = ((u.u > w.u) ? 1 : 0) | ((v.v > w.v) ? 2 : 0);
}

MICROMESH_API void MICROMESH_CALL micromeshBlockTriangleSplitDispC1(const BlockTriangle* inTri, BlockTriangle* outTris, uint32_t outStride)
{
    /*
    //         C(v)
    //        / \
    //       / V \
    //      vw _ uv
    //     / \ M / \
    //    / W \ / U \
    // A(w) __ uw __ B(u)
    // 
    */

    const uint32_t triW = 0 * outStride;
    const uint32_t triM = 1 * outStride;
    const uint32_t triU = 2 * outStride;
    const uint32_t triV = 3 * outStride;

    // flip state
    outTris[triW].flipped = inTri->flipped;
    outTris[triM].flipped = inTri->flipped ^ 1;
    outTris[triU].flipped = inTri->flipped;
    outTris[triV].flipped = inTri->flipped ^ 1;

    BaryUV_uint32 w = {inTri->vertices[0].u, inTri->vertices[0].v};
    BaryUV_uint32 u = {inTri->vertices[1].u, inTri->vertices[1].v};
    BaryUV_uint32 v = {inTri->vertices[2].u, inTri->vertices[2].v};

    BaryUV_uint32 uw = {(u.u + w.u) / 2, (u.v + w.v) / 2};
    BaryUV_uint32 uv = {(u.u + v.u) / 2, (u.v + v.v) / 2};
    BaryUV_uint32 vw = {(v.u + w.u) / 2, (v.v + w.v) / 2};

    setVertices(&outTris[triW], w, uw, vw);
    setVertices(&outTris[triM], vw, uv, uw);
    setVertices(&outTris[triU], uw, u, uv);
    setVertices(&outTris[triV], uv, vw, v);

    const uint32_t baseEdge0      = (inTri->baseEdgeIndices >> 0) & 3;
    const uint32_t baseEdge1      = (inTri->baseEdgeIndices >> 2) & 3;
    const uint32_t baseEdge2      = (inTri->baseEdgeIndices >> 4) & 3;
    const uint32_t baseEdgeUnused = 3;

    outTris[triW].baseEdgeIndices = (baseEdge0 << 0) | (baseEdgeUnused << 2) | (baseEdge2 << 4);
    outTris[triM].baseEdgeIndices = (baseEdgeUnused << 0) | (baseEdgeUnused << 2) | (baseEdgeUnused << 4);
    outTris[triU].baseEdgeIndices = (baseEdge0 << 0) | (baseEdge1 << 2) | (baseEdgeUnused << 4);
    outTris[triV].baseEdgeIndices = (baseEdgeUnused << 0) | (baseEdge2 << 2) | (baseEdge1 << 4);
}

MICROMESH_API Result MICROMESH_CALL micromeshBlockFormatDispC1GetBlockTriangles(BlockFormatDispC1 blockFormat,
                                                                                uint32_t          baseSubdivLevel,
                                                                                uint32_t          blockTrisCount,
                                                                                BlockTriangle*    blockTris)
{
    uint32_t blockSubdivLevel = blockFormatDispC1GetSubdivLevel(blockFormat);
    uint32_t blockByteSize    = blockFormatDispC1GetByteSize(blockFormat);

    if(blockSubdivLevel == ~0)
    {
        return Result::eInvalidBlockFormat;
    }

    if(blockTrisCount != micromeshBlockFormatDispC1GetBlockCount(blockFormat, baseSubdivLevel))
    {
        return Result::eInvalidRange;
    }

    uint32_t selfSubdiv  = baseSubdivLevel < blockSubdivLevel ? blockSubdivLevel : baseSubdivLevel;
    uint32_t splitSubdiv = selfSubdiv - blockSubdivLevel;

    uint16_t splitMaxCoord       = (1 << selfSubdiv);
    blockTris[0].baseEdgeIndices = (0 << 2) | (1 << 2) | (2 << 4);
    blockTris[0].flipped         = 0;
    setVertices(&blockTris[0], {0, 0}, {splitMaxCoord, 0}, {0, splitMaxCoord});

    uint32_t stride = blockTrisCount;
    for(uint32_t level = 0; level < splitSubdiv; level++)
    {
        uint32_t strideNext = stride / 4;
        uint32_t levelCount = subdivLevelGetTriangleCount(level);
        for(uint32_t i = 0; i < levelCount; i++)
        {
            // load previous split state
            BlockTriangle tri = blockTris[i * stride];
            // save new split states
            micromeshBlockTriangleSplitDispC1(&tri, blockTris + (i * stride), strideNext);
        }

        stride = strideNext;
    }

    for(uint32_t i = 0; i < blockTrisCount; i++)
    {
        blockTris[i].blockByteOffset = i * blockByteSize;
    }

    return Result::eSuccess;
}

MICROMESH_API Result MICROMESH_CALL micromeshBlockFormatDispC1FillBlocks(BlockFormatDispC1     blockFormat,
                                                                         uint32_t              blockTrisCount,
                                                                         const BlockTriangle*  blockTris,
                                                                         const MicromapLayout* baseLayout,
                                                                         uint32_t              baseSubdivLevel,
                                                                         const ArrayInfo*      baseData,
                                                                         size_t                blocksDataSize,
                                                                         void*                 blocksData)
{
    uint8_t* blocksBytes = reinterpret_cast<uint8_t*>(blocksData);

    if(blockFormat != BlockFormatDispC1::eR11_unorm_lvl3_pack512)
    {
        return Result::eInvalidBlockFormat;
    }
    if(baseData->format != Format::eR11_unorm_pack16 && baseData->format != Format::eR11_unorm_packed_align32)
    {
        return Result::eInvalidFormat;
    }
    if(blocksDataSize != 64 * blockTrisCount)
    {
        return Result::eInvalidRange;
    }

    uint32_t blockSubdivLevel = std::min(baseSubdivLevel, 3u);
    uint16_t blockVtxPerEdge  = uint16_t(subdivLevelGetSegmentCount(blockSubdivLevel) + 1);

    // for each block pull the relevant values from base triangle
    for(uint32_t b = 0; b < blockTrisCount; b++)
    {
        // iterate all vertices within block
        for(uint16_t u = 0; u < blockVtxPerEdge; u++)
        {
            for(uint16_t v = 0; v < blockVtxPerEdge - u; v++)
            {
                // from block to base UV
                BaryUV_uint16 blockUV = {u, v};
                BaryUV_uint16 baseUV  = blockTriangleLocalToBaseUV(&blockTris[b], blockUV);

                // fetch base value
                uint32_t baseValueIndex =
                    baseLayout->pfnGetMicroVertexIndex(baseUV.u, baseUV.v, baseSubdivLevel, baseLayout->userData);
                uint16_t baseValue = baseData->format == Format::eR11_unorm_packed_align32 ?
                                         packedReadR11UnormPackedAlign32(baseData->data, baseValueIndex) :
                                         arrayGetV<uint16_t>(*baseData, baseValueIndex);

                // write unorm11 in bird order into current block's bytes
                uint32_t blockValueIndex = bird_getMicroVertexIndex(u, v, blockSubdivLevel, nullptr);
                packedWriteR11UnormPackedAlign32(blocksBytes + blockTris[b].blockByteOffset, blockValueIndex, baseValue);
            }
        }
    }

    return Result::eSuccess;
}

}  // namespace micromesh
