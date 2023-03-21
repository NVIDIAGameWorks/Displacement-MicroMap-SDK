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

#include <micromesh/micromesh_types.h>

// clang-format off

// Optional header for those who prefer to use auto etc.

namespace micromesh
{
struct FormatType
{
    struct r8_snorm { int8_t  r; };
    struct rg8_snorm  { int8_t r; int8_t g; };
    struct rgb8_snorm  { int8_t r; int8_t g; uint8_t b; };
    struct rgba8_snorm  { int8_t r; int8_t g; int8_t b; int8_t a; };

    struct r8_unorm { uint8_t r; };
    struct rg8_unorm  { uint8_t r; uint8_t g; };
    struct rgb8_unorm  { uint8_t r; uint8_t g; uint8_t b; };
    struct rgba8_unorm  { uint8_t r; uint8_t g; uint8_t b; uint8_t a; };

    struct r8_uint  { uint8_t r; };
    struct rg8_uint  { uint8_t r; uint8_t g; };
    struct rgb8_uint  { uint8_t r; uint8_t g; uint8_t b; };
    struct rgba8_uint  { uint8_t r; uint8_t g; uint8_t b; uint8_t a; };

    struct r8_sint  { int8_t r; };
    struct rg8_sint  { int8_t r; int8_t g; };
    struct rgb8_sint  { int8_t r; int8_t g; int8_t b; };
    struct rgba8_sint  { int8_t r; int8_t g; int8_t b; int8_t a; };

    struct r16_uint  { uint16_t r; };
    struct rg16_uint  { uint16_t r; uint16_t g; };
    struct rgb16_uint  { uint16_t r; uint16_t g; uint16_t b; };
    struct rgba16_uint  { uint16_t r; uint16_t g; uint16_t b; uint16_t a; };

    struct r16_sint  { int16_t r; };
    struct rg16_sint  { int16_t r; int16_t g; };
    struct rgb16_sint  { int16_t r; int16_t g; int16_t b; };
    struct rgba16_sint  { int16_t r; int16_t g; int16_t b; int16_t a; };

    struct r16_snorm { int16_t  r; };
    struct rg16_snorm  { int16_t r; int16_t g; };
    struct rgb16_snorm  { int16_t r; int16_t g; uint16_t b; };
    struct rgba16_snorm  { int16_t r; int16_t g; int16_t b; int16_t a; };

    struct r16_unorm { uint16_t r; };
    struct rg16_unorm  { uint16_t r; uint16_t g; };
    struct rgb16_unorm  { uint16_t r; uint16_t g; uint16_t b; };
    struct rgba16_unorm  { uint16_t r; uint16_t g; uint16_t b; uint16_t a; };

    struct r16_sfloat  { uint16_t r; };
    struct rg16_sfloat  { uint16_t r; uint16_t g; };
    struct rgb16_sfloat  { uint16_t r; uint16_t g; uint16_t b; };
    struct rgba16_sfloat  { uint16_t r; uint16_t g; uint16_t b; uint16_t a; };

    struct r32_sfloat  { float r; };
    struct rg32_sfloat  { float r; float g; };
    struct rgb32_sfloat  { float r; float g; float b; };
    struct rgba32_sfloat  { float r; float g; float b; float a; };

    struct r32_uint  { uint32_t r; };
    struct rg32_uint  { uint32_t r; uint32_t g; };
    struct rgb32_uint  { uint32_t r; uint32_t g; uint32_t b; };
    struct rgba32_uint  { uint32_t r; uint32_t g; uint32_t b; uint32_t a; };

    struct r32_sint  { int32_t r; };
    struct rg32_sint  { int32_t r; int32_t g; };
    struct rgb32_sint  { int32_t r; int32_t g; int32_t b; };
    struct rgba32_sint  { int32_t r; int32_t g; int32_t b; int32_t a; };

    struct r64_sfloat  { double r; };
    struct rg64_sfloat  { double r; double g; };
    struct rgb64_sfloat  { double r; double g; double b; };
    struct rgba64_sfloat  { double r; double g; double b; double a; };

    struct r64_uint  { uint64_t r; };
    struct rg64_uint  { uint64_t r; uint64_t g; };
    struct rgb64_uint  { uint64_t r; uint64_t g; uint64_t b; };
    struct rgba64_uint  { uint64_t r; uint64_t g; uint64_t b; uint64_t a; };

    struct r64_sint  { int64_t r; };
    struct rg64_sint  { int64_t r; int64_t g; };
    struct rgb64_sint  { int64_t r; int64_t g; int64_t b; };
    struct rgba64_sint  { int64_t r; int64_t g; int64_t b; int64_t a; };

    struct opac1_rx_uint_block { uint8_t packed; };
    struct dispc1_r11_unorm_block { uint32_t packed[16]; };

    struct r11_unorm_pack16 { uint16_t r;};
    struct r11_unorm_packed_align32 { uint32_t packed;};

    template <class T>
    static Format fmt();
};

template <> inline Format FormatType::fmt<int8_t>() { return Format::eR8_sint; }
template <> inline Format FormatType::fmt<uint8_t>() { return Format::eR8_uint; }
template <> inline Format FormatType::fmt<int16_t>() { return Format::eR16_sint; }
template <> inline Format FormatType::fmt<uint16_t>() { return Format::eR16_uint; }
template <> inline Format FormatType::fmt<int32_t>() { return Format::eR32_sint; }
template <> inline Format FormatType::fmt<uint32_t>() { return Format::eR32_uint; }
template <> inline Format FormatType::fmt<float>() { return Format::eR32_sfloat; }

template <> inline Format FormatType::fmt<Vector_uint32_2>() { return Format::eRG32_uint; }
template <> inline Format FormatType::fmt<Vector_uint32_3>() { return Format::eRGB32_uint; }

template <> inline Format FormatType::fmt<Vector_float_2>() { return Format::eRG32_sfloat; }
template <> inline Format FormatType::fmt<Vector_float_3>() { return Format::eRGB32_sfloat; }
template <> inline Format FormatType::fmt<Vector_float_4>() { return Format::eRGBA32_sfloat; }

template <> inline Format FormatType::fmt<FormatType::r8_unorm>() { return Format::eR8_unorm; }
template <> inline Format FormatType::fmt<FormatType::rg8_unorm>() { return Format::eRG8_unorm; }
template <> inline Format FormatType::fmt<FormatType::rgb8_unorm>() { return Format::eRGB8_unorm; }
template <> inline Format FormatType::fmt<FormatType::rgba8_unorm>() { return Format::eRGBA8_unorm; }

template <> inline Format FormatType::fmt<FormatType::r8_snorm>() { return Format::eR8_snorm; }
template <> inline Format FormatType::fmt<FormatType::rg8_snorm>() { return Format::eRG8_snorm; }
template <> inline Format FormatType::fmt<FormatType::rgb8_snorm>() { return Format::eRGB8_snorm; }
template <> inline Format FormatType::fmt<FormatType::rgba8_snorm>() { return Format::eRGBA8_snorm; }

template <> inline Format FormatType::fmt<FormatType::r8_sint>() { return Format::eR8_sint; }
template <> inline Format FormatType::fmt<FormatType::rg8_sint>() { return Format::eRG8_sint; }
template <> inline Format FormatType::fmt<FormatType::rgb8_sint>() { return Format::eRGB8_sint; }
template <> inline Format FormatType::fmt<FormatType::rgba8_sint>() { return Format::eRGBA8_sint; }

template <> inline Format FormatType::fmt<FormatType::r8_uint>() { return Format::eR8_uint; }
template <> inline Format FormatType::fmt<FormatType::rg8_uint>() { return Format::eRG8_uint; }
template <> inline Format FormatType::fmt<FormatType::rgb8_uint>() { return Format::eRGB8_uint; }
template <> inline Format FormatType::fmt<FormatType::rgba8_uint>() { return Format::eRGBA8_uint; }

template <> inline Format FormatType::fmt<FormatType::r16_unorm>() { return Format::eR16_unorm; }
template <> inline Format FormatType::fmt<FormatType::rg16_unorm>() { return Format::eRG16_unorm; }
template <> inline Format FormatType::fmt<FormatType::rgb16_unorm>() { return Format::eRGB16_unorm; }
template <> inline Format FormatType::fmt<FormatType::rgba16_unorm>() { return Format::eRGBA16_unorm; }

template <> inline Format FormatType::fmt<FormatType::r16_snorm>() { return Format::eR16_snorm; }
template <> inline Format FormatType::fmt<FormatType::rg16_snorm>() { return Format::eRG16_snorm; }
template <> inline Format FormatType::fmt<FormatType::rgb16_snorm>() { return Format::eRGB16_snorm; }
template <> inline Format FormatType::fmt<FormatType::rgba16_snorm>() { return Format::eRGBA16_snorm; }

template <> inline Format FormatType::fmt<FormatType::r16_sfloat>() { return Format::eR16_sfloat; }
template <> inline Format FormatType::fmt<FormatType::rg16_sfloat>() { return Format::eRG16_sfloat; }
template <> inline Format FormatType::fmt<FormatType::rgb16_sfloat>() { return Format::eRGB16_sfloat; }
template <> inline Format FormatType::fmt<FormatType::rgba16_sfloat>() { return Format::eRGBA16_sfloat; }

template <> inline Format FormatType::fmt<FormatType::r16_sint>() { return Format::eR16_sint; }
template <> inline Format FormatType::fmt<FormatType::rg16_sint>() { return Format::eRG16_sint; }
template <> inline Format FormatType::fmt<FormatType::rgb16_sint>() { return Format::eRGB16_sint; }
template <> inline Format FormatType::fmt<FormatType::rgba16_sint>() { return Format::eRGBA16_sint; }

template <> inline Format FormatType::fmt<FormatType::r16_uint>() { return Format::eR16_uint; }
template <> inline Format FormatType::fmt<FormatType::rg16_uint>() { return Format::eRG16_uint; }
template <> inline Format FormatType::fmt<FormatType::rgb16_uint>() { return Format::eRGB16_uint; }
template <> inline Format FormatType::fmt<FormatType::rgba16_uint>() { return Format::eRGBA16_uint; }

template <> inline Format FormatType::fmt<FormatType::r32_sfloat>() { return Format::eR32_sfloat; }
template <> inline Format FormatType::fmt<FormatType::rg32_sfloat>() { return Format::eRG32_sfloat; }
template <> inline Format FormatType::fmt<FormatType::rgb32_sfloat>() { return Format::eRGB32_sfloat; }
template <> inline Format FormatType::fmt<FormatType::rgba32_sfloat>() { return Format::eRGBA32_sfloat; }

template <> inline Format FormatType::fmt<FormatType::r32_sint>() { return Format::eR32_sint; }
template <> inline Format FormatType::fmt<FormatType::rg32_sint>() { return Format::eRG32_sint; }
template <> inline Format FormatType::fmt<FormatType::rgb32_sint>() { return Format::eRGB32_sint; }
template <> inline Format FormatType::fmt<FormatType::rgba32_sint>() { return Format::eRGBA32_sint; }

template <> inline Format FormatType::fmt<FormatType::r32_uint>() { return Format::eR32_uint; }
template <> inline Format FormatType::fmt<FormatType::rg32_uint>() { return Format::eRG32_uint; }
template <> inline Format FormatType::fmt<FormatType::rgb32_uint>() { return Format::eRGB32_uint; }
template <> inline Format FormatType::fmt<FormatType::rgba32_uint>() { return Format::eRGBA32_uint; }

template <> inline Format FormatType::fmt<FormatType::r64_sfloat>() { return Format::eR64_sfloat; }
template <> inline Format FormatType::fmt<FormatType::rg64_sfloat>() { return Format::eRG64_sfloat; }
template <> inline Format FormatType::fmt<FormatType::rgb64_sfloat>() { return Format::eRGB64_sfloat; }
template <> inline Format FormatType::fmt<FormatType::rgba64_sfloat>() { return Format::eRGBA64_sfloat; }

template <> inline Format FormatType::fmt<FormatType::r64_sint>() { return Format::eR64_sint; }
template <> inline Format FormatType::fmt<FormatType::rg64_sint>() { return Format::eRG64_sint; }
template <> inline Format FormatType::fmt<FormatType::rgb64_sint>() { return Format::eRGB64_sint; }
template <> inline Format FormatType::fmt<FormatType::rgba64_sint>() { return Format::eRGBA64_sint; }

template <> inline Format FormatType::fmt<FormatType::r64_uint>() { return Format::eR64_uint; }
template <> inline Format FormatType::fmt<FormatType::rg64_uint>() { return Format::eRG64_uint; }
template <> inline Format FormatType::fmt<FormatType::rgb64_uint>() { return Format::eRGB64_uint; }
template <> inline Format FormatType::fmt<FormatType::rgba64_uint>() { return Format::eRGBA64_uint; }

template <> inline Format FormatType::fmt<FormatType::opac1_rx_uint_block>() { return Format::eOpaC1_rx_uint_block; }
template <> inline Format FormatType::fmt<FormatType::dispc1_r11_unorm_block>() { return Format::eDispC1_r11_unorm_block; }
template <> inline Format FormatType::fmt<FormatType::r11_unorm_pack16>() { return Format::eR11_unorm_pack16; }
template <> inline Format FormatType::fmt<FormatType::r11_unorm_packed_align32>() { return Format::eR11_unorm_packed_align32; }

// this only works for types T that implement the static `FormatType::fmt<T>()` function
template <typename T>
inline void arraySetFormatTypeData(ArrayInfo& info, const T* data, uint64_t count, uint32_t stride = 0)
{
    info.data       = const_cast<void*>(reinterpret_cast<const void*>(data));
    info.count      = count;
    info.format     = FormatType::fmt<T>();
    info.byteStride = stride ? stride : uint32_t(sizeof(T));
}

// compressed formats must use byteStride = 1
template <>
inline void arraySetFormatTypeData<FormatType::opac1_rx_uint_block>(ArrayInfo& info, const FormatType::opac1_rx_uint_block* data, uint64_t count, uint32_t stride)
{
    info.data       = const_cast<void*>(reinterpret_cast<const void*>(data));
    info.count      = count;
    info.format     = Format::eOpaC1_rx_uint_block;
    info.byteStride = 1;
}

// compressed formats must use byteStride = 1
template <>
inline void arraySetFormatTypeData<FormatType::dispc1_r11_unorm_block>(ArrayInfo& info, const FormatType::dispc1_r11_unorm_block* data, uint64_t count, uint32_t stride)
{
    info.data       = const_cast<void*>(reinterpret_cast<const void*>(data));
    info.count      = count;
    info.format     = Format::eDispC1_r11_unorm_block;
    info.byteStride = 1;
}

// packed formats must use byteStride = 1
template <>
inline void arraySetFormatTypeData<FormatType::r11_unorm_packed_align32>(ArrayInfo& info, const FormatType::r11_unorm_packed_align32* data, uint64_t count, uint32_t stride)
{
    info.data       = const_cast<void*>(reinterpret_cast<const void*>(data));
    info.count      = count;
    info.format     = Format::eR11_unorm_packed_align32;
    info.byteStride = 1;
}

template <typename T>
inline void arraySetFormatTypeDataVec(ArrayInfo& info, const T& vec)
{
    arraySetFormatTypeData(info, vec.data(), vec.size(), 0);
}

template <typename Ta>
inline Format accessorFormat(const Ta&) {
    return FormatType::fmt<Ta::value_type>();
}

}