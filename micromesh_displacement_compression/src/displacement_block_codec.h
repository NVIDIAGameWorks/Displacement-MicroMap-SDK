/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Internal interface for the block encoder and decoder. Ported from the
// old displacement encoder.
// The idea is that this exposes the interface the mesh encoder needs to be
// efficient: 11-bit UNORM values in u-major layout, all API preconditions
// already checked. Data transfer and conversion operations are left up to the
// higher-level API functions. Ideally, these would be templated so we could
// get good code generation.
// This also exposes the functions that are unit-tested.

#pragma once

#include <micromesh/micromesh_types.h>

// Using __restrict here tells compilers that e.g. our input and output don't
// alias, which might give some additional optimizations.
// TODO: Test how this affects codegen/performance.
#define DISPENC_RESTRICT __restrict

namespace micromesh
{
namespace dispenc
{
uint16_t predict(uint16_t a, uint16_t b);

// if numCorrectionBits is 0, returns 0. Otherwise, returns the integer v such that
// * `(decoderWordMask & (prediction + (v << shift)))` is as close to `reference` as possible
// * v is between -2^(numCorrectionBits-1) and 2^(numCorrectionBits-1)-1
int16_t correct(uint16_t prediction, uint16_t reference, uint16_t shift, uint16_t numCorrectionBits);

static constexpr uint32_t MAX_COMPRESSION_LEVELS = 5;

enum class VertexType : uint8_t
{
    eInterior = 0,
    eEdge0,
    eEdge1,
    eEdge2,
    eNUM_VERTEXTYPES
};
static constexpr uint8_t NUM_VERTEXTYPES = (uint8_t)VertexType::eNUM_VERTEXTYPES;

struct Shifts
{
    struct Level
    {
        // Shift correction values by this amount prior to applying it as delta to prediction
        uint16_t vertex[NUM_VERTEXTYPES]{};
    };

    // levels[i] contains the shifts for subdiv level i+2; subdiv levels 0
    // (anchors) and 1 are always lossless, so don't need shifts
    Level levels[MAX_COMPRESSION_LEVELS - 1]{};
};

// Structure the encoder writes to.
struct Intermediate
{
    Shifts m_shifts;
    // These maps are umajor indexed.
    //
    // Decoded displacement map (i.e. what the decoder would output, although it's generated while encoding...)
    uint16_t* DISPENC_RESTRICT m_decoded = nullptr;
    // Correction terms (applied delta to predicted value from parent vertices)
    int16_t* DISPENC_RESTRICT m_corrections = nullptr;
};

// NOTE: With the current design, for formats other than
// eDispC1_r11_unorm_block, we have to go through blockEncodeAndPack directly.
// This could be cleaner - my concern is about the performance impact of
// packing when we discard the information for a less compressed format.
void blockEncode(BlockFormatDispC1 fmtC1, const uint32_t subdivLevel, const uint16_t* DISPENC_RESTRICT reference, Intermediate& DISPENC_RESTRICT result);

// Same note as blockEncode
void blockPackData(BlockFormatDispC1 fmtC1, const uint32_t subdivLevel, const Intermediate& DISPENC_RESTRICT ie, void* DISPENC_RESTRICT encodedData);

void blockDecode(Format                       family,
                 BlockFormatDispC1            fmt,
                 const uint32_t               subdivLevel,
                 const void* DISPENC_RESTRICT encodedData,
                 uint16_t* DISPENC_RESTRICT   decoded);

}  // namespace dispenc
}  // namespace micromesh