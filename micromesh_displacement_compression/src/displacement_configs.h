/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Internal interface for low-level details about block formats - such as where
// the individual correction sections are, the bit widths of different levels,
// and so on.

#pragma once

#include <array>
#include <micromesh_internal/micromesh_containers.h>
#include <micromesh/micromesh_types.h>

namespace micromesh
{
namespace dispenc
{
static const uint32_t config_maxSubdLevel          = 13u;
static const uint32_t config_decoderWordSizeInBits = 11u;

// Configuration-specific data describing the full configuration. Const during
// encoding.
struct MicromeshConfig
{
    uint32_t              subdiv{};
    uint32_t              numSegments{};
    uint32_t              blockSizeInBits{};
    BlockFormatDispC1     fmt{};
    container::vector<uint32_t> numCorrBits;
    container::vector<uint32_t> bary2BlockAddrTable;   // umajor indexed
    container::vector<uint32_t> correctionSizeInBits;  // umajor indexed
    container::vector<uint32_t> subdLevelBitAddr;      // bitpos where each subdiv level starts

    inline bool hasFlatEncoding() const { return numSegments == 8 && blockSizeInBits == 512; }
};

static constexpr uint16_t NUM_C1_BLOCK_FORMATS = 3;
static_assert(uint16_t(BlockFormatDispC1::eInvalid) == 0, "Assumption on BlockFormatDispC1 layout broke!");
static_assert(uint16_t(BlockFormatDispC1::eR11_unorm_lvl5_pack1024) == 3, "Assumption on BlockFormatDispC1 layout broke!");
inline uint16_t blockFormatDispC1ToConfigIdx(const BlockFormatDispC1& fmt){
    return uint16_t(fmt) - 1;
}

// Table of MicromeshConfig data so that we only have to generate them once
extern std::array<MicromeshConfig, NUM_C1_BLOCK_FORMATS> s_micromeshConfigs;

void initMicromeshConfigs();

const inline MicromeshConfig& getMicromeshConfig(const BlockFormatDispC1& fmt)
{
    initMicromeshConfigs();
    return s_micromeshConfigs[blockFormatDispC1ToConfigIdx(fmt)];
}

}  // namespace dispenc
}  // namespace micromesh
