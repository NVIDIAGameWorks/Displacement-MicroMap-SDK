/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <cassert>
#include <mutex>
#include "displacement_configs.h"
#include <micromesh/micromesh_utils.h>

#include <memory>
#include <string.h>

namespace micromesh
{
// CODE THAT SHOULD BE IN INTERNAL API
// Compute 2 16-bit prefix XORs in a 32-bit register
uint32_t prefixEor2(uint32_t x)
{
    x ^= (x >> 1) & 0x7fff7fff;
    x ^= (x >> 2) & 0x3fff3fff;
    x ^= (x >> 4) & 0x0fff0fff;
    x ^= (x >> 8) & 0x00ff00ff;
    return x;
}

// Interleave 16 even bits from x with 16 odd bits from y
uint32_t interleaveBits(uint32_t x, uint32_t y)
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
uint32_t getTripletIndex(uint32_t u, uint32_t v, uint32_t w, uint32_t level)
{
    const uint32_t coordMask = ((1U << level) - 1);

    uint32_t b0 = ~(u ^ w) & coordMask;
    uint32_t t  = (u ^ v) & b0;
    uint32_t c  = (((u & v & w) | (~u & ~v & ~w)) & coordMask) << 16;
    uint32_t f  = prefixEor2(t | c) ^ u;
    uint32_t b1 = (f & ~b0) | t;

    uint32_t dist = interleaveBits(b0, b1);

    f >>= 16;
    b0 <<= 1;
    return (dist + (b0 & ~f) - (b0 & f)) >> 3;
}
// END CODE THAT SHOULD BE IN INTERNAL API

namespace dispenc
{
// Mechanism for only initializing the format table once - ideally we'd
// generate this at compile-time.
static bool       s_configsInitialized = false;
static std::mutex s_modifyingConfigs;
std::array<MicromeshConfig, NUM_C1_BLOCK_FORMATS> s_micromeshConfigs{};

void genMicromeshConfigTables(MicromeshConfig& microMeshConfig)
{
    // Stamp subdivision level on base triangle
    const uint32_t maxSubdiv   = microMeshConfig.subdiv;
    const uint32_t numSegments = microMeshConfig.numSegments;

    const uint32_t numVtxPerEdge = 1u + numSegments;
    const uint32_t numElems      = subdivLevelGetVertexCount(maxSubdiv);

    // Allocate decoded displacement map (i.e. what the decoder would output, although it's generated while encoding..)
    std::unique_ptr<int32_t[]> decodedPtr = std::make_unique<int32_t[]>(numElems);
    auto                       decoded    = decodedPtr.get();

    {
        int32_t  subdLevel = 0u;
        uint32_t ns        = numSegments;

        // w, u and v anchors addresses
        const uint32_t anchors_addr[3] = {
            umajorUVtoLinear(0, 0, maxSubdiv),
            umajorUVtoLinear(ns, 0, maxSubdiv),
            umajorUVtoLinear(0, ns, maxSubdiv),
        };
        for(uint32_t i = 0; i < 3; i++)
        {
            decoded[anchors_addr[i]] = subdLevel;
        }

        subdLevel++;  // done with anchors, move to the next level..

        // Just stamp the subdivision level
        while(ns > 1)
        {
            const uint32_t hns = ns >> 1;
            for(uint32_t u = 0; u < numVtxPerEdge; u += ns)
            {
                for(uint32_t v = hns; v < numVtxPerEdge - u; v += ns)
                {
                    decoded[umajorUVtoLinear(u, v, maxSubdiv)]       = subdLevel;
                    decoded[umajorUVtoLinear(v, u, maxSubdiv)]       = subdLevel;
                    decoded[umajorUVtoLinear(u + hns, v, maxSubdiv)] = subdLevel;
                }
            }

            // Move to the next (finer) subdivision level
            ns = hns;
            subdLevel++;
        }
    }

    // Generate useful info about each level of subdivision

    struct SubdInfo
    {
        uint32_t segments;     // number of segments per base face edge at this subd level
        uint32_t triangles;    // total number of triangles at this subd level
        uint32_t edges;        // total number of edges at this subd level
        uint32_t vertices;     // total number of vertices at this subd level
        uint32_t corrections;  // number of corrections needed *at* this subd level (incremental, not total)
    };
    container::vector<SubdInfo> subdInfo;
    {
        SubdInfo info{};
        info.segments    = 1u;
        info.triangles   = 1u;
        info.edges       = 3u;
        info.vertices    = 3u;
        info.corrections = 3u;
        subdInfo.push_back(info);
        for(uint32_t i = 0; i < maxSubdiv; i++)
        {
            const auto& p         = subdInfo.back();
            auto        segments  = 2 * p.segments;
            auto        triangles = 4 * p.triangles;
            auto edges = 2 * p.edges + 3 * p.triangles;  // each edge is split in 2, plus 3 new edges for each split triangle
            auto vertices    = edges + p.vertices;       // each edge generates a new vertex
            auto corrections = p.edges;
            subdInfo.push_back({segments, triangles, edges, vertices, corrections});
        }
    }

    // Allocate mem for our table (stored in a triangular matrix) and also for a secondary table
    // that stores correction sizes, which is only used to later pack data.
    const uint32_t totalNumVertices = ((numVtxPerEdge * (1 + numVtxPerEdge) / 2));
    const uint32_t tableSizeInBytes = sizeof(uint32_t) * totalNumVertices;
    microMeshConfig.bary2BlockAddrTable.resize(totalNumVertices);
    microMeshConfig.correctionSizeInBits.resize(totalNumVertices);
    microMeshConfig.subdLevelBitAddr.resize(1u + maxSubdiv);

    // Initialize per-subdivision-level bit address mini table
    uint32_t bitsUsedSoFar = 0u;
    for(uint32_t subdLevel = 0; subdLevel <= maxSubdiv; subdLevel++)
    {
        microMeshConfig.subdLevelBitAddr[subdLevel] = bitsUsedSoFar;
        bitsUsedSoFar += microMeshConfig.numCorrBits[subdLevel] * subdInfo[subdLevel].corrections;
    }
    auto bitAddr = microMeshConfig.subdLevelBitAddr;

    auto tableData          = microMeshConfig.bary2BlockAddrTable.data();
    auto corrSizeInBitsData = microMeshConfig.correctionSizeInBits.data();
    memset((void*)tableData, 0xcdu, tableSizeInBytes);
    memset((void*)corrSizeInBitsData, 0xcdu, tableSizeInBytes);

    for(uint32_t u = 0; u < numVtxPerEdge; u++)
    {
        // We only fill the upper triangular half of table
        for(uint32_t v = 0; v < numVtxPerEdge - u; v++)
        {
            // TODO: These are the same now!
            const uint32_t addressinTexture = umajorUVtoLinear(u, v, maxSubdiv);
            const uint32_t addressInTable   = umajorUVtoLinear(u, v, maxSubdiv);
            const uint32_t subdLevel        = (uint32_t)decoded[addressinTexture];
            const uint32_t numCorrBits      = microMeshConfig.numCorrBits[subdLevel];
            const uint32_t shift            = maxSubdiv - subdLevel;

            const uint32_t uu = u >> shift;
            const uint32_t vv = v >> shift;


            const uint32_t numSubdSegments = 1u << subdLevel;
            const uint32_t ww              = numSubdSegments - uu - vv;
            uint32_t       addressinBlock  = getTripletIndex(uu, vv, ww, subdLevel);
            addressinBlock *= 3;

            const uint32_t r_u = uu & 0x1u;
            const uint32_t r_v = vv & 0x1u;

            // The anchors of the base triangle are handled as a special case
            if(subdLevel == 0)
            {
                if(uu == 1u && vv == 0u)
                    addressinBlock += 1;
                if(uu == 0u && vv == 1u)
                    addressinBlock += 2;
            }
            else
            {
                if(r_u == 1 && r_v == 1)
                    addressinBlock += 1;
                if(r_u == 1 && r_v == 0)
                    addressinBlock += 2;
            }

            addressinBlock *= numCorrBits;
            addressinBlock += microMeshConfig.subdLevelBitAddr[subdLevel];

            // Read the level of subdvision for this vertex and update table

            tableData[addressInTable]          = addressinBlock;
            corrSizeInBitsData[addressInTable] = numCorrBits;

            // Now update the number of bits we've written so far for this subd level
            bitAddr[subdLevel] += numCorrBits;
        }
    }

    // Correctness check
    for(uint32_t subdLevel = 0; subdLevel < maxSubdiv; subdLevel++)
    {
        // If test is not passed then we miscounted the number of bits required to store
        // corrections for each subdivision level OR we did not find all the corrections
        // that belong to that level.
        assert(bitAddr[subdLevel] == microMeshConfig.subdLevelBitAddr[subdLevel + 1] && "Table is not consistent.");
    }
}

void initMicromeshConfigs()
{
    // Don't reinitialize if we've already initialized it:
    if(s_configsInitialized)
    {
        return;
    }

    // Prevent multiple threads from modifying the vector at the same time
    std::scoped_lock lock(s_modifyingConfigs);

    if(s_configsInitialized)
    {
        return;  // Two threads arrived at s_modifyingConfigs and we were the second
    }

    const uint32_t largeBlockSizeInBits = 1024;
    const uint32_t smallBlockSizeInBits = largeBlockSizeInBits / 2;

    // 1b/triangle
    MicromeshConfig& lvl5 = s_micromeshConfigs[blockFormatDispC1ToConfigIdx(BlockFormatDispC1::eR11_unorm_lvl5_pack1024)];
    lvl5.subdiv           = 5;
    lvl5.numSegments      = 32u;
    lvl5.blockSizeInBits  = largeBlockSizeInBits;
    lvl5.numCorrBits      = {11u, 11u, 8u, 4u, 2u, 1u};
    lvl5.fmt              = BlockFormatDispC1::eR11_unorm_lvl5_pack1024;

    // 4b/triangle
    MicromeshConfig& lvl4 = s_micromeshConfigs[blockFormatDispC1ToConfigIdx(BlockFormatDispC1::eR11_unorm_lvl4_pack1024)];
    lvl4.subdiv           = 4;
    lvl4.numSegments      = 16u;
    lvl4.blockSizeInBits  = largeBlockSizeInBits;
    lvl4.numCorrBits      = {11u, 11u, 11u, 10u, 5u};
    lvl4.fmt              = BlockFormatDispC1::eR11_unorm_lvl4_pack1024;

    // 8b/triangle --> special configuration for lossless encoding
    MicromeshConfig& lvl3 = s_micromeshConfigs[blockFormatDispC1ToConfigIdx(BlockFormatDispC1::eR11_unorm_lvl3_pack512)];
    lvl3.subdiv           = 3;
    lvl3.numSegments      = 8u;
    lvl3.blockSizeInBits  = smallBlockSizeInBits;
    lvl3.numCorrBits      = {11u, 11u, 11u, 11u};
    lvl3.fmt              = BlockFormatDispC1::eR11_unorm_lvl3_pack512;

    for(MicromeshConfig& config : s_micromeshConfigs)
    {
        genMicromeshConfigTables(config);
    }

    s_configsInitialized = true;
}

}  // namespace dispenc
}  // namespace micromesh
