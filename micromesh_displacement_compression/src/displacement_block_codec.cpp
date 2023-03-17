/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <cmath>
#include <micromesh/micromesh_utils.h>
#include "cpp_compatibility.h"
#include "displacement_block_codec.h"
#include "displacement_configs.h"

#include <string.h>

namespace micromesh
{
namespace dispenc
{
// Utility functions
uint32_t floorLog2(uint32_t v)
{
    if(v == 0)
    {
        return 0;  // otherwise countl_zero returns 32
    }
    return 31u - compat::countl_zero(v);
}

// Returns bits srcBitPos...srcBitPos+numBitsToRead-1, sign-extending if signedInput is true.
int32_t readUpTo16Bits(const uint8_t* src, uint32_t srcBitPos, uint32_t numBitsToRead, bool signedInput = true)
{
    if(numBitsToRead > 16)
    {
        assert(false);
        return 0;
    }
    const uint32_t bits = packedBitRead(src, srcBitPos, numBitsToRead);
    if(signedInput)
    {
        const uint32_t shift = 32u - numBitsToRead;
        return (int32_t(bits << shift)) >> shift;
    }
    else
    {
        return int32_t(bits);
    }
}

//////////////////////////////////////////////////////////////////////////////
// Primary functions

uint16_t predict(uint16_t a, uint16_t b)
{
    uint16_t sum = a + b + 1;
    return sum >> 1;
}

int16_t correct(uint16_t prediction, uint16_t reference, uint16_t shift, uint16_t numCorrectionBits)
{
    if(numCorrectionBits == 0)
        return 0;

    // The metric for this function is tricky! It's tempting to start with
    // reference - prediction and then work mod 2048, but the issue is that the
    // error metric isn't distance mod 2048, but rather distance over the range
    // of integers [0, 2047]. For instance, suppose prediction == 100,
    // reference == 2047, shift == 0, and numCorrectionBits == 7. If we were
    // working mod 2048, going as far as we can in the negative direction
    // (100-64) would be the closest. But in the metric over the integers,
    // we want to go as far as we can in the positive direction (100+63)!

    // We'll generate two candidate points and choose the best one.
    uint16_t iLo, iHi;
    // Split into two cases. The first tells us that R will be closest to either
    // the minimum or maximum representable value.
    const int16_t  diff          = (int16_t)reference - (int16_t)prediction;
    const uint16_t negativeRange = 1u << (numCorrectionBits - 1 + shift);
    const uint16_t positiveRange = negativeRange - (1u << shift);
    const uint16_t modulus       = 1u << config_decoderWordSizeInBits;
    const uint16_t mask          = modulus - 1;

    if(((diff & mask) > positiveRange)                // "too far for the positive side"
       && (modulus - (diff & mask) > negativeRange))  // "too far for the negative side"
    {
        iLo = -(1 << (numCorrectionBits - 1));
        iHi = (1 << (numCorrectionBits - 1)) - 1;
    }
    else
    {
        // Without the modulus, R is between two representable values.
        // With the modulus, it's a bit more complex. First, "unwrap" things
        // so that the difference R-P will produce a value in range:
        int16_t unwrappedDiff = diff;
        if(diff > (int16_t)positiveRange)
        {
            unwrappedDiff -= modulus;
        }
        else if(diff < -(int16_t)negativeRange)
        {
            unwrappedDiff += modulus;
        }
        // Now read off the indices:
        iLo = (int16_t)std::floor((float)unwrappedDiff / (float)(1 << shift));
        iHi = iLo + 1;
    }

    // Choose the one with a closer image, or the lower one in case of ties.
    const int16_t errorLo = (int16_t)reference - (int16_t)(mask & (uint16_t)(prediction + (iLo << shift)));
    const int16_t errorHi = (int16_t)reference - (int16_t)(mask & (uint16_t)(prediction + (iHi << shift)));
    if(std::abs(errorLo) <= std::abs(errorHi))
    {
        return iLo;
    }
    else
    {
        return iHi;
    }
}

uint32_t getNumBitsNeeded(uint32_t x)
{
    return 32u - compat::countl_zero(x);
}

VertexType getVertexType(BaryUV_uint16 uv, uint32_t numSegments)
{
    const uint32_t w = numSegments - uv.u - uv.v;

    // Corners/anchors don't have a vertex type
    // If we hit one of the cases below we called this function with incorrect arguments.
    assert(!(uv.u == numSegments && uv.v == 0 && w == 0));
    assert(!(uv.u == 0 && uv.v == numSegments && w == 0));
    assert(!(uv.u == 0 && uv.v == 0 && w == numSegments));

    // Vertices are identified in counter-clockwise order, starting from w: w -> u -> v
    // Edges follow the same rule: e0 -> e1 -> e2
    // ************************
    // *           V          *
    // *          /\          *
    // *         /  \         *
    // *        /    \        *
    // *       /      \       *
    // *   e2 /________\ e1   *
    // *     /\        /\     *
    // *    /  \      /  \    *
    // *   /    \    /    \   *
    // *  /      \  /      \  *
    // * /        \/        \ *
    // *W ________e0________ U*
    // ************************

    if(w == 0u)
        return VertexType::eEdge1;
    else if(uv.u == 0)
        return VertexType::eEdge2;
    else if(uv.v == 0)
        return VertexType::eEdge0;
    else
        return VertexType::eInterior;
}

// Calls the given function to process a value for level `subdivLevel` from
// two vertices in level `subdivLevel - 1`.
// Please call this with the function in the template argument! This will
// (hopefully) allow the compiler to inline the function.
// That function should be of the form
// typedef void (*NewVertexFunction)(BaryUV_uint16 newVtxIdx, BaryUV_uint16 aIdx, BaryUV_uint16 bIdx);
// and should be symmetric with respect to aIdx and bIdx.
// numVtxPerEdge is the number of vertices per edge in the finest level.
// ns is the number of segments between vertices in level `subdivLevel - 1`.
template <class NewVertexFunc>
void applyPerNewVertex(const uint32_t numVtxPerEdge, const uint32_t ns, NewVertexFunc newVertexFunc)
{
    const uint32_t hns = ns >> 1;  // "half num segments"
    // Marco, who originally wrote this code, was very tricky here.
    // This iterates over the new level's values, interpolating
    // from the previous values (marked * in the diagram below,
    // with ns = 2 and hns = 1). It takes one set (marked 0 below),
    // u down, v right:
    // *0*0*0*0*
    //
    // *0*0*0*
    //
    // *0*0*
    //
    // *0*
    //
    // *
    // and gets the remaining points by transposing it (case 1)
    // and shifting it by hns in the v direction (case 2):
    // *0*0*0*0*
    // 12121212
    // *0*0*0*
    // 121212
    // *0*0*
    // 1212
    // *0*
    // 12
    // *
    // Case 0 is interpolated from (u, v) +- (0, hns).
    // Case 1 is interpolated from (u, v) +- (hns, 0).
    // Case 2 is interpolated from (u, v) +- (hns, -hns).
    for(uint32_t u = 0; u < numVtxPerEdge; u += ns)
    {
        for(uint32_t v = hns; v < numVtxPerEdge - u; v += ns)
        {
            for(uint32_t i = 0; i < 3; i++)
            {
                BaryUV_uint16 newVtxIdx{};
                BaryUV_uint16 aIdx{};
                BaryUV_uint16 bIdx{};
                // We need these casts here even if u, v, and hns are already
                // uint16_t, because summing two uint16_ts produces an int due
                // to integer type promotion rules! Alternatively, we could
                // avoid using uniform initialization here.
                switch(i)
                {
                case(0):
                    newVtxIdx = {uint16_t(u), uint16_t(v)};
                    aIdx      = {newVtxIdx.u, uint16_t(newVtxIdx.v + hns)};
                    bIdx      = {newVtxIdx.u, uint16_t(newVtxIdx.v - hns)};
                    break;
                case(1):
                    newVtxIdx = {uint16_t(v), uint16_t(u)};
                    aIdx      = {uint16_t(newVtxIdx.u + hns), newVtxIdx.v};
                    bIdx      = {uint16_t(newVtxIdx.u - hns), newVtxIdx.v};
                    break;
                case(2):
                    newVtxIdx = {uint16_t(u + hns), uint16_t(v)};
                    aIdx      = {uint16_t(newVtxIdx.u + hns), uint16_t(newVtxIdx.v - hns)};
                    bIdx      = {uint16_t(newVtxIdx.u - hns), uint16_t(newVtxIdx.v + hns)};
                    break;
                }
                newVertexFunc(newVtxIdx, aIdx, bIdx);
            }
        }
    }
}

void blockEncode(BlockFormatDispC1 fmtC1, const uint32_t subdivLevel, const uint16_t* DISPENC_RESTRICT reference, Intermediate& DISPENC_RESTRICT result)
{
    const auto& microMeshConfig = getMicromeshConfig(fmtC1);

    const uint32_t numSegments   = (1u << subdivLevel);
    const uint32_t numVtxPerEdge = numSegments + 1;

    // Lossless micromesh type use flat encoding
    if(microMeshConfig.hasFlatEncoding())
    {
        // For lossless micromeshes we just copy the reference data as is
        const auto size = sizeof(uint16_t) * subdivLevelGetVertexCount(subdivLevel);

        memcpy(result.m_decoded, reference, size);
        memcpy(result.m_corrections, reference, size);

        // Add zero shifts
        result.m_shifts = {};
    }
    else
    {
        const uint32_t decoderWordMask = (1u << config_decoderWordSizeInBits) - 1u;

        // Initialize anchors
        uint32_t subdLevel = 0u;
        uint32_t ns        = numSegments;
        // w, u and v anchors addresses
        const uint32_t anchors_addr[3] = {umajorUVtoLinear(0, 0, subdivLevel),   //
                                          umajorUVtoLinear(ns, 0, subdivLevel),  //
                                          umajorUVtoLinear(0, ns, subdivLevel)};
        for(uint32_t i = 0; i < 3; i++)
        {
            result.m_decoded[anchors_addr[i]]     = reference[anchors_addr[i]];
            result.m_corrections[anchors_addr[i]] = reference[anchors_addr[i]];
        }

        // done with anchors, move to the next subdivision level..
        subdLevel++;

        // Hierarchical breadth-first encoder loop
        while(ns > 1)
        {
            const uint32_t numCorrectionBits = microMeshConfig.numCorrBits[subdLevel];

            // 1st pass: compute shifts
            // The 0th and 1st subdiv levels are always lossless (i.e. corrections are 11-bit), so we only compute
            // shifts for subdiv levels 2+.
            Shifts::Level* subd_level_shifts = subdLevel >= 2 ? &result.m_shifts.levels[subdLevel - 2] : nullptr;
            if(subdLevel >= 2)
            {
                uint32_t max_pos_corr[4] = {0, 0, 0, 0};
                uint32_t max_neg_corr[4] = {0, 0, 0, 0};
                applyPerNewVertex(numVtxPerEdge, ns, [&](BaryUV_uint16 newVtxIdx, BaryUV_uint16 aIdx, BaryUV_uint16 bIdx) {
                    // address of the new micro-vertex
                    const uint32_t new_vtx_addr = umajorUVtoLinear(newVtxIdx, subdivLevel);
                    // fetch micro-vertex parents & predict midpoint
                    const uint16_t a          = result.m_decoded[umajorUVtoLinear(aIdx, subdivLevel)];
                    const uint16_t b          = result.m_decoded[umajorUVtoLinear(bIdx, subdivLevel)];
                    const uint16_t prediction = predict(a, b);
                    // Compute the correction without shifts
                    int16_t correction = int16_t(reference[new_vtx_addr]) - int16_t(prediction);
                    // When applying corrections, we use 11-bit wrapping (arithmetic mod 2^11).
                    // This means that sometimes we can use a small negative correction instead of a large positive
                    // correction, or vice versa. For instance, +1792 == 2^11 + (-256), and -1792 == -2^11 + (256).
                    // Smaller corrections allow us to reduce the shifts and thus get more accurate results.
                    // Here, we wrap to [-2^10, 2^10-1].
                    // FIXME: correction can be 2048, in which case this should be 0 instead of +1024
                    if(correction >= (1 << (config_decoderWordSizeInBits - 1)))
                        correction -= (1u << config_decoderWordSizeInBits);
                    else if(correction < -(1 << (config_decoderWordSizeInBits - 1)))
                        correction += (1u << config_decoderWordSizeInBits);
                    const uint32_t abs_corr = (uint32_t)std::abs(correction);  // compute lossless correction

                    const uint32_t vtype = uint32_t(getVertexType(newVtxIdx, numSegments));
                    if(correction >= 0)
                        max_pos_corr[vtype] = abs_corr > max_pos_corr[vtype] ? abs_corr : max_pos_corr[vtype];
                    else
                        max_neg_corr[vtype] = abs_corr > max_neg_corr[vtype] ? abs_corr : max_neg_corr[vtype];
                });

                // Loop over the 4 vertex types
                for(uint32_t i = 0; i < 4; i++)
                {
                    // Here we determine the smallest shift value needed to cover the entire corrections range for this subdivision level & vertex type
                    // We do it by subtracting the number of correction bits available at this level, from the number of bits
                    // needed to represent the largest (in absolute value) lossless correction.
                    uint32_t numBitsNeeded;
                    if(max_pos_corr[i] >= max_neg_corr[i])
                    {
                        // If the "largest" lossless correction is the positive one, to represent it we need just 1 extra bit for the sign
                        // (note: all corrections are sign extended before being left-shifted and accumulated)
                        numBitsNeeded = 1u + getNumBitsNeeded(max_pos_corr[i]);
                    }
                    else
                    {
                        // If the "largest" lossless correction is the negative one, we need an extra bit to represent it only if it is not a power of 2.
                        numBitsNeeded = getNumBitsNeeded(max_neg_corr[i]);
                        numBitsNeeded += (max_neg_corr[i] & (max_neg_corr[i] - 1u)) != 0u ? 1u : 0u;  // add 1 only if not a power of 2.
                    }

                    // Finally compute shift value as the number of bits necessary to left-align the largest correction with its max range
                    // (i.e. ignore the correction LSBs we can't represent)
                    uint32_t shift = (uint32_t)std::max(0, (int32_t)numBitsNeeded - (int32_t)numCorrectionBits);

                    // Determine max shift available for the current micromesh type and subdvision level
                    uint32_t maxShift = 0;
                    using SubdivTable = uint32_t const[];
                    if(fmtC1 == BlockFormatDispC1::eR11_unorm_lvl5_pack1024)
                    {
                        maxShift = SubdivTable{0, 0, 3, 7, 15, 15}[subdLevel];
                    }
                    else if(fmtC1 == BlockFormatDispC1::eR11_unorm_lvl4_pack1024)
                    {
                        maxShift = SubdivTable{0, 0, 0, 1, 7}[subdLevel];
                    }

                    shift                        = std::min(shift, maxShift);
                    subd_level_shifts->vertex[i] = shift;
                }
            }

            // 2nd pass: encode corrections
            applyPerNewVertex(numVtxPerEdge, ns, [&](BaryUV_uint16 newVtxIdx, BaryUV_uint16 aIdx, BaryUV_uint16 bIdx) {
                const uint32_t new_vtx_addr = umajorUVtoLinear(newVtxIdx, subdivLevel);
                const uint16_t a            = result.m_decoded[umajorUVtoLinear(aIdx, subdivLevel)];
                const uint16_t b            = result.m_decoded[umajorUVtoLinear(bIdx, subdivLevel)];
                const uint16_t prediction   = predict(a, b);
                // compute correction & store decoded result
                const uint32_t vtype               = uint32_t(getVertexType(newVtxIdx, numSegments));
                const uint32_t shift               = (subdLevel <= 1) ? 0 : subd_level_shifts->vertex[vtype];
                result.m_corrections[new_vtx_addr] = correct(prediction, reference[new_vtx_addr], shift, numCorrectionBits);
                result.m_decoded[new_vtx_addr] =
                    (int32_t)(decoderWordMask & (prediction + (result.m_corrections[new_vtx_addr] << shift)));
            });

            // Move to the next (finer) subdivision level
            ns = ns >> 1;
            subdLevel++;
        }
    }
}


void blockPackData(BlockFormatDispC1 fmtC1, const uint32_t subdivLevel, const Intermediate& DISPENC_RESTRICT ie, void* DISPENC_RESTRICT encodedData)
{
    const auto& microMeshConfig              = getMicromeshConfig(fmtC1);
    const auto  displacementBlockSizeInBytes = microMeshConfig.blockSizeInBits / 8u;
    uint8_t*    encodedBytes                 = reinterpret_cast<uint8_t*>(encodedData);
    //////////////////////////////////////////////////////
    // Pack shift bits, starting with the 2nd and 3rd subd levels (four 3b values per subdlevel X 2 --> 3 bytes)
    // Note that we don't use the top 2 MSBs - these are reserved for future use in the format.
    const uint32_t numSegments = 1u << subdivLevel;
    uint64_t       shiftGroups = 0;

    // 1024t_512b, 1024t_1024b, 256t_512b
    if(subdivLevel == 5 || (subdivLevel == 4 && displacementBlockSizeInBytes == 64))
    {
        // Subd level 2
        shiftGroups += (uint64_t)ie.m_shifts.levels[0].vertex[int(VertexType::eInterior)] << 54;
        shiftGroups += (uint64_t)ie.m_shifts.levels[0].vertex[int(VertexType::eEdge0)] << 56;
        shiftGroups += (uint64_t)ie.m_shifts.levels[0].vertex[int(VertexType::eEdge1)] << 58;
        shiftGroups += (uint64_t)ie.m_shifts.levels[0].vertex[int(VertexType::eEdge2)] << 60;

        // Subd level 3
        shiftGroups += (uint64_t)ie.m_shifts.levels[1].vertex[int(VertexType::eInterior)] << 42;
        shiftGroups += (uint64_t)ie.m_shifts.levels[1].vertex[int(VertexType::eEdge0)] << 45;
        shiftGroups += (uint64_t)ie.m_shifts.levels[1].vertex[int(VertexType::eEdge1)] << 48;
        shiftGroups += (uint64_t)ie.m_shifts.levels[1].vertex[int(VertexType::eEdge2)] << 51;

        // Subd level 4
        shiftGroups += (uint64_t)ie.m_shifts.levels[2].vertex[int(VertexType::eInterior)] << 26;
        shiftGroups += (uint64_t)ie.m_shifts.levels[2].vertex[int(VertexType::eEdge0)] << 30;
        shiftGroups += (uint64_t)ie.m_shifts.levels[2].vertex[int(VertexType::eEdge1)] << 34;
        shiftGroups += (uint64_t)ie.m_shifts.levels[2].vertex[int(VertexType::eEdge2)] << 38;

        if(subdivLevel == 5 && displacementBlockSizeInBytes == 128)
        {
            // Subd level 5
            shiftGroups += (uint64_t)ie.m_shifts.levels[3].vertex[int(VertexType::eInterior)] << 10;
            shiftGroups += (uint64_t)ie.m_shifts.levels[3].vertex[int(VertexType::eEdge0)] << 14;
            shiftGroups += (uint64_t)ie.m_shifts.levels[3].vertex[int(VertexType::eEdge1)] << 18;
            shiftGroups += (uint64_t)ie.m_shifts.levels[3].vertex[int(VertexType::eEdge2)] << 22;
        }
    }
    // 256t_1024b
    else if(subdivLevel == 4 && displacementBlockSizeInBytes == 128)
    {
        // Subd level 3
        shiftGroups += (uint64_t)ie.m_shifts.levels[1].vertex[int(VertexType::eInterior)] << 58;
        shiftGroups += (uint64_t)ie.m_shifts.levels[1].vertex[int(VertexType::eEdge0)] << 59;
        shiftGroups += (uint64_t)ie.m_shifts.levels[1].vertex[int(VertexType::eEdge1)] << 60;
        shiftGroups += (uint64_t)ie.m_shifts.levels[1].vertex[int(VertexType::eEdge2)] << 61;

        // Subd level 4
        shiftGroups += (uint64_t)ie.m_shifts.levels[2].vertex[int(VertexType::eInterior)] << 46;
        shiftGroups += (uint64_t)ie.m_shifts.levels[2].vertex[int(VertexType::eEdge0)] << 49;
        shiftGroups += (uint64_t)ie.m_shifts.levels[2].vertex[int(VertexType::eEdge1)] << 52;
        shiftGroups += (uint64_t)ie.m_shifts.levels[2].vertex[int(VertexType::eEdge2)] << 55;
    }

    // Can't assume address is aligned to 64b, so we write the shift groups in two 32b steps.
    uint32_t* shiftGroupsAddr = (uint32_t*)(encodedBytes + displacementBlockSizeInBytes - 8u);
    *shiftGroupsAddr++        = shiftGroups & 0xffffffff;
    *shiftGroupsAddr          = shiftGroups >> 32;
    //////////////////////////////////////////////////////

    auto           corrBitsAddr     = microMeshConfig.bary2BlockAddrTable.data();
    auto           corrBitsSize     = microMeshConfig.correctionSizeInBits.data();
    auto           corrections      = ie.m_corrections;
    const uint32_t numVtxPerEdge    = 1u + numSegments;
    const uint32_t configMultiplier = microMeshConfig.numSegments / numSegments;

    for(uint32_t u = 0; u < numVtxPerEdge; u++)
    {
        for(uint32_t v = 0; v < numVtxPerEdge - u; v++)
        {
            // Compute addresses
            const uint32_t addressInTexture = umajorUVtoLinear(u, v, subdivLevel);

            // The table operates in the dimensions of the config
            // which can be greater than the local dimension of the triangle.
            // In that case the coordinates must be scaled accordingly
            const uint32_t addressInTable = umajorUVtoLinear(u * configMultiplier, v * configMultiplier, microMeshConfig.subdiv);

            packedBitWrite(encodedBytes, corrBitsAddr[addressInTable], corrBitsSize[addressInTable],
                           uint32_t(corrections[addressInTexture]));
        }
    }
}

void blockDecode(Format family, BlockFormatDispC1 fmt, const uint32_t subdivLevel, const void* DISPENC_RESTRICT encodedData, uint16_t* DISPENC_RESTRICT decoded)
{
    assert(family == Format::eDispC1_r11_unorm_block);

    // Full block decompression logic

    const uint8_t*         correctionsBlock = reinterpret_cast<const uint8_t*>(encodedData);
    const MicromeshConfig& microMeshConfig  = getMicromeshConfig(fmt);

    //////////////////////////////////////////////////////////
    // First we load & decode the correction shifts
    const uint32_t numSegments = subdivLevelGetSegmentCount(subdivLevel);

    // Load shift groups from the displacemet block
    uint32_t* shiftGroupsAddr = (uint32_t*)(correctionsBlock + (microMeshConfig.blockSizeInBits / 8) - 8u);
    uint64_t  shiftGroups     = *shiftGroupsAddr++;
    shiftGroups += (uint64_t)(*shiftGroupsAddr) << 32;

    // Allocate and initialize shifts
    // Technically speaking 1st and 2nd subd levels have no shifts because they are always lossless..
    Shifts shifts;
    memset(&shifts, 0, sizeof(Shifts));

    // 1024t_512b, 1024t_1024b, 256t_512b
    if(subdivLevel == 5 || (subdivLevel == 4 && (microMeshConfig.blockSizeInBits == 512)))
    {
        // Subd level 2
        shifts.levels[0].vertex[int(VertexType::eInterior)] = (shiftGroups >> 54) & 0x3;
        shifts.levels[0].vertex[int(VertexType::eEdge0)]    = (shiftGroups >> 56) & 0x3;
        shifts.levels[0].vertex[int(VertexType::eEdge1)]    = (shiftGroups >> 58) & 0x3;
        shifts.levels[0].vertex[int(VertexType::eEdge2)]    = (shiftGroups >> 60) & 0x3;

        // Subd level 3
        shifts.levels[1].vertex[int(VertexType::eInterior)] = (shiftGroups >> 42) & 0x7;
        shifts.levels[1].vertex[int(VertexType::eEdge0)]    = (shiftGroups >> 45) & 0x7;
        shifts.levels[1].vertex[int(VertexType::eEdge1)]    = (shiftGroups >> 48) & 0x7;
        shifts.levels[1].vertex[int(VertexType::eEdge2)]    = (shiftGroups >> 51) & 0x7;

        // Subd level 4
        shifts.levels[2].vertex[int(VertexType::eInterior)] = (shiftGroups >> 26) & 0xf;
        shifts.levels[2].vertex[int(VertexType::eEdge0)]    = (shiftGroups >> 30) & 0xf;
        shifts.levels[2].vertex[int(VertexType::eEdge1)]    = (shiftGroups >> 34) & 0xf;
        shifts.levels[2].vertex[int(VertexType::eEdge2)]    = (shiftGroups >> 38) & 0xf;

        if(subdivLevel == 5 && microMeshConfig.blockSizeInBits == 1024)
        {
            // Subd level 5
            shifts.levels[3].vertex[int(VertexType::eInterior)] = (shiftGroups >> 10) & 0xf;
            shifts.levels[3].vertex[int(VertexType::eEdge0)]    = (shiftGroups >> 14) & 0xf;
            shifts.levels[3].vertex[int(VertexType::eEdge1)]    = (shiftGroups >> 18) & 0xf;
            shifts.levels[3].vertex[int(VertexType::eEdge2)]    = (shiftGroups >> 22) & 0xf;
        }
    }
    // 256t_1024b
    else if(subdivLevel == 4 && microMeshConfig.blockSizeInBits == 1024)
    {
        // Subd level 3
        shifts.levels[1].vertex[int(VertexType::eInterior)] = (shiftGroups >> 58) & 0x1;
        shifts.levels[1].vertex[int(VertexType::eEdge0)]    = (shiftGroups >> 59) & 0x1;
        shifts.levels[1].vertex[int(VertexType::eEdge1)]    = (shiftGroups >> 60) & 0x1;
        shifts.levels[1].vertex[int(VertexType::eEdge2)]    = (shiftGroups >> 61) & 0x1;

        // Subd level 4
        shifts.levels[2].vertex[int(VertexType::eInterior)] = (shiftGroups >> 46) & 0x7;
        shifts.levels[2].vertex[int(VertexType::eEdge0)]    = (shiftGroups >> 49) & 0x7;
        shifts.levels[2].vertex[int(VertexType::eEdge1)]    = (shiftGroups >> 52) & 0x7;
        shifts.levels[2].vertex[int(VertexType::eEdge2)]    = (shiftGroups >> 55) & 0x7;
    }
    //////////////////////////////////////////////////////////

    const uint32_t decoderWordMask = (1u << config_decoderWordSizeInBits) - 1u;
    const uint8_t* corrections     = correctionsBlock;

    auto           corrTableBitAddr = microMeshConfig.bary2BlockAddrTable.data();
    auto           numCorrBits      = microMeshConfig.numCorrBits.data();
    const uint32_t configMultiplier = microMeshConfig.numSegments / numSegments;

    // Initialize anchors
    uint32_t       thisSubdLevel = 0u;
    uint16_t       ns            = numSegments;
    const uint32_t numVtxPerEdge = 1u + numSegments;

    if(microMeshConfig.hasFlatEncoding())
    {
        for(uint32_t u = 0; u < numVtxPerEdge; u++)
        {
            for(uint32_t v = 0; v < numVtxPerEdge - u; v++)
            {
                const uint32_t new_vtx_addr = umajorUVtoLinear(u, v, subdivLevel);
                const uint32_t new_vtx_table_index =
                    umajorUVtoLinear(u * configMultiplier, v * configMultiplier, microMeshConfig.subdiv);
                decoded[new_vtx_addr] =
                    readUpTo16Bits(corrections, corrTableBitAddr[new_vtx_table_index], config_decoderWordSizeInBits, false);
            }
        }
    }
    else
    {
        const uint32_t      numAnchorBits    = numCorrBits[0];
        const BaryUV_uint16 anchors_barys[3] = {{0, 0}, {ns, 0}, {0, ns}};  // w, u and v anchors barys as (u,v) pairs
        for(uint32_t i = 0; i < 3; i++)
        {
            uint32_t anchor_addr        = umajorUVtoLinear(anchors_barys[i].u, anchors_barys[i].v, subdivLevel);
            uint32_t anchor_table_index = umajorUVtoLinear(anchors_barys[i].u * configMultiplier,
                                                           anchors_barys[i].v * configMultiplier, microMeshConfig.subdiv);
            decoded[anchor_addr] = readUpTo16Bits(correctionsBlock, corrTableBitAddr[anchor_table_index], numAnchorBits, false);
        }

        // done with anchors, move to the next level..
        thisSubdLevel++;

        // Hierarchical decoder loop
        while(ns > 1)
        {
            const uint32_t       numBitsToDecode = numCorrBits[thisSubdLevel];
            const Shifts::Level* shiftLevel      = (thisSubdLevel >= 2) ? &shifts.levels[thisSubdLevel - 2] : nullptr;

            applyPerNewVertex(numVtxPerEdge, ns, [&](BaryUV_uint16 newVtxIdx, BaryUV_uint16 aIdx, BaryUV_uint16 bIdx) {
                // address of the new micro-vertex
                const uint32_t new_vtx_addr = umajorUVtoLinear(newVtxIdx, subdivLevel);
                const uint32_t new_vtx_table_index =
                    umajorUVtoLinear(newVtxIdx.u * configMultiplier, newVtxIdx.v * configMultiplier, microMeshConfig.subdiv);
                // fetch micro-vertex parents & predict midpoint
                const uint16_t a          = decoded[umajorUVtoLinear(aIdx, subdivLevel)];
                const uint16_t b          = decoded[umajorUVtoLinear(bIdx, subdivLevel)];
                const uint16_t prediction = predict(a, b);
                const int32_t correction = readUpTo16Bits(corrections, corrTableBitAddr[new_vtx_table_index], numBitsToDecode);
                const uint32_t vtype  = uint32_t(getVertexType(newVtxIdx, numSegments));
                const uint32_t shift  = (thisSubdLevel <= 1) ? 0 : shiftLevel->vertex[vtype];
                decoded[new_vtx_addr] = (int32_t)(decoderWordMask & (prediction + (correction << shift)));
            });

            // Move to the next (finer) subdivision level
            ns = ns >> 1;
            thisSubdLevel++;
        }
    }
}

}  // namespace dispenc
}  // namespace micromesh
