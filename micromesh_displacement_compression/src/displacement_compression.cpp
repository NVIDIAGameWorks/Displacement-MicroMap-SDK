//
// Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstring>

#include "displacement_block_codec.h"
#include "displacement_mesh_codec.h"
#include <micromesh_internal/micromesh_context.h>
#include <micromesh/micromesh_displacement_compression.h>
#include <micromesh/micromesh_utils.h>

namespace micromesh
{
// Regardless of the format, don't allow subdivLevels where
// 1 << subdivLevel would be >= 32768 to avoid integer overflow.
#define CHECK_MAXSUBDIV(messageCallbackInfo, subdivLevel)                                                              \
    {                                                                                                                  \
        if((subdivLevel) >= MICROMESH_MAX_SUBDIV_LEVEL)                                                                \
        {                                                                                                              \
            MLOGE(messageCallbackInfo, #subdivLevel " (%u) was too large! That would correspond to 2^%u triangles.",   \
                  (subdivLevel), (subdivLevel));                                                                       \
            return Result::eInvalidValue;                                                                              \
        }                                                                                                              \
    }

static void meshEncoderPtrDeleter(void* payload)
{
    delete reinterpret_cast<dispenc::MeshEncoder*>(payload);
}

MICROMESH_API Result MICROMESH_CALL micromeshOpCompressDisplacementBegin(OpContext                              ctx,
                                                                         const OpCompressDisplacement_settings* settings,
                                                                         const OpCompressDisplacement_input* inputUncompressed,
                                                                         OpCompressDisplacement_output* outputCompressed)
{
    // Check API preconditions.
    {
        CHECK_CTX_NONNULL(ctx);
        CHECK_CTX_BEGIN(ctx);
        CHECK_NONNULL(ctx, settings);
        CHECK_NONNULL(ctx, inputUncompressed);
        CHECK_NONNULL(ctx, inputUncompressed->data);
        if(!micromapIsValid(*inputUncompressed->data))
        {
            LOGE(ctx, "inputUncompressed->data was not a valid uncompressed micromap.");
            return Result::eInvalidValue;
        }
        if(inputUncompressed->data->values.format != Format::eR11_unorm_pack16)
        {
            LOGE(ctx, "inputUncompressed->data->values.format must be eR11_unorm_pack16.");
            return Result::eInvalidFormat;
        }
        if(inputUncompressed->data->frequency != Frequency::ePerMicroVertex)
        {
            LOGE(ctx, "inputUncompressed->data->frequency must be ePerMicroVertex.");
            return Result::eInvalidFrequency;
        }
        CHECK_NONNULL(ctx, inputUncompressed->topology);
        if(!meshtopoIsValid(*inputUncompressed->topology))
        {
            LOGE(ctx, "inputUncompressed->topology was invalid.");
            return Result::eInvalidValue;
        }
        if(inputUncompressed->perVertexImportance.data)
        {
            CHECK_ARRAYVALIDTYPED(ctx, inputUncompressed->perVertexImportance);
        }
        if(inputUncompressed->meshTriangleMappings.data)
        {
            CHECK_ARRAYVALIDTYPED(ctx, inputUncompressed->meshTriangleMappings);
        }
        if(inputUncompressed->compressedFormatFamily != Format::eDispC1_r11_unorm_block)
        {
            LOGE(ctx, "inputUncompressed->compressedFormatFamily must be Format::eDispC1_r11_unorm_block.");
            return Result::eInvalidFormat;
        }
        CHECK_MAXSUBDIV(&ctx->m_messageCallbackInfo, inputUncompressed->data->maxSubdivLevel);

        CHECK_NONNULL(ctx, outputCompressed);
    }

    // Encode it; see if it succeeded.
    std::unique_ptr<dispenc::MeshEncoder> meshEncoder = std::make_unique<dispenc::MeshEncoder>();
    Result                                result      = meshEncoder->batchEncode(ctx, settings, inputUncompressed);
    if(result != Result::eSuccess)
    {
        return result;
    }

    // Success! Set the output sizes and store the mesh encoder inside the
    // context payload so we can retrieve it in End().
    MicromapCompressed* compressed = outputCompressed->compressed;
    meshEncoder->fillEncodedSizes(*compressed);

    // take existing valueFloatExpansion from uncompressed
    compressed->valueFloatExpansion = inputUncompressed->data->valueFloatExpansion;
    if(outputCompressed->mipData)
    {
        outputCompressed->mipData->valueFloatExpansion = inputUncompressed->data->valueFloatExpansion;
    }

    // Well, we still might have one more issue if the encoded data is too
    // large. Technically, a MicromapCompressed representing exactly 2^32 bytes
    // could be possible, but such a situation is likely to break other things.
    if(compressed->values.count > std::numeric_limits<uint32_t>::max())
    {
        LOGE(ctx,
             "Writing the compressed data would require %" PRIu64
             " bytes, which is larger than the maximum of 2^32-1 from "
             "the type of outputCompressed->compressed->triangleValueByteOffsets.",
             compressed->values.count);
        return Result::eFailure;
    }

    ctx->setPayload(meshEncoder.release(), meshEncoderPtrDeleter);
    ctx->setNextSequenceFn(micromeshOpCompressDisplacementEnd);

    return Result::eSuccess;
}

MICROMESH_API Result MICROMESH_CALL micromeshOpCompressDisplacementEnd(OpContext ctx, OpCompressDisplacement_output* outputCompressed)
{
    // Check API preconditions.
    {
        CHECK_CTX_NONNULL(ctx);
        CHECK_CTX_END(ctx, micromeshOpCompressDisplacementEnd);
        CHECK_NONNULL(ctx, outputCompressed);
        CHECK_NONNULL(ctx, outputCompressed->compressed);
        if(!micromapIsValid(*outputCompressed->compressed))
        {
            LOGE(ctx, "outputCompressed->compressed was not set up correctly to be a compressed micromap.");
        }
        // We assume the app hasn't modified the fields of MicromapCompressed
        // we set in micromeshOpCompressDisplacementBegin() so that they're now
        // invalid; that would be a strange thing for the app to do because it
        // would likely require extra programming work.
        CHECK_NONNULL(ctx, outputCompressed->compressed);
        CHECK_ARRAYVALID(ctx, outputCompressed->compressed->values);
        CHECK_ARRAYVALIDTYPED(ctx, outputCompressed->compressed->triangleSubdivLevels);
        CHECK_ARRAYVALIDTYPED(ctx, outputCompressed->compressed->triangleValueByteOffsets);
        CHECK_ARRAYVALIDTYPED(ctx, outputCompressed->compressed->triangleBlockFormats);
        const MicromapCompressed& compressed = *outputCompressed->compressed;
        // Also validate `triangleMinMaxs` and `mipData` at this time if
        // they're set.
        if(outputCompressed->triangleMinMaxs.data)
        {
            CHECK_ARRAYVALID(ctx, outputCompressed->triangleMinMaxs);
            if(outputCompressed->triangleMinMaxs.format != Format::eR11_unorm_pack16)
            {
                LOGE(ctx, "outputCompressed->triangleMinMaxs.format must be Format::eR11_unorm_pack16.");
                return Result::eInvalidFormat;
            }
            if(outputCompressed->triangleMinMaxs.byteStride != 2)
            {
                LOGE(ctx, "outputCompressed->triangleMinMaxs.byteStride must be 2.");
                return Result::eInvalidFormat;
            }
            if(outputCompressed->triangleMinMaxs.count != outputCompressed->compressed->triangleSubdivLevels.count * 2)
            {
                LOGE(ctx,
                     "There must be 2 elements of outputCompressed->triangleMinMaxs for each triangle, but there were "
                     "%" PRIu64 " triangles and %" PRIu64 " elements in outputCompressed->triangleMinMaxs.",
                     outputCompressed->compressed->triangleSubdivLevels.count, outputCompressed->triangleMinMaxs.count);
                return Result::eInvalidRange;
            }
        }
        if(outputCompressed->mipData)
        {
            const MicromapPacked& mipData = *outputCompressed->mipData;
            if(!micromapIsValid(mipData))
            {
                LOGE(ctx, "outputCompressed->mipData was not set up correctly to be a packed micromap.");
                return Result::eInvalidValue;
            }
            if(mipData.triangleSubdivLevels.count != compressed.triangleSubdivLevels.count)
            {
                LOGE(ctx,
                     "outputCompressed->mipData->triangleSubdivLevels.count was %" PRIu64
                     ", but outputCompressed->compressed->triangleSubdivLevels.count was %" PRIu64
                     ". These must be the same.",
                     mipData.triangleSubdivLevels.count, compressed.triangleSubdivLevels.count);
            }
            // TODO: More validation
        }
    }

    // Retrieve the MeshEncoder payload and extract the packed blocks.
    dispenc::MeshEncoder* meshEncoder = reinterpret_cast<dispenc::MeshEncoder*>(ctx->m_opPayload);
    assert(meshEncoder != nullptr);  // Should never happen

    meshEncoder->writeCompressedData(ctx, *outputCompressed);

    ctx->resetSequence();

    return Result::eSuccess;
}

MICROMESH_API Result MICROMESH_CALL micromeshOpDecompressDisplacementBegin(OpContext                 ctx,
                                                                           const MicromapCompressed* inputCompressed,
                                                                           Micromap*                 outputDecompressed)
{
    // Check API preconditions.
    {
        CHECK_CTX_NONNULL(ctx);
        CHECK_CTX_BEGIN(ctx);
        CHECK_NONNULL(ctx, inputCompressed);
        CHECK_NONNULL(ctx, outputDecompressed);
        CHECK_ARRAYVALIDTYPED(ctx, inputCompressed->triangleBlockFormats);
        CHECK_ARRAYVALIDTYPED(ctx, inputCompressed->triangleSubdivLevels);
        CHECK_ARRAYVALIDTYPED(ctx, inputCompressed->triangleValueByteOffsets);
        CHECK_ARRAYVALID(ctx, inputCompressed->values);

        if(inputCompressed->values.format != Format::eDispC1_r11_unorm_block)
        {
            LOGE(ctx,
                 "inputCompressed->values.format must be the format for displacement compression family 1, "
                 "eDispC1_r11_unorm_block; it was set to %u.",
                 uint32_t(inputCompressed->values.format));
            return Result::eInvalidFormat;
        }

        if(!micromapLayoutIsValid(outputDecompressed->layout))
        {
            LOGE(ctx, "outputDecompressed->layout does not have function pointers");
            return Result::eInvalidLayout;
        }
    }

    outputDecompressed->frequency           = Frequency::ePerMicroVertex;
    outputDecompressed->valueFloatExpansion = inputCompressed->valueFloatExpansion;
    // Set the fields of `ArrayInfo` objects we're allowed to set.
    outputDecompressed->values.format                   = Format::eR11_unorm_pack16;
    outputDecompressed->values.count                    = 0;
    outputDecompressed->triangleSubdivLevels.count      = inputCompressed->triangleSubdivLevels.count;
    outputDecompressed->triangleValueIndexOffsets.count = inputCompressed->triangleValueByteOffsets.count;
    // We assume min/maxSubdivLevel is set correctly.
    outputDecompressed->minSubdivLevel = inputCompressed->minSubdivLevel;
    outputDecompressed->maxSubdivLevel = inputCompressed->maxSubdivLevel;

    // Iterate over `triangleSubdivLevels` to determine how many values we'll
    // write in total. Writing more than 2^32-1 values would likely require an
    // out-of-bounds triangleValueIndexOffset, so we treat that as an error.
    // This could also be implemented as a parallel sum.
    {
        uint64_t numValues = 0;
        for(uint64_t i = 0; i < inputCompressed->triangleSubdivLevels.count; i++)
        {
            const uint16_t subdivLevel = arrayGetV<uint16_t>(inputCompressed->triangleSubdivLevels, i);
            CHECK_MAXSUBDIV(&ctx->m_messageCallbackInfo, subdivLevel);
            numValues += subdivLevelGetVertexCount(subdivLevel);
        }
        if(numValues > std::numeric_limits<uint32_t>::max())
        {
            LOGE(ctx, "This MicromapCompressed represents %" PRIu64 " decompressed values, which is more than 2^32-1.", numValues);
            return Result::eInvalidValue;
        }
        outputDecompressed->values.count = numValues;
    }

    // Store the payload - in this case, the inputCompressed pointer.
    ctx->setPayload(const_cast<MicromapCompressed*>(inputCompressed), nullptr);
    ctx->setNextSequenceFn(micromeshOpDecompressDisplacementEnd);

    return Result::eSuccess;
}

// When we copy subtriangles into the full triangle during decompression,
// like this:
// w-+-+-+-+ > v
// |/|/|/|/
// +-+-+-+
// |/|/|/
// +-+-+
// |/|/
// +-+
// |/
// +
//
// v
// u
// we need to know the location and orientation of each subtriangle.
// There's six possibilities here - so we don't optimize too much, and only
// store the vertices of each triangle in a grid such that +1 == moving by 1
// subtriangle edge.
struct SubtrianglePositionInfo
{
    BaryUV_uint16 w;
    BaryUV_uint16 u;
    BaryUV_uint16 v;
};


MICROMESH_API Result MICROMESH_CALL micromeshOpDecompressDisplacementEnd(OpContext ctx, Micromap* outputDecompressed)
{
    // Check API preconditions.
    {
        CHECK_CTX_NONNULL(ctx);
        CHECK_CTX_END(ctx, micromeshOpDecompressDisplacementEnd);
        CHECK_NONNULL(ctx, outputDecompressed);
        // Make sure we have fully valid arrays at this point
        CHECK_ARRAYVALID(ctx, outputDecompressed->values);
        CHECK_ARRAYVALIDTYPED(ctx, outputDecompressed->triangleSubdivLevels);
        CHECK_ARRAYVALIDTYPED(ctx, outputDecompressed->triangleValueIndexOffsets);
    }

    // Retrieve the payload, which is the const MicromapCompressed* pointer
    // from Begin.
    const MicromapCompressed* inputCompressed = reinterpret_cast<MicromapCompressed*>(ctx->m_opPayload);
    if(inputCompressed == nullptr)
    {
        // This should never happen
        LOGE(ctx, "The context payload disappeared before calling micromeshOpDecompressDisplacementEnd!");
        return Result::eFailure;
    }

    // The output subdivision levels are the same as the input subdivision levels
    if(outputDecompressed->triangleSubdivLevels.data != inputCompressed->triangleSubdivLevels.data)
    {
        ctx->arrayCopy<uint16_t>(outputDecompressed->triangleSubdivLevels, inputCompressed->triangleSubdivLevels);
    }
    else
    {
        if(!arrayIsEqual(outputDecompressed->triangleSubdivLevels, inputCompressed->triangleSubdivLevels))
        {
            LOGE(ctx,
                 "The input and output triangleSubdivLevels had the same .data pointer, but their ArrayInfo objects "
                 "were not equal. Copying from an array to itself while changing its format, stride, or count is not "
                 "supported.");
            return Result::eFailure;
        }
    }

    // Compute triangleValueIndexOffsets - this could also be a parallel prefix sum:
    {
        uint64_t nextOffset = 0;
        for(uint64_t i = 0; i < inputCompressed->triangleSubdivLevels.count; i++)
        {
            // Begin() checks that this never overflows a uint32_t:
            arraySetV<uint32_t>(outputDecompressed->triangleValueIndexOffsets, i, uint32_t(nextOffset));
            nextOffset += subdivLevelGetVertexCount(arrayGetV<uint16_t>(inputCompressed->triangleSubdivLevels, i));
        }
    }

    // Create tables for locating subtriangles at different subdivision levels.
    // We probably generate more than we need here.
    container::vector<container::vector<SubtrianglePositionInfo>> subtrianglePositionIndex(
        1 + std::max(0, int32_t(outputDecompressed->maxSubdivLevel) - 3));
    {
        // We kind of use a breadth-first-search-like approach here, building
        // up each level from the previous, using the bird curve as implemented
        // in MeshEncoder::Triangle::Triangle. There may be more clever ways
        // of doing this!
        SubtrianglePositionInfo tri{};
        tri.w = makeBaryUV_uint16(0, 0);
        tri.u = makeBaryUV_uint16(1, 0);
        tri.v = makeBaryUV_uint16(0, 1);
        subtrianglePositionIndex[0].push_back(tri);
        for(size_t n = 1; n < subtrianglePositionIndex.size(); n++)
        {
            // Build level n from n-1.
            container::vector<SubtrianglePositionInfo>& newLevel = subtrianglePositionIndex[n];
            for(SubtrianglePositionInfo srcTri : subtrianglePositionIndex[n - 1])
            {
                // Double its coordinates so we can subdivide it
                srcTri.u.u *= 2;
                srcTri.u.v *= 2;
                srcTri.v.u *= 2;
                srcTri.v.v *= 2;
                srcTri.w.u *= 2;
                srcTri.w.v *= 2;
                // Construct midpoints
                const BaryUV_uint16 uw = makeBaryUV_uint16((srcTri.w.u + srcTri.u.u) / 2, (srcTri.w.v + srcTri.u.v) / 2);
                const BaryUV_uint16 uv = makeBaryUV_uint16((srcTri.u.u + srcTri.v.u) / 2, (srcTri.u.v + srcTri.v.v) / 2);
                const BaryUV_uint16 vw = makeBaryUV_uint16((srcTri.v.u + srcTri.w.u) / 2, (srcTri.v.v + srcTri.w.v) / 2);
                // Build the four child triangles
                SubtrianglePositionInfo tri0;
                tri0.w = srcTri.w;
                tri0.u = uw;
                tri0.v = vw;
                newLevel.push_back(tri0);
                SubtrianglePositionInfo tri1;
                tri1.w = vw;
                tri1.u = uv;
                tri1.v = uw;
                newLevel.push_back(tri1);
                SubtrianglePositionInfo tri2;
                tri2.w = uw;
                tri2.u = srcTri.u;
                tri2.v = uv;
                newLevel.push_back(tri2);
                SubtrianglePositionInfo tri3;
                tri3.w = uv;
                tri3.u = vw;
                tri3.v = srcTri.v;
                newLevel.push_back(tri3);
            }
        }
    }

    // Finally, decompress the value data in parallel. Each micromap triangle
    // is one work item. Because the subtriangles are listed in bird curve
    // order, this means that the value data as a whole should be in bird curve
    // order.
    // Note that this is a good target for optimization, since it'll be a hot
    // path in Omniverse! One route would be monomorphization - then we can
    // avoid the heap allocations here and likely reduce the number of memory
    // loads when unpacking.
    ctx->parallel_items(inputCompressed->triangleSubdivLevels.count, [&](uint64_t micromapTriangle, uint32_t, void*) {
        const uint16_t micromapTriangleSubdivLevel = arrayGetV<uint16_t>(inputCompressed->triangleSubdivLevels, micromapTriangle);
        const BlockFormatDispC1 subtriFmt =
            static_cast<BlockFormatDispC1>(arrayGetV<uint16_t>(inputCompressed->triangleBlockFormats, micromapTriangle));
        FormatInfo subtriFmtInfo{};
        // If the format's unknown, we don't write anything
        if(Result::eSuccess != micromeshBlockFormatDispC1GetInfo(subtriFmt, &subtriFmtInfo))
        {
            return;
        }
        // Also get the subdivision level. TODO: Turn this into a helper function!
        uint16_t subtriSubdivLevel;
        switch(subtriFmt)
        {
        case BlockFormatDispC1::eR11_unorm_lvl3_pack512:
            subtriSubdivLevel = 3;
            break;
        case BlockFormatDispC1::eR11_unorm_lvl4_pack1024:
            subtriSubdivLevel = 4;
            break;
        case BlockFormatDispC1::eR11_unorm_lvl5_pack1024:
            subtriSubdivLevel = 5;
            break;
        case BlockFormatDispC1::eInvalid:
        default:
            return;
        }
        // How many subtriangles are there? For this, we make use of uniform
        // block formats. This could also be encapsulated into a helper function.
        const uint64_t numSubtriangles = ((micromapTriangleSubdivLevel <= subtriSubdivLevel) ?
                                              1 :
                                              1ULL << (2 * (micromapTriangleSubdivLevel - subtriSubdivLevel)));
        const uint32_t subtriSegments  = subdivLevelGetSegmentCount(subtriSubdivLevel);

        const uint32_t triCompressedByteOffset = arrayGetV<uint32_t>(inputCompressed->triangleValueByteOffsets, micromapTriangle);
        const uint32_t outputValueOffset = arrayGetV<uint32_t>(outputDecompressed->triangleValueIndexOffsets, micromapTriangle);

        // What level are we decoding to? Normally this is the block level, but
        // if the micromap triangle level is below 3, we need to specify that.
        const uint32_t decodeSubdivLevel = std::min(subtriSubdivLevel, micromapTriangleSubdivLevel);
        const uint32_t decodeSegments    = subdivLevelGetSegmentCount(decodeSubdivLevel);
        const uint64_t decodeValueCount  = subdivLevelGetVertexCount(decodeSubdivLevel);
        // Allocate a buffer we'll reuse for unpacked u-major data.
        container::vector<uint16_t> unpackedUMajor(decodeValueCount);

        for(uint64_t subtri = 0; subtri < numSubtriangles; subtri++)
        {
            // Get the packed block data (1 subtriangle == 1 block)
            const uint8_t* packedBlock = reinterpret_cast<const uint8_t*>(inputCompressed->values.data)
                                         + triCompressedByteOffset + subtri * subtriFmtInfo.byteSize;
            // Unpack it
            // TODO: MicromapCompressed needs a block format family member!
            dispenc::blockDecode(Format::eDispC1_r11_unorm_block, subtriFmt, decodeSubdivLevel, packedBlock,
                                 unpackedUMajor.data());
            // Get the location of the w=(0,0) vertex of the subtriangle, as
            // well as how stepping in the subtriangle +u and +v directions
            // moves within the micromap triangle.
            const SubtrianglePositionInfo& spi =
                subtrianglePositionIndex[std::max(0, int32_t(micromapTriangleSubdivLevel - subtriSubdivLevel))][subtri];
            const std::array<int32_t, 2> wOffset{int32_t(spi.w.u * subtriSegments), int32_t(spi.w.v * subtriSegments)};
            const std::array<int32_t, 2> triUStep{int32_t(spi.u.u - spi.w.u), int32_t(spi.u.v - spi.w.v)};
            const std::array<int32_t, 2> triVStep{int32_t(spi.v.u - spi.w.u), int32_t(spi.v.v - spi.w.v)};

            // Iterate over the unpacked block in u-major order.
            // If the micromap triangle uses a subdiv level less than 3, w
            // must make sure we don't index out of bounds.
            for(uint32_t blockU = 0; blockU <= decodeSegments; blockU++)
            {
                for(uint32_t blockV = 0; blockV <= decodeSegments - blockU; blockV++)
                {
                    // Read the unpacked data
                    const uint32_t valueIdx = umajorUVtoLinear(blockU, blockV, decodeSubdivLevel);
                    const uint16_t value    = unpackedUMajor[valueIdx];
                    // Figure out where to write to
                    const uint32_t fullTriangleIdx = outputDecompressed->layout.pfnGetMicroVertexIndex(
                        wOffset[0] + blockU * triUStep[0] + blockV * triVStep[0],
                        wOffset[1] + blockU * triUStep[1] + blockV * triVStep[1], micromapTriangleSubdivLevel,
                        outputDecompressed->layout.userData);
                    // And copy the data!
                    arraySetV<uint16_t>(outputDecompressed->values, outputValueOffset + fullTriangleIdx, value);
                }
            }
        }
    });

    ctx->resetSequence();

    return Result::eSuccess;
}

// Common validation for displacement block functions.
// Assumes settings != nullptr.
Result validateDisplacementBlock(const DisplacementBlock_settings* settings,
                                 const void*                       decompressed,
                                 const void*                       compressed,
                                 const MessageCallbackInfo*        messageCallbackInfo)
{
    if(settings->decompressedFormat != Format::eR11_unorm_pack16)
    {
        MLOGE(messageCallbackInfo, "settings->decompressedFormat (%s) must be eR11_unorm_pack16.",
              micromeshGetFormatString(settings->decompressedFormat));
        return Result::eInvalidFormat;
    }

    CHECK_MAXSUBDIV(messageCallbackInfo, settings->subdivLevel);
    const size_t numValues = subdivLevelGetVertexCount(settings->subdivLevel);

    uintptr_t compressedBytes = 0;

    if(settings->compressedFormat == Format::eDispC1_r11_unorm_block)
    {
        switch(settings->compressedBlockFormatDispC1)
        {
            // NOTE: The BlockFormatDispC1 -> maximum subdivision level logic
            // appears in more than one place; it could be refactored to only
            // one place.
        case BlockFormatDispC1::eR11_unorm_lvl3_pack512:
            if(settings->subdivLevel > 3)
            {
                MLOGE(messageCallbackInfo,
                      "settings->subdivLevel (%u) must be 0, 1, 2, or 3, since the compressed format was "
                      "eR11_unorm_lvl3_pack512..",
                      settings->subdivLevel);
                return Result::eInvalidValue;
            }
            break;
        case BlockFormatDispC1::eR11_unorm_lvl4_pack1024:
            if(settings->subdivLevel != 4)
            {
                MLOGE(messageCallbackInfo,
                      "settings->subdivLevel (%u) must be 4, since the compressed format was eR11_unorm_lvl4_pack1024.",
                      settings->subdivLevel);
                return Result::eInvalidValue;
            }
            break;
        case BlockFormatDispC1::eR11_unorm_lvl5_pack1024:
            if(settings->subdivLevel != 5)
            {
                MLOGE(messageCallbackInfo,
                      "settings->subdivLevel (%u) must be 5, since the compressed format was eR11_unorm_lvl5_pack1024.",
                      settings->subdivLevel);
                return Result::eInvalidValue;
            }
            break;
        case BlockFormatDispC1::eInvalid:
        default:
            MLOGE(messageCallbackInfo,
                  "settings->compressedBlockFormatDispC1 (%" PRIu16 ") was invalid for compressedFormat %s.",
                  static_cast<uint16_t>(settings->compressedBlockFormatDispC1),
                  micromeshGetFormatString(settings->compressedFormat));
            return Result::eInvalidBlockFormat;
            break;
        }

        micromesh::FormatInfo fmtInfo{};
        (void)micromeshBlockFormatDispC1GetInfo(settings->compressedBlockFormatDispC1, &fmtInfo);
        compressedBytes = fmtInfo.byteSize;
    }
    else
    {
        MLOGE(messageCallbackInfo, "settings->compressedFormat (%s) must be eDispC1_r11_unorm_block.",
              micromeshGetFormatString(settings->compressedFormat));
        return Result::eInvalidFormat;
    }

    micromesh::FormatInfo decompressedFmtInfo{};
    (void)micromeshFormatGetInfo(settings->decompressedFormat, &decompressedFmtInfo);
    const uintptr_t decompressedBytes = decompressedFmtInfo.byteSize * numValues;

    // Make sure the input and output pointers don't overlap so __restrict
    // optimizations later on are correct
    const uintptr_t decompressedStart = reinterpret_cast<uintptr_t>(decompressed);
    const uintptr_t compressedStart   = reinterpret_cast<uintptr_t>(compressed);
    bool            pointersValid = (decompressedStart < UINTPTR_MAX - decompressedBytes) && (compressedStart < UINTPTR_MAX - compressedBytes);
    if(pointersValid)
    {
        if(compressedStart <= decompressedStart)
        {
            pointersValid = (compressedStart + compressedBytes < decompressedStart);
        }
        else
        {
            pointersValid = (decompressedStart + decompressedBytes < compressedStart);
        }
    }
    if(!pointersValid)
    {
        MLOGE(messageCallbackInfo,
              "Compressed and decompressed ranges overlapped: compressed values are %" PRIxPTR
              " bytes starting at %p; decompressed values are %" PRIxPTR " bytes starting at %p.",
              compressedBytes, compressed, decompressedBytes, decompressed);
        return Result::eInvalidRange;
    }

    return Result::eSuccess;
}

// Determine whether the layout is u-major. This should allow the compiler to
// inline u-major function calls with link-time optimization. However, it
// can't detect if the user provided a custom function equivalent to the
// u-major layout.
inline bool isLayoutUMajor(const MicromapLayout& layout)
{
    return micromeshLayoutGetStandardType(&layout) == StandardLayoutType::eUmajor;
}

MICROMESH_API uint64_t MICROMESH_CALL micromeshGetDisplacementBlockScratchSize(const DisplacementBlock_settings* settings)
{
    const uint32_t numValues = subdivLevelGetVertexCount(settings->subdivLevel);
    // in theory only * 2 if we have umajor, but let's keep it simple
    return numValues * sizeof(uint16_t) * 3;
}

template <class T>
void bitCopyLayoutTToUmaj16(const T* input, uint16_t* umaj, const MicromapLayout& layout, const uint32_t subdivLevel)
{
    const uint32_t numValues = subdivLevelGetVertexCount(subdivLevel);
    if(layout.pfnGetMicroVertexIndex)
    {
        // NOTE: Is it faster if we iterate over (u, v)?
        for(uint32_t umajIdx = 0; umajIdx < numValues; umajIdx++)
        {
            const BaryUV_uint16 uv       = umajorLinearToUV(umajIdx, subdivLevel);
            const uint32_t      inputIdx = layout.pfnGetMicroVertexIndex(uv.u, uv.v, subdivLevel, layout.userData);
            umaj[umajIdx]                = uint16_t(input[inputIdx]);
        }
    }
    else
    {
        for(uint32_t umajIdx = 0; umajIdx < numValues; umajIdx++)
        {
            umaj[umajIdx] = uint16_t(input[umajIdx]);
        }
    }
}

template <class T>
void bitCopyUmaj16ToLayoutT(const uint16_t* umaj, T* output, const MicromapLayout& layout, const uint32_t subdivLevel)
{
    const uint32_t numValues = subdivLevelGetVertexCount(subdivLevel);
    if(layout.pfnGetMicroVertexIndex)
    {
        // NOTE: Is it faster if we iterate over (u, v)?
        for(uint32_t umajIdx = 0; umajIdx < numValues; umajIdx++)
        {
            const BaryUV_uint16 uv        = umajorLinearToUV(umajIdx, subdivLevel);
            const uint32_t      outputIdx = layout.pfnGetMicroVertexIndex(uv.u, uv.v, subdivLevel, layout.userData);
            output[outputIdx]             = T(umaj[umajIdx]);
        }
    }
    else
    {
        for(uint32_t umajIdx = 0; umajIdx < numValues; umajIdx++)
        {
            output[umajIdx] = T(umaj[umajIdx]);
        }
    }
}

MICROMESH_API Result MICROMESH_CALL micromeshCompressDisplacementBlock(const DisplacementBlock_settings* settings,
                                                                       uint64_t    scratchDataSize,
                                                                       void*       scratchData,
                                                                       const void* inputDecompressedValues,
                                                                       void*       outputCompressedValues,
                                                                       const MessageCallbackInfo* messageCallbackInfo)
{
    // Check API preconditions.
    {
        CHECK_NONNULLM(messageCallbackInfo, settings);
        CHECK_NONNULLM(messageCallbackInfo, scratchData);
        CHECK_NONNULLM(messageCallbackInfo, inputDecompressedValues);
        CHECK_NONNULLM(messageCallbackInfo, outputCompressedValues);
        const Result settingsValidation =
            validateDisplacementBlock(settings, inputDecompressedValues, outputCompressedValues, messageCallbackInfo);
        if(settingsValidation != Result::eSuccess)
        {
            return settingsValidation;
        }
        if(micromeshGetDisplacementBlockScratchSize(settings) != scratchDataSize)
        {
            return Result::eInvalidValue;
        }
    }

    const MicromapLayout& layout   = settings->decompressedLayout;
    const bool            isUMajor = isLayoutUMajor(layout);

    uint8_t* scratchBytes = reinterpret_cast<uint8_t*>(scratchData);
    uint32_t vertexCount  = subdivLevelGetVertexCount(settings->subdivLevel);

    dispenc::Intermediate intermediate{};
    intermediate.m_decoded     = reinterpret_cast<uint16_t*>(scratchBytes + sizeof(uint16_t) * vertexCount * 0);
    intermediate.m_corrections = reinterpret_cast<int16_t*>(scratchBytes + sizeof(uint16_t) * vertexCount * 1);

    const uint16_t* inputValues;

    // If we already have u16 values in u-major layout, we have a
    // fast path where we don't need to do any format or layout conversion.
    if(settings->decompressedFormat == Format::eR11_unorm_pack16 && isUMajor)
    {
        inputValues = static_cast<const uint16_t*>(inputDecompressedValues);
    }
    else
    {
        // Convert the input to eR11_unorm_pack16 umajor, then call encodeAndPack.
        uint16_t* scratchValues = reinterpret_cast<uint16_t*>(scratchBytes + sizeof(uint16_t) * vertexCount * 2);

        if(settings->decompressedFormat == Format::eR11_unorm_pack16)
        {
            bitCopyLayoutTToUmaj16(static_cast<const uint16_t*>(inputDecompressedValues), scratchValues, layout,
                                   settings->subdivLevel);
        }
        else if(settings->decompressedFormat == Format::eR32_uint)
        {
            bitCopyLayoutTToUmaj16(static_cast<const uint32_t*>(inputDecompressedValues), scratchValues, layout,
                                   settings->subdivLevel);
        }
        else
        {
            assert(!"Unhandled format; error in earlier validation!");
        }

        inputValues = scratchValues;
    }

    dispenc::blockEncode(settings->compressedBlockFormatDispC1, settings->subdivLevel, inputValues, intermediate);
    dispenc::blockPackData(settings->compressedBlockFormatDispC1, settings->subdivLevel, intermediate, outputCompressedValues);

    return Result::eSuccess;
}

MICROMESH_API Result MICROMESH_CALL micromeshDecompressDisplacementBlock(const DisplacementBlock_settings* settings,
                                                                         uint64_t    scratchDataSize,
                                                                         void*       scratchData,
                                                                         const void* inputCompressedValues,
                                                                         void*       outputDecompressedValues,
                                                                         const MessageCallbackInfo* messageCallbackInfo)
{
    // Check API preconditions.
    {
        CHECK_NONNULLM(messageCallbackInfo, settings);
        CHECK_NONNULLM(messageCallbackInfo, scratchData);
        CHECK_NONNULLM(messageCallbackInfo, inputCompressedValues);
        CHECK_NONNULLM(messageCallbackInfo, outputDecompressedValues);
        // Note (decompressed, compressed) order here
        const Result settingsValidation =
            validateDisplacementBlock(settings, outputDecompressedValues, inputCompressedValues, messageCallbackInfo);
        if(settingsValidation != Result::eSuccess)
        {
            return settingsValidation;
        }
        if(micromeshGetDisplacementBlockScratchSize(settings) > scratchDataSize)
        {
            return Result::eInvalidValue;
        }
    }

    const MicromapLayout& layout   = settings->decompressedLayout;
    const bool            isUMajor = isLayoutUMajor(layout);

    uint16_t* decodedValues;

    // If we already have u16 values in u-major layout, we have a
    // fast path where we don't need to do any format or layout conversion.
    if(settings->decompressedFormat == Format::eR11_unorm_pack16 && isUMajor)
    {
        decodedValues = static_cast<uint16_t*>(outputDecompressedValues);
    }
    else
    {
        decodedValues = reinterpret_cast<uint16_t*>(scratchData);
    }

    dispenc::blockDecode(settings->compressedFormat, settings->compressedBlockFormatDispC1, settings->subdivLevel,
                         inputCompressedValues, decodedValues);

    if(decodedValues != outputDecompressedValues)
    {
        if(settings->decompressedFormat == Format::eR11_unorm_pack16)
        {
            bitCopyUmaj16ToLayoutT(decodedValues, static_cast<uint16_t*>(outputDecompressedValues), layout, settings->subdivLevel);
        }
        else if(settings->decompressedFormat == Format::eR32_uint)
        {
            bitCopyUmaj16ToLayoutT(decodedValues, static_cast<uint32_t*>(outputDecompressedValues), layout, settings->subdivLevel);
        }
        else
        {
            assert(!"Unhandled format; error in earlier validation!");
        }
    }

    return Result::eSuccess;
}

}  // namespace micromesh
