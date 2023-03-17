//
// Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

// This file defines the API for converting displacements to and from a
// compressed representation. If you're using these functions for the first
// time, take a look at the README.md file for
// micromesh_displacement_compression; we include a few tutorials on how to
// compress meshes using this code, and describe the unique properties of these
// compressed formats. The tests include some additional examples as well.

#pragma once

#include <micromesh/micromesh_operations.h>

namespace micromesh
{
///////////////////////////////////////////////////////////////////////////////
// Compression and decompression for meshes. These compression functions can
// ensure that watertightness is maintained.

// This struct provides compression settings for the mesh compressor.
// One will often compress many meshes with the same settings.
struct OpCompressDisplacement_settings
{
    // Requires each compressed block to have a peak signal-to-noise ratio
    // = 10 * log_10(2048^2 / [mean square error]) of at least minimumPSNR.
    // Must not be infinity or NaN. Typical values are 30-50 (typically lower
    // when fitting is enabled for the same quality).
    float minimumPSNR = 50.0f;

    // Check that adjacent triangles have matching edge displacements before
    // encoding; returns Result::MISMATCHING_PRE_ENCODING_EDGES if there are
    // any discrepancies (i.e. if the decompressed mesh is not watertight).
    bool validateInputs = false;

    // Check that adjacent triangles have matching edge displacements after
    // encoding; returns Result::MISMATCHING_POST_ENCODING_EDGES if there are
    // any discrepancies (i.e. if the compressed mesh is not watertight).
    // Note that compression preserves watertightness: this should never occur
    // if the decompressed mesh is watertight.
    bool validateOutputs = false;

    // A bitmask of formats the compressor is allowed to use. For instance, to
    // enable `BlockFormatDispC1::eR11_unorm_lvl4_pack1024`, do
    // enabledBlockFormatBits |= (1u << (uint32_t)eR11_unorm_lvl4_pack1024)
    uint32_t enabledBlockFormatBits = ~0u;

    // If set to `true`, forces the compressor to use the lossless format for
    // triangles around non-manifold edges of a mesh (for instance, the outer
    // edges of a tessellated plane). This is useful when compressing meshes
    // that must fit together with other meshes without cracks.
    bool requireLosslessMeshEdges = true;

    // See OpCompressedDisplacement_output::mipData.
    uint16_t mipIgnoredSubdivLevel = 0xFFFF;
};

// This struct specifies the input to the compressor.
struct OpCompressDisplacement_input
{
    // The decompressed values.
    // `data->values.format` must be eR11UnormPack16.
    // `data->frequency` must be ePerMicroVertex.
    const Micromap* data = nullptr;

    // The topology of the mesh. The compressor uses this connectivity
    // information to ensure compressed edges are watertight when faces have
    // differing compression formats or subdivision levels differing by 1.
    const MeshTopology* topology = nullptr;

    // This must be either null or an array of 32-bit floats,
    // giving a weight for each vertex in the MeshTopology.
    // These weights are interpolated over each triangle and used when
    // computing loss/error. The idea is that errors in some areas affect the
    // quality more than in others, and the encoder can account for that.
    // A good choice is length(directions[i]) * displacementBounds[i].scale.
    // This array should be normalized so that its average value is 1.
    ArrayInfo_float perVertexImportance;

    // Optionally, specify a mapping from mesh triangles to
    // micromap triangles: mesh triangle i maps to micromap triangle
    // meshTriangleMappings[i]. If not specified, mesh triangle i maps
    // to micromap triangle i.
    // An interesting thing here is that this mapping can be many-to-one:
    // values can be compressed, ensuring watertightness of micromap triangles
    // on several mesh triangles simultaneously.
    // Similarly, it's possible that not all micromap triangles are values of
    // this mapping. The output's `triangleValueByteOffsets` and
    // `triangleBlockFormats` will have a .count for the entire micromap, but
    // `values` will only be allocated for referenced micromap triangles, and
    // the values of per-micromap triangle arrays for non-referenced triangles
    // are not defined.
    ArrayInfo_uint32 meshTriangleMappings;

    // Which family of compressed formats to generate. This must be
    // `eDispC1_r11_unorm_block`.
    Format compressedFormatFamily = Format::eDispC1_r11_unorm_block;
};

// This struct specifies the output of the compressor. In addition to
// MicromapCompressed, the compressor can output other useful information.
struct OpCompressDisplacement_output
{
    // The output compressed micromap. Required.
    MicromapCompressed* compressed = nullptr;

    // Optionally, the minimum and maximum of the compressed data of each
    // micromap triangle. Element 2*meshTriangleIndex + 0 stores the minimum,
    // and 2*meshTriangleIndex + 1 stores the maximum.
    // The format of this array must be eR11_unorm_packed16.
    ArrayInfo triangleMinMaxs;

    // Optional: For each micromap triangle, this will store the decompressed
    // values of the triangle (i.e. after compressing and then decompressing),
    // reduced to the level in `mipData->triangleSubdivisionLevels`, using
    // mipData's existing `triangleByteOffsets`.
    // Triangles can be excluded from having a mip computed for them by setting
    // their level in `mipData.triangleSubdivisionLevels` to
    // OpCompressDisplacement_input::mipIgnoredSubdivLevel.
    // If not ignored, each triangle's subdivision level in `mipData` must be
    // less than or equal to its subdivision level in the input.
    // Only its valueFloatExpansion field will be set by
    // micromeshOpCompressDisplacementBegin().
    // It must use a bird curve layout, and eR11_unorm_packed_align32.
    MicromapPacked* mipData = nullptr;
};

// Determine the space needed to store the result of compressing
// `inputDecompressed` with the given settings. This will set all members of
// `outputCompressed->compressed`, except for `data` members of untyped
// `ArrayInfo`s and `data`, `format`, and `byteStride` members of typed
// `ArrayInfo`s.
// If this succeeds, the application must then allocate the required
// corresponding memory and call `micromeshOpCompressDisplacementEnd()`, and
// pointers `settings` and `inputDecompressed` must remain valid until
// `micromeshOpCompressDisplacementEnd()` returns.
MICROMESH_API Result MICROMESH_CALL micromeshOpCompressDisplacementBegin(OpContext                              ctx,
                                                                         const OpCompressDisplacement_settings* settings,
                                                                         const OpCompressDisplacement_input* inputDecompressed,
                                                                         OpCompressDisplacement_output* outputCompressed);

// After calling `micromeshOpCompressDisplacementBegin` and allocating data,
// call this function to write the result of compression
// into `outputCompressed`. This may only be called once per call to
// `micromeshOpCompressDisplacementBegin()`.
MICROMESH_API Result MICROMESH_CALL micromeshOpCompressDisplacementEnd(OpContext ctx, OpCompressDisplacement_output* outputCompressed);

// Determine the space needed to store the decompressed form of
// `inputCompressed`. This will set all members of `outputDecompressed`,
// except for `data` and `byteStride` members of untyped `ArrayInfo`s and `data`, `format`, and
// `byteStride` members of typed `ArrayInfo`s.
//
// If this succeeds, the application must then allocate the corresponding
// memory, set all unset members, and call
// `micromeshOpDecompressDisplacementEnd()`, and the `inputCompressed` pointer
// must remain valid until `micromeshOpDecompressDisplacementEnd()` returns.
//
// Notes:
// * outputDecompressed->layout must be valid.
// * outputDecompressed->values.format will be set to eR11_unorm_pack16.
// * Since the output triangle subdiv levels are the same as the input
// triangle subdiv levels, it is valid to call this function with
// inputCompressed->triangleSubdivLevels.data == outputDecompressed->triangleSubdivLevels.data.
//
// Attempting to decompress a MicromapCompressed representing more than 2^32-1
// R11_unorm_pack16 values (just under 8 GB) will produce an error.
MICROMESH_API Result MICROMESH_CALL micromeshOpDecompressDisplacementBegin(OpContext                 ctx,
                                                                           const MicromapCompressed* inputCompressed,
                                                                           Micromap* outputDecompressed);

// After calling `micromeshOpDecompressDisplacementBegin()` and allocating
// data, call this to write the decompressed result into `outputDecompressed`.
// This may only be called once per call to
// `micromeshOpDecompressDisplacementBegin()`.
MICROMESH_API Result MICROMESH_CALL micromeshOpDecompressDisplacementEnd(OpContext ctx, Micromap* outputDecompressed);


///////////////////////////////////////////////////////////////////////////////
// Compression and decompression for invdividual blocks of values.

// We use the same structure for both compression and decompression.
struct DisplacementBlock_settings
{
    // The layout (e.g. u-major or bird curve) of the decompressed values.
    // If not set, defaults to u-major.
    MicromapLayout decompressedLayout;

    // The format of the decompressed values.
    // Must be eR11_unorm_pack16 or eR32_uint (we may relax this requirement later).
    Format decompressedFormat = Format::eR11_unorm_pack16;

    // The subdivision level of the input data. If using the
    // eR11_unorm_lvl3_pack512 DispC1 format, this must be 0, 1, 2, or 3.
    // Otherwise, this must be equal to the compressed block format's
    // subdivision level.
    uint32_t subdivLevel = 0;

    // `compressedFormat` and the following union (`compressedBlockFormatDispC1`,
    // `compressedBlockFormat`) specify the format of the compressed data.
    // This is designed for potential future expansion: `outputFormat`
    // specifies the family of the encoding, and must currently be
    // `eDispC1_r11_unorm_block`. `compressedFormatDispC1` is the
    // specific block displacement compression format of type 1.
    Format compressedFormat = Format::eDispC1_r11_unorm_block;
    union
    {
        BlockFormatDispC1 compressedBlockFormatDispC1;
        uint16_t          compressedBlockFormat;
    };
};

// computes `scratchDataSize` used in the below functions.
// Only reads `settings.compressedFormat`, `settings.decompressedFormat` and `settings.subdivLevel`
// The returned size is guaranteed to fit all permutations with equal or less `subdivLevel`
MICROMESH_API uint64_t MICROMESH_CALL micromeshGetDisplacementBlockScratchSize(const DisplacementBlock_settings* settings);

// Compresses values into a single packed block.
// This is a fast, thread-safe operation and does not require an OpContext.
//`outputCompressedValues` must have enough space for one block of the
// compressed block format. `inputDecompressedValues` and
// `outputCompressedValues` must not overlap.
MICROMESH_API Result MICROMESH_CALL micromeshCompressDisplacementBlock(const DisplacementBlock_settings* settings,
                                                                       uint64_t    scratchDataSize,
                                                                       void*       scratchData,
                                                                       const void* inputDecompressedValues,
                                                                       void*       outputCompressedValues,
                                                                       const MessageCallbackInfo* messageCallbackInfo);

// Decompresses values from a single packed block to the layout and format
// specified in the `DisplacementBlock_settings`.
// This is a fast, thread-safe operation and does not require an OpContext.
// `inputCompressedValues` must contain one block of the compressed block
// format. `inputCompressedValues` and `outputDecompressedValues` must
// not overlap.
MICROMESH_API Result MICROMESH_CALL micromeshDecompressDisplacementBlock(const DisplacementBlock_settings* settings,
                                                                         uint64_t    scratchDataSize,
                                                                         void*       scratchData,
                                                                         const void* inputCompressedValues,
                                                                         void*       outputDecompressedValues,
                                                                         const MessageCallbackInfo* messageCallbackInfo);

}  // namespace micromesh
