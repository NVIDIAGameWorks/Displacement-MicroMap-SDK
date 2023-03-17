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
#include <array>
#include <chrono>
#include <cstdlib>
#include <random>
#include <string.h>
#include <vector>

#include <micromesh/micromesh_operations.h>
#include <micromesh/micromesh_utils.h>
#include <micromesh/micromesh_format_types.h>
#include <micromesh/micromesh_displacement_compression.h>

#include "../src/displacement_block_codec.h"
#define DIAG_VERBOSE
#include "../src/displacement_diagnostic_utils.h"

// Ensure assertions are always checked even in release mode
#undef NDEBUG

using namespace micromesh;
std::default_random_engine gen(1);  // Ensure the RNG is seeded so tests are reproducible

static const std::array<BlockFormatDispC1, 3> allBlockFormats{BlockFormatDispC1::eR11_unorm_lvl3_pack512,
                                                              BlockFormatDispC1::eR11_unorm_lvl4_pack1024,
                                                              BlockFormatDispC1::eR11_unorm_lvl5_pack1024};

#define PRINT_AND_ASSERT_FALSE(...)                                                                                    \
    {                                                                                                                  \
        printf(__VA_ARGS__);                                                                                           \
        assert(false);                                                                                                 \
    }

#define TEST_TRUE(a)                                                                                                   \
    if(!(a))                                                                                                           \
    {                                                                                                                  \
        PRINT_AND_ASSERT_FALSE("Test failed: " #a "\n");                                                               \
        return false;                                                                                                  \
    }

#define TEST_SUCCESS(a)                                                                                                \
    if((a) != Result::eSuccess)                                                                                        \
    {                                                                                                                  \
        PRINT_AND_ASSERT_FALSE("Test did not return Result::eSuccess: " #a " \n");                                     \
        return false;                                                                                                  \
    }

void basicMessageCallback(MessageSeverity severity, const char* message, uint32_t threadIndex, const void* userData)
{
    if(severity == MessageSeverity::eInfo)
    {
        printf("INFO: %s\n", message);
    }
    else if(severity == MessageSeverity::eWarning)
    {
        printf("WARNING: %s\n", message);
    }
    else if(severity == MessageSeverity::eError)
    {
        PRINT_AND_ASSERT_FALSE("ERROR: %s\n", message);
    }
}

const MessageCallbackInfo messenger{basicMessageCallback, nullptr};

// Creates data members needed for encoding or decoding a single block of a given format and subdiv level.
// The goal of this is to make tests involving single-block encoding or decoding shorter.
template <class DecompressedT>
struct BlockEnvironment
{
    BlockFormatDispC1 fmt{};
    FormatInfo        fmtInfo{};
    Format            decompressedFormat{};
    uint32_t          subdivLevel{};
    // We use multiple allocations here instead of 1 so Address Sanitizer can
    // detect buffer overruns more easily.
    std::vector<DecompressedT> reference;
    std::vector<uint8_t>       packed;

    BlockEnvironment(BlockFormatDispC1 _fmt, uint32_t _subdivLevel = ~0u)
        : fmt(_fmt)
    {
        if(micromeshBlockFormatDispC1GetInfo(fmt, &fmtInfo) != Result::eSuccess)
        {
            PRINT_AND_ASSERT_FALSE("Failed to get format info!\n");
        }

        if(_subdivLevel == ~0u)
        {
            // NOTE: Maybe we should have a helper function for "what's the
            // maximum subdiv level for this block format"?
            switch(fmt)
            {
            case BlockFormatDispC1::eR11_unorm_lvl3_pack512:
                subdivLevel = 3;
                break;
            case BlockFormatDispC1::eR11_unorm_lvl4_pack1024:
                subdivLevel = 4;
                break;
            case BlockFormatDispC1::eR11_unorm_lvl5_pack1024:
                subdivLevel = 5;
                break;
            default:
                PRINT_AND_ASSERT_FALSE("Unknown format!\n");
            }
        }
        else
        {
            subdivLevel = _subdivLevel;
        }

        if(std::is_same_v<DecompressedT, uint16_t>)
        {
            decompressedFormat = Format::eR11_unorm_pack16;
        }
        else if(std::is_same_v<DecompressedT, uint32_t>)
        {
            decompressedFormat = Format::eR32_uint;
        }
        else
        {
            PRINT_AND_ASSERT_FALSE("Unknown DecompressedT!\n");
        }

        reference.resize(fmtInfo.valueCount);
        packed.resize(fmtInfo.byteSize);
    }

    DisplacementBlock_settings generateSettings()
    {
        DisplacementBlock_settings result;
        micromeshLayoutInitStandard(&result.decompressedLayout, StandardLayoutType::eUmajor);
        result.decompressedFormat          = decompressedFormat;
        result.subdivLevel                 = subdivLevel;
        result.compressedFormat            = Format::eDispC1_r11_unorm_block;
        result.compressedBlockFormatDispC1 = fmt;
        return result;
    }
};

// Simplified C++ class for allocating and building a MeshTopology from an
// index buffer.
struct MeshTopologyBuilder
{
    MeshTopology topology;
    // NOTE: We could combine this all into one allocation.
    std::vector<micromesh::Vector_uint32_3> triangleVertices;
    std::vector<micromesh::Vector_uint32_3> triangleEdges;
    std::vector<micromesh::Range32>         vertexEdgeRanges;
    std::vector<micromesh::Range32>         vertexTriangleRanges;
    std::vector<uint32_t>                   vertexTriangleConnections;
    std::vector<uint32_t>                   vertexEdgeConnections;
    std::vector<uint32_t>                   edgeVertices;
    std::vector<micromesh::Range32>         edgeTriangleRanges;
    std::vector<uint32_t>                   edgeTriangleConnections;

    MeshTopologyBuilder(){};
    Result build(OpContext ctx, const std::vector<uint32_t>& indices, size_t numVertices)
    {
        const size_t numTriangles = indices.size() / 3;

        // Allocate space for triangleEdges, vertexEdgeRanges, and vertexTriangleRanges
        topology = MeshTopology{};

        // Make a copy of the index buffer - otherwise we run into lifetime issues
        triangleVertices.resize(numTriangles);
        memcpy(triangleVertices.data(), indices.data(), triangleVertices.size() * sizeof(Vector_uint32_3));
        arraySetDataVec(topology.triangleVertices, triangleVertices);

        triangleEdges.resize(numTriangles);
        arraySetDataVec(topology.triangleEdges, triangleEdges);

        topology.vertexEdgeRanges.count = numVertices;
        vertexEdgeRanges.resize(numVertices);
        topology.vertexEdgeRanges.data = vertexEdgeRanges.data();

        topology.vertexTriangleRanges.count = numVertices;
        vertexTriangleRanges.resize(numVertices);
        topology.vertexTriangleRanges.data = vertexTriangleRanges.data();

        // Fill those 3 arrays and get sizes for remaining MeshTopology arrays
        Result result = micromeshOpBuildMeshTopologyBegin(ctx, &topology);
        if(result != Result::eSuccess)
            return result;

        // Allocate remaining output.
        vertexTriangleConnections.resize(topology.vertexTriangleConnections.count);
        topology.vertexTriangleConnections.data = vertexTriangleConnections.data();

        vertexEdgeConnections.resize(topology.vertexEdgeConnections.count);
        topology.vertexEdgeConnections.data = vertexEdgeConnections.data();

        edgeVertices.resize(topology.edgeVertices.count * 2);
        topology.edgeVertices.data = edgeVertices.data();

        edgeTriangleRanges.resize(topology.edgeTriangleRanges.count);
        topology.edgeTriangleRanges.data = edgeTriangleRanges.data();

        edgeTriangleConnections.resize(topology.edgeTriangleConnections.count);
        topology.edgeTriangleConnections.data = edgeTriangleConnections.data();

        // Okay, now build the topology!
        return micromeshOpBuildMeshTopologyEnd(ctx, &topology);
    }
    uint64_t vertCount() const { return topology.vertexEdgeRanges.count; }
};

// Tests that correct() always returns the best possible correction.
bool blockCorrectMath()
{
    // For every numCorrectionBits in [1,11], shift in [0, 10], prediction in
    // [0,2047] and reference in [0,2047],
    // correct(prediction, reference, shift, numCorrectionBits) should return
    // the integer between -2^(numCorrectionBits-1) and 2^(numCorrectionBits)-1
    // that makes (2047 & (prediction + (v << shift))) as close to
    // `reference` as possible.
    std::uniform_int_distribution<uint16_t> distCorrectionBits(1, 11);
    std::uniform_int_distribution<uint16_t> distShift(0, 10);
    std::uniform_int_distribution<uint16_t> distValue(0, 2047);
    for(int test = 0; test < 100; test++)
    {
        const uint16_t prediction = distValue(gen), reference = distValue(gen);
        const uint16_t shift          = distShift(gen);
        const uint16_t correctionBits = distCorrectionBits(gen);

        // What does BlockEncoder::correct return?
        const int16_t reported = dispenc::correct(prediction, reference, shift, correctionBits);

        const uint16_t distance = std::abs(reference - (2047 & (prediction + (reported << shift))));
        // Is there a value that gives a better distance?
        const int16_t minV = -(1 << (correctionBits - 1));
        const int16_t maxV = (1 << (correctionBits - 1)) - 1;
        // Make sure `reported` is within the valid range
        TEST_TRUE((reported >= minV) && (reported <= maxV));
        for(int16_t v = minV; v <= maxV; v++)
        {
            uint16_t myDistance = std::abs(reference - (2047 & (prediction + (v << shift))));
            if(myDistance < distance)
            {
                printf("BlockEncoder::correct(%i, %i, %u, %u) returned %i, but %i was closer!", prediction, reference,
                       shift, correctionBits, reported, v);
                dispenc::correct(prediction, reference, shift, correctionBits);
                return false;
            }
        }
    }
    return true;
}

// This is the BlockEncoder example from README.md.
bool docExampleBlockEncoder()
{
    // Specify the values for the block.
    // These are stored in an u16 array in u-major order: a triangular array,
    // u vertically, v horizontally.
    // In general code, this should have subdivLevelGetVertexCount(subdivLevel)
    // elements and be filled using the functions from a MicromeshLayout,
    // instead of hardcoding this information.
    const uint32_t                  subdivLevel = 4;
    const std::array<uint16_t, 153> referenceValues{
        1319, 1339, 1321, 1423, 1375, 1241, 1309, 1243, 1087, 1045, 593, 273, 435, 615, 627, 807, 947,
        1435, 1343, 1369, 1431, 1349, 1261, 1297, 1119, 981,  941,  467, 311, 467, 619, 647, 849,  //
        1465, 1359, 1377, 1427, 1323, 1281, 1237, 1077, 1067, 767,  445, 259, 421, 661, 689,       //
        1451, 1433, 1363, 1405, 1355, 1257, 1119, 1067, 913,  635,  473, 163, 423, 709,            //
        1385, 1385, 1351, 1443, 1329, 1155, 967,  1043, 825,  649,  249, 107, 543,                 //
        1395, 1347, 1351, 1419, 1265, 991,  1039, 867,  691,  595,  410, 225,                      //
        1421, 1339, 1333, 1373, 1135, 953,  969,  739,  713,  383,  161,                           //
        1329, 1275, 1307, 1313, 875,  997,  815,  759,  611,  183,                                 //
        1235, 1205, 1315, 1167, 963,  815,  757,  815,  551,                                       //
        1165, 1189, 1235, 1021, 977,  795,  879,  803,                                             //
        1069, 1083, 1257, 965,  785,  897,  941,                                                   //
        1047, 1123, 1097, 907,  893,  1037,                                                        //
        1009, 1099, 903,  899,  1065,                                                              //
        989,  887,  1021, 1095,                                                                    //
        839,  1021, 1137,                                                                          //
        945,  1201,                                                                                //
        1185};

    // Define the compression settings we'll use.
    DisplacementBlock_settings settings;
    micromeshLayoutInitStandard(&settings.decompressedLayout, StandardLayoutType::eUmajor);  // Use a U-major layout.
    settings.decompressedFormat = Format::eR11_unorm_pack16;  // Our input is 11-bit values in a u16 array
    settings.subdivLevel        = 4;                          // at subdivision level 4
    settings.compressedFormat = Format::eDispC1_r11_unorm_block;  // Compress to a format in the eDispC1_r11_unorm_block family
    settings.compressedBlockFormatDispC1 = BlockFormatDispC1::eR11_unorm_lvl4_pack1024;  // The level-4, 1024-byte format

    // Allocate space for the compressed block.
    // Here we'll show general usage of micromeshBlockFormatDispC1GetInfo():
    FormatInfo blockFmtInfo{};
    if(micromeshBlockFormatDispC1GetInfo(settings.compressedBlockFormatDispC1, &blockFmtInfo) != Result::eSuccess)
    {
        // This should never happen, but often good to check:
        PRINT_AND_ASSERT_FALSE("Failed to get format info!\n");
        return false;
    }
    std::vector<uint8_t> packedBlock(blockFmtInfo.byteSize);

    uint64_t             scratchSize = micromeshGetDisplacementBlockScratchSize(&settings);
    std::vector<uint8_t> scratchData(scratchSize);

    // Encode!
    const Result encodeResult = micromeshCompressDisplacementBlock(&settings, scratchSize, scratchData.data(),
                                                                   referenceValues.data(), packedBlock.data(), &messenger);
    // Also good to check for errors here; usually failures will be due to
    // incorrect settings, so we shouldn't see errors here.
    TEST_SUCCESS(encodeResult);

    printBitsIfVerbose(packedBlock);

    // Make sure this matches the printout in the documentation.
    // Note that this may break if we improve the block encoder! That's OK if
    // so - just double-check the result.
    {
        // Note that the words here are technically in mixed-endian. Reversing
        // the vector gives a little-endian block as intended.
        std::vector<uint32_t> docPackedBlock{0b00000010010001110100000000010000u,  //
                                             0b01111100011000011000010001100101u,  //
                                             0b11110111001000111101000110110111u,  //
                                             0b11100000000000010011011110110011u,  //
                                             0b01100010111111100010000011111111u,  //
                                             0b01011111000001001000010100011000u,  //
                                             0b01111111111100010000010000000000u,  //
                                             0b11101110000001000001010000000000u,  //
                                             0b10000001000101111011110111010000u,  //
                                             0b10000100000111010001000000111101u,  //
                                             0b11110001100100111100000111101000u,  //
                                             0b10111110010111011000011110100001u,  //
                                             0b11110000000000100010111100000111u,  //
                                             0b11100011000001111011010000011111u,  //
                                             0b10000000000000110001100000000001u,  //
                                             0b11111111111111000010000000010000u,  //
                                             0b10000010001000101000110001011110u,  //
                                             0b00101111100000111101111001111001u,  //
                                             0b00011110000000001101111110100011u,  //
                                             0b11101011000000001100110011111101u,  //
                                             0b00100110111000110110010000101000u,  //
                                             0b00000001011111110000000110011011u,  //
                                             0b11010111100101101000011110011011u,  //
                                             0b00011111101010001001011111001001u,  //
                                             0b01111100011000001110100010011100u,  //
                                             0b00110111100001100111111111010000u,  //
                                             0b11100011111111101111110011000000u,  //
                                             0b00011011100110010101101110101110u,  //
                                             0b01101110001100010100001000110000u,  //
                                             0b01101100000101010000001010110011u,  //
                                             0b11110111110111111101111110100100u,  //
                                             0b11101100111001010000110100100111u};
        std::reverse(docPackedBlock.begin(), docPackedBlock.end());
        TEST_TRUE(memcmp(packedBlock.data(), docPackedBlock.data(), packedBlock.size()) == 0);
    }

    // Now decode it!
    std::vector<uint16_t> decodedValues(subdivLevelGetVertexCount(subdivLevel));
    // We can use the same settings to reverse the process!
    const DisplacementBlock_settings decompressionSettings = settings;
    TEST_SUCCESS(micromeshDecompressDisplacementBlock(&decompressionSettings, scratchSize, scratchData.data(),
                                                      packedBlock.data(), decodedValues.data(), &messenger));

    printBlockArrayIfVerbose(decodedValues, subdivLevel);

    return true;
}


// Verifies that encoding random blocks doesn't crash and that the two MSBs of
// each format are zero (since they're reserved for future use).
bool encodeRandomBlocks()
{
    for(BlockFormatDispC1 fmt : allBlockFormats)
    {
        for(int iter = 0; iter < 100; iter++)
        {
            using DecompressedT = uint16_t;
            BlockEnvironment<DecompressedT> b(fmt);

            // Generate some random data
            std::uniform_int_distribution<DecompressedT> dist(0u, (1u << 11) - 1u);
            for(DecompressedT& v : b.reference)
            {
                v = dist(gen);
            }

            DisplacementBlock_settings settings    = b.generateSettings();
            uint64_t                   scratchSize = micromeshGetDisplacementBlockScratchSize(&settings);
            std::vector<uint8_t>       scratchData(scratchSize);

            // Choose a random layout
            {
                std::uniform_int_distribution<int> randomLayoutDist(0u, 2u);
                switch(randomLayoutDist(gen))
                {
                case 0:
                    // Test default behavior
                    settings.decompressedLayout.pfnGetMicroTriangleIndex = nullptr;
                    settings.decompressedLayout.pfnGetMicroVertexIndex   = nullptr;
                    break;
                case 1:
                    // U-major
                    micromeshLayoutInitStandard(&settings.decompressedLayout, StandardLayoutType::eUmajor);
                    break;
                case 2:
                    // Bird curve
                    micromeshLayoutInitStandard(&settings.decompressedLayout, StandardLayoutType::eBirdCurve);
                    break;
                }
            }

            // Compress it
            TEST_SUCCESS(micromeshCompressDisplacementBlock(&settings, scratchSize, scratchData.data(),
                                                            b.reference.data(), b.packed.data(), &messenger));

            // Verify that the 2 MSBs are zero
            TEST_TRUE((b.packed.back() & 0b11000000u) == 0);

            // Try to decompress it
            TEST_SUCCESS(micromeshDecompressDisplacementBlock(&settings, scratchSize, scratchData.data(),
                                                              b.packed.data(), b.reference.data(), &messenger));
        }
    }

    return true;
}

// Tests perfect re-encoding, and that decode(pack(encode(decode(x)))) == decode(x).
// (i.e. that decode(pack(encode(y))) == y for all y in the space of compressed displacements)
bool perfectReencoding()
{
    std::uniform_int_distribution<int16_t> dist(0, 255);

    const size_t CASES_PER_TEST = 1000;

    for(BlockFormatDispC1 fmt : allBlockFormats)
    {
        using DecompressedT = uint16_t;
        BlockEnvironment<DecompressedT> b(fmt);

        for(size_t iter = 0; iter < CASES_PER_TEST; iter++)
        {
            // Generate random *encoded* data
            std::vector<uint8_t> originalPacked(b.packed.size());
            for(uint8_t& v : originalPacked)
            {
                v = uint8_t(dist(gen));
            }
            // Zero out the two most significant bits so that it's valid
            originalPacked.back() &= 0b00111111u;

            // reference = decode(x)
            DisplacementBlock_settings settings    = b.generateSettings();
            uint64_t                   scratchSize = micromeshGetDisplacementBlockScratchSize(&settings);
            std::vector<uint8_t>       scratchData(scratchSize);

            TEST_SUCCESS(micromeshDecompressDisplacementBlock(&settings, scratchSize, scratchData.data(),
                                                              originalPacked.data(), b.reference.data(), &messenger));

            // packed = pack(encoded)
            TEST_SUCCESS(micromeshCompressDisplacementBlock(&settings, scratchSize, scratchData.data(),
                                                            b.reference.data(), b.packed.data(), &messenger));

            // final_decoded = decode(packed)
            std::vector<uint16_t> unpackedAndDecoded(b.reference.size());
            TEST_SUCCESS(micromeshDecompressDisplacementBlock(&settings, scratchSize, scratchData.data(),
                                                              b.packed.data(), unpackedAndDecoded.data(), &messenger));

            const uint32_t numSegments = subdivLevelGetSegmentCount(b.subdivLevel);
            for(uint32_t v = 0; v <= numSegments; v++)
            {
                for(uint32_t u = 0; u <= numSegments; u++)
                {
                    if(u + v <= numSegments)
                    {
                        const uint32_t adr = umajorUVtoLinear(u, v, b.subdivLevel);
                        if(unpackedAndDecoded[adr] != b.reference[adr])
                        {
#ifdef DIAG_VERBOSE
                            printf("Mismatch in decoded == reference!\n");
                            printf("Reference:\n");
                            printBlockArrayIfVerbose(b.reference, b.subdivLevel);
                            printf("Decoded:\n");
                            printBlockArrayIfVerbose(unpackedAndDecoded, b.subdivLevel);
                            std::vector<uint16_t> diff(unpackedAndDecoded.size());
                            for(size_t i = 0; i < unpackedAndDecoded.size(); i++)
                            {
                                diff[i] = b.reference[i] - unpackedAndDecoded[i];
                            }
                            printf("Reference - decoded:\n");
                            printBlockArrayIfVerbose(diff, b.subdivLevel);
#endif
                            TEST_TRUE(false);
                        }
                    }
                }
            }
        }
    }

    return true;
}


// Micromesh SDK version of the docExampleEncodeTwoTriangles() test from the
// displacement_encoder.
bool docExampleEncodeTwoTriangles(OpContext context)
{
    // Build the mesh topology for the following mesh:
    // 0-1
    //  2-3
    constexpr size_t            NUM_VERTICES  = 4;
    constexpr size_t            NUM_TRIANGLES = 2;
    const std::vector<uint32_t> indices{0, 1, 2, 1, 3, 2};

    MeshTopologyBuilder topoBuilder;
    TEST_SUCCESS(topoBuilder.build(context, indices, NUM_VERTICES));

    // Now, generate the uncompressed data.
    // We'll have two triangles with subdiv levels 4 and 5, and the values
    // will be random 11-bit UNORM values - including mismatches at edges!
    Micromap uncompressed{};
    micromeshLayoutInitStandard(&uncompressed.layout, StandardLayoutType::eUmajor);
    uncompressed.frequency = Frequency::ePerMicroVertex;
    for(int i = 0; i < 4; i++)
    {
        uncompressed.valueFloatExpansion.scale[i] = 1.0f;
    }
    uncompressed.minSubdivLevel = 4;
    uncompressed.maxSubdivLevel = 5;

    std::array<uint16_t, NUM_TRIANGLES> subdivLevels{4, 5};
    uncompressed.triangleSubdivLevels.count = subdivLevels.size();
    uncompressed.triangleSubdivLevels.data  = subdivLevels.data();

    std::array<uint32_t, NUM_TRIANGLES> uncompressedIndexOffsets{0, subdivLevelGetVertexCount(subdivLevels[0])};
    uncompressed.triangleValueIndexOffsets.count = uncompressedIndexOffsets.size();
    uncompressed.triangleValueIndexOffsets.data  = uncompressedIndexOffsets.data();

    std::vector<uint16_t>                   values(subdivLevelGetVertexCount(4) + subdivLevelGetVertexCount(5));
    std::default_random_engine              gen;
    std::uniform_int_distribution<uint16_t> dist(0, (1 << 11) - 1);
    for(uint16_t& v : values)
    {
        v = dist(gen);
    }
    uncompressed.values.count      = values.size();
    uncompressed.values.data       = values.data();
    uncompressed.values.byteStride = sizeof(uint16_t);
    uncompressed.values.format     = Format::eR11_unorm_pack16;

    // We now have the uncompressed data. Encode it!
    OpCompressDisplacement_input input{};
    input.data                   = &uncompressed;
    input.topology               = &topoBuilder.topology;
    input.compressedFormatFamily = Format::eDispC1_r11_unorm_block;

    // Rely on the encoder fixing cracks
    OpCompressDisplacement_settings settings{};
    settings.validateInputs           = false;
    settings.validateOutputs          = true;
    settings.requireLosslessMeshEdges = false;

    MicromapCompressed            compressed{};
    OpCompressDisplacement_output output{};
    output.compressed = &compressed;

    TEST_SUCCESS(micromeshOpCompressDisplacementBegin(context, &settings, &input, &output));

    // Allocate the output. values.{count, byteStride, format} are already set by Begin.
    std::vector<uint8_t> compressedData(compressed.values.count * compressed.values.byteStride);
    compressed.values.data = compressedData.data();

    // The different .count fields are set by Begin:
    compressed.triangleSubdivLevels.data = subdivLevels.data();

    std::vector<uint32_t> compressedValueByteOffsets(compressed.triangleValueByteOffsets.count);
    compressed.triangleValueByteOffsets.data = compressedValueByteOffsets.data();

    std::vector<uint16_t> compressedBlockFormats(compressed.triangleBlockFormats.count);
    compressed.triangleBlockFormats.data = compressedBlockFormats.data();

    // Finish compression.
    TEST_SUCCESS(micromeshOpCompressDisplacementEnd(context, &output));

    return true;
}

// Tests that decoding an encoding mesh works properly: i.e.
// encode(decode(encode(M, S)), S) == encode(M, S) for any micromap M and
// settings (including mesh topology) S.
bool perfectMeshReencoding(OpContext ctx)
{
    // We could procedurally generate a random mesh topology -- but here I'll
    // just use a constant graph, a triangular dipyramid minus one face
    // (so we have some non-manifold edges).
    //   0---.
    //  /|\  |
    // 1-2-3 |
    // |\|/| |
    // ( 4 ) |
    //  \|/  |
    //   5---'
    constexpr size_t            NUM_VERTICES  = 6;
    constexpr size_t            NUM_TRIANGLES = 7;
    const std::vector<uint32_t> indices{0, 2, 1,  //
                                        0, 3, 2,  //
                                        1, 2, 4,  //
                                        2, 3, 4,  //
                                        1, 4, 5,  //
                                        4, 3, 5,  //
                                        0, 5, 3};
    MeshTopologyBuilder         topoBuilder;
    TEST_SUCCESS(topoBuilder.build(ctx, indices, NUM_VERTICES));

    // Distributions.
    std::uniform_int_distribution<uint32_t> distBool(0, 1);
    const uint16_t                          MAX_SUBDIV = 7;
    std::poisson_distribution<uint16_t>     distSubdivUnclamped(4);
    std::uniform_int_distribution<uint16_t> distValues(0, (1 << 11) - 1);
    std::uniform_real_distribution<float>   distPSNR(0.0f, 50.0f);

    // Collect some information about subdivision levels and formats used.
    std::vector<size_t> subdivLevelsUsed(MAX_SUBDIV + 1);
    std::vector<size_t> blockFmtsUsed(size_t(BlockFormatDispC1::eR11_unorm_lvl5_pack1024) + 1);

    // Do some number of random tests.
    for(uint32_t test = 0; test < 100; test++)
    {
        // Now, generate the uncompressed data M_0.
        // This will be somewhat ill-formed, since I want to really see how this
        // can break.
        Micromap micromap0;
        micromap0.frequency           = Frequency::ePerMicroVertex;
        micromap0.valueFloatExpansion = MicromapValueFloatExpansion{};
        // Random layout!
        micromeshLayoutInitStandard(&micromap0.layout,
                                    (distBool(gen) == 0 ? StandardLayoutType::eUmajor : StandardLayoutType::eBirdCurve));
        // We'll give each triangle a random subdivision level from 0 to MAX_SUBDIV.
        micromap0.minSubdivLevel = ~0;
        micromap0.maxSubdivLevel = 0;
        std::vector<uint16_t> m0TriangleSubdivLevels(NUM_TRIANGLES);
        for(uint16_t& subdivLevel : m0TriangleSubdivLevels)
        {
            subdivLevel              = std::min(distSubdivUnclamped(gen), MAX_SUBDIV);
            micromap0.minSubdivLevel = std::min(micromap0.minSubdivLevel, uint32_t(subdivLevel));
            micromap0.maxSubdivLevel = std::max(micromap0.maxSubdivLevel, uint32_t(subdivLevel));
        }
        // Sanitize the subdivision levels along edges (otherwise we'll get
        // an assertion error inside the compressor)
        OpSanitizeSubdivLevels_input sanitizeSubdivInput{};
        sanitizeSubdivInput.maxSubdivLevel = micromap0.maxSubdivLevel;
        sanitizeSubdivInput.meshTopo       = &topoBuilder.topology;
        arraySetDataVec(sanitizeSubdivInput.meshTriangleSubdivLevels, m0TriangleSubdivLevels);
        OpSanitizeSubdivLevels_output sanitizeSubdivOutput{};
        arraySetDataVec(sanitizeSubdivOutput.meshTriangleSubdivLevels, m0TriangleSubdivLevels);
        TEST_SUCCESS(micromeshOpSanitizeSubdivLevels(ctx, &sanitizeSubdivInput, &sanitizeSubdivOutput));

        arraySetDataVec(micromap0.triangleSubdivLevels, m0TriangleSubdivLevels);
        // Generate random values. Note that not only will the triangles likely
        // mismatch along edges - they also likely will have overlapping value
        // ranges (e.g. triangles will start in the middle of other triangles'
        // data)! This is likely invalid to write to a .bary file, but let's
        // see if we can handle it.
        const size_t valuesForLargestTriangle = subdivLevelGetVertexCount(micromap0.maxSubdivLevel);
        std::uniform_int_distribution<size_t> distValueCount(valuesForLargestTriangle, 2 * valuesForLargestTriangle);
        std::vector<uint16_t>                 m0Values(distValueCount(gen));
        for(uint16_t& v : m0Values)
        {
            v = distValues(gen);
        }
        micromap0.values.byteStride = 2;
        micromap0.values.count      = m0Values.size();
        micromap0.values.data       = m0Values.data();
        micromap0.values.format     = micromesh::Format::eR11_unorm_pack16;
        std::vector<uint32_t> m0ValueOffsets(NUM_TRIANGLES);
        for(size_t tri = 0; tri < NUM_TRIANGLES; tri++)
        {
            const uint32_t numValuesRequired = subdivLevelGetVertexCount(m0TriangleSubdivLevels[tri]);
            std::uniform_int_distribution<uint32_t> distValueOffset(0, uint32_t(m0Values.size() - numValuesRequired));
            m0ValueOffsets[tri] = distValueOffset(gen);
        }
        arraySetDataVec(micromap0.triangleValueIndexOffsets, m0ValueOffsets);

        // Create random settings.
        OpCompressDisplacement_settings settings{};
        settings.minimumPSNR              = distPSNR(gen);
        settings.validateInputs           = false;
        settings.validateOutputs          = false;
        settings.requireLosslessMeshEdges = false;


        // Encode it to get E_0 := encode(M_0, S).
        MicromapCompressed            e0{};
        OpCompressDisplacement_input  e0Input{};
        OpCompressDisplacement_output e0Output{};
        e0Input.compressedFormatFamily = micromesh::Format::eDispC1_r11_unorm_block;
        e0Input.data                   = &micromap0;
        e0Input.topology               = &topoBuilder.topology;
        e0Output.compressed            = &e0;
        TEST_SUCCESS(micromeshOpCompressDisplacementBegin(ctx, &settings, &e0Input, &e0Output));
        std::vector<uint8_t> e0Values(e0.values.count);
        e0.values.data = e0Values.data();
        std::vector<uint16_t> e0TriangleSubdivLevels(e0.triangleSubdivLevels.count);
        e0.triangleSubdivLevels.data = e0TriangleSubdivLevels.data();
        std::vector<uint16_t> e0TriangleBlockFormats(e0.triangleBlockFormats.count);
        e0.triangleBlockFormats.data = e0TriangleBlockFormats.data();
        std::vector<uint32_t> e0TriangleValueByteOffsets(e0.triangleValueByteOffsets.count);
        e0.triangleValueByteOffsets.data = e0TriangleValueByteOffsets.data();
        TEST_SUCCESS(micromeshOpCompressDisplacementEnd(ctx, &e0Output));


        // Decode it to get M_1 := decode(E_0)
        Micromap m1{};
        m1.layout = micromap0.layout;
        TEST_SUCCESS(micromeshOpDecompressDisplacementBegin(ctx, &e0, &m1));
        m1.valueFloatExpansion = micromap0.valueFloatExpansion;
        std::vector<uint16_t> m1Values(m1.values.count);
        m1.values.data       = m1Values.data();
        m1.values.byteStride = 2;
        std::vector<uint16_t> m1TriangleSubdivLevels(m1.triangleSubdivLevels.count);
        m1.triangleSubdivLevels.data = m1TriangleSubdivLevels.data();
        std::vector<uint32_t> m1TriangleValueIndexOffsets(m1.triangleValueIndexOffsets.count);
        m1.triangleValueIndexOffsets.data = m1TriangleValueIndexOffsets.data();
        TEST_SUCCESS(micromeshOpDecompressDisplacementEnd(ctx, &m1));


        // Re-encode it to get E_1 := encode(M_1, S').
        // We require perfect re-encoding here (so that we hopefully choose the
        // same compression formats), so set the minimum PSNR to a really large
        // number (TODO: Test if +infinity works)
        settings.minimumPSNR = 1000.0f;
        MicromapCompressed            e1{};
        OpCompressDisplacement_input  e1Input{};
        OpCompressDisplacement_output e1Output{};
        e1Input.compressedFormatFamily = micromesh::Format::eDispC1_r11_unorm_block;
        e1Input.data                   = &m1;
        e1Input.topology               = &topoBuilder.topology;
        e1Output.compressed            = &e1;
        TEST_SUCCESS(micromeshOpCompressDisplacementBegin(ctx, &settings, &e1Input, &e1Output));
        std::vector<uint8_t> e1Values(e1.values.count);
        e1.values.data = e1Values.data();
        std::vector<uint16_t> e1TriangleSubdivLevels(e1.triangleSubdivLevels.count);
        e1.triangleSubdivLevels.data = e1TriangleSubdivLevels.data();
        std::vector<uint16_t> e1TriangleBlockFormats(e1.triangleBlockFormats.count);
        e1.triangleBlockFormats.data = e1TriangleBlockFormats.data();
        std::vector<uint32_t> e1TriangleValueByteOffsets(e1.triangleValueByteOffsets.count);
        e1.triangleValueByteOffsets.data = e1TriangleValueByteOffsets.data();
        TEST_SUCCESS(micromeshOpCompressDisplacementEnd(ctx, &e1Output));


        // And now, check that E_0 == E_1
        for(uint64_t i = 0; i < e1.triangleSubdivLevels.count; i++)
        {
            TEST_TRUE(e0TriangleSubdivLevels[i] == e1TriangleSubdivLevels[i]);
            subdivLevelsUsed[e1TriangleSubdivLevels[i]]++;
        }
        for(uint64_t i = 0; i < e1.triangleBlockFormats.count; i++)
        {
            TEST_TRUE(e0TriangleBlockFormats[i] == e1TriangleBlockFormats[i]);
            blockFmtsUsed[size_t(e1TriangleBlockFormats[i])]++;
        }
        for(uint64_t i = 0; i < e1.triangleValueByteOffsets.count; i++)
        {
            TEST_TRUE(e0TriangleValueByteOffsets[i] == e1TriangleValueByteOffsets[i]);
        }
        // Because our data has so many discontinuities, we will occasionally
        // get value mismatches. These appear to be okay, and I haven't worked
        // out exactly where they're coming from - my guess is the edge
        // propagation step modifies triangle data, then unsuccessfully undoes
        // it. If there was a major error in CompressDisplacement, this would
        // be large.
        uint64_t mismatchCount = 0;
        for(uint64_t i = 0; i < e1.values.count; i++)
        {
            if(e0Values[i] != e1Values[i])
            {
                mismatchCount++;
            }
        }
        TEST_TRUE(mismatchCount <= 64);
    }

#ifdef DIAG_VERBOSE
    printf("Histogram of subdivision levels encountered: ");
    for(const size_t& count : subdivLevelsUsed)
    {
        printf("%zu ", count);
    }
    printf("\nHistogram of block formats used: ");
    for(const size_t& count : blockFmtsUsed)
    {
        printf("%zu ", count);
    }
    printf("\n");
#endif

    return true;
}

// Utility: This function prevents a compiler from optimizing out an operation,
// in case you're compiling in release mode with aggressive optimization.
template <class T>
void doNotOptimizeOut(T const& t)
{
    volatile T sinkhole = t;
}

template <class LambdaFunction>
void benchmark(const char* benchmarkName, LambdaFunction func)
{
    // Time the function by running it 10 times, then keep running it until
    // the sample standard deviation is less than accuracyFrac * the mean, or
    // we complete too many iterations (to avoid an infinite loop).
    using Clock = std::chrono::steady_clock;
    // Note that this isn't guaranteed to match the tick resolution of the
    // clock - hopefully it's a good enough proxy, though.
    static_assert(std::ratio_less_equal_v<Clock::period, std::micro>,
                  "The monotonic clock's period was larger than a microsecond - possibly too large to benchmark "
                  "accurately.");
    const double accuracyFrac = 0.05;
    // We use the running sample variance algorithm from https://www.johndcook.com/blog/standard_deviation/.
    size_t runCount      = 0;
    double runningMean   = 0;
    double runningS      = 0;
    double runningStddev = 0;
    while(true)
    {
        const Clock::time_point start = Clock::now();
        doNotOptimizeOut(func());
        const Clock::time_point end = Clock::now();
        runCount++;
        const double thisTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

        // Update running sample variance
        if(runCount == 1)
        {
            runningMean = thisTime;
            runningS    = 0;
        }
        else
        {
            const double oldMean = runningMean;
            const double k       = double(runCount);
            runningMean          = runningMean + (thisTime - oldMean) / k;
            runningS             = runningS + (thisTime - oldMean) * (thisTime - runningMean);
            runningStddev        = std::sqrt(runningS / (k - 1));
        }

        if(runCount >= 10 && runningStddev < accuracyFrac * runningMean)
        {
            break;
        }
        else if(runCount > 10000)
        {
            printf("Warning: runCount reached %zu; benchmark information below probably unreliable.\n", runCount);
            break;
        }
    }
    printf("%s:\tmean = %f\tstandard_deviation = %f\t(seconds)\truns = %zu\n", benchmarkName, runningMean, runningStddev, runCount);
}

int main(int argc, const char** argv)
{
    OpContext context;
    {
        OpConfig config;
        config.threadCount = 2;
        TEST_SUCCESS(micromeshCreateOpContext(&config, &context, &messenger));
    }

    TEST_TRUE(blockCorrectMath());
    TEST_TRUE(docExampleBlockEncoder());
    TEST_TRUE(encodeRandomBlocks());
    TEST_TRUE(perfectReencoding());
    TEST_TRUE(docExampleEncodeTwoTriangles(context));
    TEST_TRUE(perfectMeshReencoding(context));

    printf("All tests passed. Running microbenchmarks...\n");

    benchmark("encodeRandomBlocks", encodeRandomBlocks);

    micromeshDestroyOpContext(context);

    printf("Done.\n");
    return 0;
}