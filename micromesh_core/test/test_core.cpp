//
// Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//


#include <cstdlib>
#include <vector>
#include <array>
#include <unordered_set>
#include <unordered_map>
#include <string>

#include <micromesh/micromesh_gpu.h>
#include <micromesh/micromesh_operations.h>
#include <micromesh/micromesh_utils.h>
#include <micromesh/micromesh_format_types.h>
#include <micromesh_internal/micromesh_math.h>

#include <string.h>
#include <stdio.h>

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
    if((a) != micromesh::Result::eSuccess)                                                                              \
    {                                                                                                                  \
        PRINT_AND_ASSERT_FALSE("Test did not return Result::eSuccess: " #a " \n");                                      \
        return false;                                                                                                  \
    }

const char*   exampleUserData = "NVIDIA";
static size_t errorsReceived  = 0;  // For message callback tests
void          basicMessageCallback(micromesh::MessageSeverity severity, const char* message, uint32_t threadIndex, const void* userData)
{
    if(severity == micromesh::MessageSeverity::eInfo)
    {
        printf("INFO: %s\n", message);
    }
    else if(severity == micromesh::MessageSeverity::eWarning)
    {
        printf("WARNING: %s\n", message);
    }
    else if(severity == micromesh::MessageSeverity::eError)
    {
        printf("ERROR: %s\n", message);
        errorsReceived++;
    }
    // Make sure the userData pointer has been passed through correctly
    assert(userData == exampleUserData);
}

bool testSanitizers(micromesh::OpContext context)
{
    // Build the mesh topology for the following mesh (ccw):
    //
    //     2 _ 3
    //    / \ /
    //   0 _ 1
    //

    constexpr size_t                                      NUM_TRIANGLES = 2;
    constexpr size_t                                      NUM_VERTICES  = 4;
    std::array<micromesh::Vector_uint32_3, NUM_TRIANGLES> indices{0, 1, 2, 1, 3, 2};

    // Allocate space for triangleEdges, vertexEdgeRanges, and vertexTriangleRanges
    micromesh::MeshTopology topo{};
    micromesh::arraySetDataVec(topo.triangleVertices, indices);
    topo.triangleEdges.count = NUM_TRIANGLES;
    std::vector<micromesh::Vector_uint32_3> topoTriangleEdgesStorage(topo.triangleEdges.count);
    topo.triangleEdges.data = topoTriangleEdgesStorage.data();

    topo.vertexEdgeRanges.count = NUM_VERTICES;
    std::vector<micromesh::Range32> topoVertexEdgeRangesStorage(NUM_VERTICES);
    topo.vertexEdgeRanges.data = topoVertexEdgeRangesStorage.data();

    topo.vertexTriangleRanges.count = NUM_VERTICES;
    std::vector<micromesh::Range32> topoVertexTriangleRangesStorage(NUM_VERTICES);
    topo.vertexTriangleRanges.data = topoVertexTriangleRangesStorage.data();

    // Fill those 3 arrays and get sizes for remaining MeshTopology arrays
    TEST_SUCCESS(micromesh::micromeshOpBuildMeshTopologyBegin(context, &topo));

    // Allocate remaining output.
    std::vector<uint32_t> topoVertexTriangleConnectionsStorage(topo.vertexTriangleConnections.count);
    topo.vertexTriangleConnections.data = topoVertexTriangleConnectionsStorage.data();

    std::vector<uint32_t> topoVertexEdgeConectionsStorage(topo.vertexEdgeConnections.count);
    topo.vertexEdgeConnections.data = topoVertexEdgeConectionsStorage.data();

    std::vector<uint32_t> topoEdgeVerticesStorage(topo.edgeVertices.count * 2);
    topo.edgeVertices.data = topoEdgeVerticesStorage.data();

    std::vector<micromesh::Range32> topoEdgeTriangleRangesStorage(topo.edgeTriangleRanges.count);
    topo.edgeTriangleRanges.data = topoEdgeTriangleRangesStorage.data();

    std::vector<uint32_t> topoEdgeTriangleConnectionsStorage(topo.edgeTriangleConnections.count);
    topo.edgeTriangleConnections.data = topoEdgeTriangleConnectionsStorage.data();

    // Okay, now build the topology!
    TEST_SUCCESS(micromesh::micromeshOpBuildMeshTopologyEnd(context, &topo));

    std::array<uint16_t, NUM_TRIANGLES> meshSubdivTriangles{2, 0};
    std::array<uint16_t, NUM_TRIANGLES> meshSubdivTrianglesExpected{2, 1};

    micromesh::OpSanitizeSubdivLevels_input  sanitizeSubdivInput;
    micromesh::OpSanitizeSubdivLevels_output sanitizeSubdivOutput;

    sanitizeSubdivInput.meshTopo       = &topo;
    sanitizeSubdivInput.maxSubdivLevel = 3;
    arraySetFormatTypeDataVec(sanitizeSubdivInput.meshTriangleSubdivLevels, meshSubdivTriangles);
    arraySetFormatTypeDataVec(sanitizeSubdivOutput.meshTriangleSubdivLevels, meshSubdivTriangles);

    TEST_SUCCESS(micromesh::micromeshOpSanitizeSubdivLevels(context, &sanitizeSubdivInput, &sanitizeSubdivOutput));

    TEST_TRUE(memcmp(meshSubdivTriangles.data(), meshSubdivTrianglesExpected.data(), sizeof(uint16_t) * NUM_TRIANGLES) == 0);

    //
    //     2 _ 3
    //    /3\2/
    //   0 _ 1
    //

    std::array<uint8_t, NUM_TRIANGLES> meshPrimitiveFlags{0, 0};
    std::array<uint8_t, NUM_TRIANGLES> meshPrimitiveFlagsExpected{1 << 1, 0};

    micromesh::OpBuildPrimitiveFlags_input  primFlagsInput;
    micromesh::OpBuildPrimitiveFlags_output primFlagsOutput;

    primFlagsInput.meshTopo                 = &topo;
    primFlagsInput.meshTriangleSubdivLevels = sanitizeSubdivOutput.meshTriangleSubdivLevels;
    arraySetFormatTypeDataVec(primFlagsOutput.meshTrianglePrimitiveFlags, meshPrimitiveFlags);

    TEST_SUCCESS(micromesh::micromeshOpBuildPrimitiveFlags(context, &primFlagsInput, &primFlagsOutput));

    TEST_TRUE(memcmp(meshPrimitiveFlags.data(), meshPrimitiveFlagsExpected.data(), sizeof(uint8_t) * NUM_TRIANGLES) == 0);

    // TODO
    // create micromap and test edge sanitizer


    return true;
}

uint32_t generateTessellatedVertex(const micromesh::VertexGenerateInfo* vertexInfo, micromesh::VertexDedup dedupState, uint32_t threadIndex, void* beginResult, void* userData)
{
    uint32_t index;
    if (dedupState) {
        micromeshVertexDedupAppendAttribute(dedupState, sizeof(float) * 3, &vertexInfo->vertexWUVfloat);
        index = micromeshVertexDedupGetIndex(dedupState);
    }
    else {
        index = vertexInfo->nonDedupIndex;
    }

    micromesh::BaryWUV_float* vertices = reinterpret_cast<micromesh::BaryWUV_float*>(userData);
    vertices[index] = vertexInfo->vertexWUVfloat;


    return index;
}

bool testTessellate(micromesh::OpContext ctx)
{
    const uint32_t maxSubdivLevel  = 5;
    const uint32_t maxSubdivLevels = maxSubdivLevel + 1;

    std::vector<uint16_t> meshSubdivLevels(maxSubdivLevels * 8);
    std::vector<uint8_t>  meshEdgeFlags(meshSubdivLevels.size());

    for(size_t i = 0; i < meshSubdivLevels.size(); i++)
    {
        meshSubdivLevels[i] = uint16_t(i / 8);
        meshEdgeFlags[i]    = i % 8;
    }

    micromesh::OpTessellateMesh_input input;
    input.maxSubdivLevel = maxSubdivLevel;
    
    micromesh::arraySetDataVec(input.meshTriangleSubdivLevels, meshSubdivLevels);
    micromesh::arraySetDataVec(input.meshTrianglePrimitiveFlags, meshEdgeFlags);
    input.pfnGenerateVertex = &generateTessellatedVertex;

    micromesh::OpTessellateMesh_output output;

    // first pass, no deduplication
    input.useVertexDeduplication = false;

    TEST_SUCCESS(micromeshOpTessellateMeshBegin(ctx, &input, &output));

    std::vector<micromesh::Vector_uint32_3> triangleIndices(output.meshTriangleVertices.count);
    output.meshTriangleVertices.data = triangleIndices.data();

    std::vector<micromesh::BaryWUV_float> verticesNoDedup(output.vertexCount);
    input.userData = verticesNoDedup.data();

    TEST_SUCCESS(micromeshOpTessellateMeshEnd(ctx, &input, &output));

    for (uint64_t i = 0; i < triangleIndices.size() * 3; i++)
    {
        uint32_t* indices = reinterpret_cast<uint32_t*>(triangleIndices.data());
        TEST_TRUE(indices[i] < output.vertexCount);
    }

    std::unordered_set<std::string> uniqueEdgeVertices;
    size_t nonEdgeVertexCount = 0;
    for(uint64_t i = 0; i < output.vertexCount; i++)
    {
        micromesh::BaryWUV_float vertex = verticesNoDedup[i];

        // only edge vertices go through deduplication
        if (baryWUVisOnEdge(vertex))
        {
            std::string vec(sizeof(vertex), 0);
            memcpy(vec.data(), &vertex, sizeof(vertex));
            uniqueEdgeVertices.insert(vec);
        }
        else {
            nonEdgeVertexCount++;
        }
    }

    // second pass with deduplication
    input.useVertexDeduplication = true;

    std::vector<micromesh::BaryWUV_float> verticesDedup(output.vertexCount);
    input.userData = verticesDedup.data();

    TEST_SUCCESS(micromeshOpTessellateMeshBegin(ctx, &input, &output));

    TEST_SUCCESS(micromeshOpTessellateMeshEnd(ctx, &input, &output));

    for(uint64_t i = 0; i < triangleIndices.size() * 3; i++)
    {
        uint32_t* indices = reinterpret_cast<uint32_t*>(triangleIndices.data());
        TEST_TRUE(indices[i] < output.vertexCount);
    }

    TEST_TRUE(output.vertexCount == (uniqueEdgeVertices.size() + nonEdgeVertexCount));

    return true;
}

int main(int argc, const char** argv)
{
    micromesh::Result result;

    {
        micromesh::ArrayInfo_uint16     arrayInfo;
        micromesh::FormatType::r16_uint data[16];

        micromesh::arraySetFormatTypeData(arrayInfo, data, 16);
        assert(arrayInfo.format == micromesh::Format::eR16_uint);
        assert(arrayInfo.byteStride == sizeof(uint16_t));

        data[1].r = uint16_t(32);

        auto testVal = arrayTypedGetV(arrayInfo, 1);
        assert(testVal == uint16_t(32));
    }

    {
        micromesh::BaryUV_uint16 coord;
        micromesh::BaryUV_uint16 coordRet;
        uint32_t                 index;
        coord    = {0, 0};
        index    = micromesh::umajorUVtoLinear(coord, 1);
        coordRet = micromesh::umajorLinearToUV(index, 1);
        assert(coord.u == coordRet.u && coord.v == coordRet.v);

        coord    = {1, 0};
        index    = micromesh::umajorUVtoLinear(coord, 1);
        coordRet = micromesh::umajorLinearToUV(index, 1);
        assert(coord.u == coordRet.u && coord.v == coordRet.v);

        coord    = {4, 17};
        index    = micromesh::umajorUVtoLinear(coord, 5);
        coordRet = micromesh::umajorLinearToUV(index, 5);
        assert(coord.u == coordRet.u && coord.v == coordRet.v);
    }

    micromesh::MessageCallbackInfo messenger{basicMessageCallback, exampleUserData};

    micromesh::OpConfig config;
    config.threadCount = 2;

    micromesh::OpContext context;
    result = micromeshCreateOpContext(&config, &context, &messenger);
    assert(result == micromesh::Result::eSuccess);

    // Tests that message callbacks work.
    {
        printf("----------\n");
        printf("There should be four errors below:\n");
        // Do the different null tests work?
        result = micromesh::micromeshCreateOpContext(nullptr, nullptr, nullptr);
        result = micromesh::micromeshCreateOpContext(nullptr, nullptr, &messenger);
        result = micromesh::micromeshOpBuildMeshTopologyEnd(nullptr, nullptr);
        result = micromesh::micromeshOpBuildMeshTopologyBegin(context, nullptr);
        // out of sequence
        result = micromesh::micromeshOpBuildMeshTopologyEnd(context, (micromesh::MeshTopology*)1);

        // Does formatting work? This results in a message with a %u specifier.
        micromesh::OpConfig  badConfig;
        micromesh::OpContext badContext;
        badConfig.contextType = (micromesh::OpContextType)123;
        result                = micromesh::micromeshCreateOpContext(&badConfig, &badContext, &messenger);
        assert(errorsReceived == 4);
        printf("----------\n");
        errorsReceived = 0;
    }

    // do stuff

    testTessellate(context);

    testSanitizers(context);

    micromeshDestroyOpContext(context);
    return EXIT_SUCCESS;
}
