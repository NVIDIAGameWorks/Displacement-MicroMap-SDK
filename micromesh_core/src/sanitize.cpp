//
// Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#include <micromesh/micromesh_operations.h>
#include <micromesh/micromesh_utils.h>
#include <micromesh_internal/micromesh_context.h>
#include <micromesh_internal/micromesh_containers.h>


#include <algorithm>
#include <string.h>  // memset

namespace micromesh
{
MICROMESH_API Result MICROMESH_CALL micromeshOpBuildPrimitiveFlags(OpContext                          ctx,
                                                                   const OpBuildPrimitiveFlags_input* input,
                                                                   OpBuildPrimitiveFlags_output*      output)
{
    CHECK_CTX_NONNULL(ctx);
    CHECK_NONNULL(ctx, input);
    CHECK_NONNULL(ctx, output);
    CHECK_NONNULL(ctx, input->meshTopo);
    CHECK_ARRAYVALIDTYPED(ctx, input->meshTriangleSubdivLevels);
    CHECK_ARRAYVALIDTYPED(ctx, output->meshTrianglePrimitiveFlags);
    CHECK_CTX_BEGIN(ctx);

    if(input->meshTriangleSubdivLevels.count != output->meshTrianglePrimitiveFlags.count)
    {
        LOGE(ctx, "input.meshTriangleSubdivLevels.count and output->meshTrianglePrimitiveFlags.count mismatch");
        return Result::eInvalidRange;
    }

    if(input->meshTriangleSubdivLevels.count != input->meshTopo->triangleVertices.count)
    {
        LOGE(ctx, "input.meshTriangleSubdivLevels.count and input->meshTopo->triangleVertices.count mismatch");
        return Result::eInvalidRange;
    }

    uint32_t errorIdx = INVALID_INDEX;

    MeshTopologyUtil topoUtil(*input->meshTopo);

    ctx->parallel_item_ranges(input->meshTriangleSubdivLevels.count, [&](uint64_t idxFirst, uint64_t idxLast,
                                                                         uint32_t threadIndex, void* userData) {
        for(uint64_t idx = idxFirst; idx < idxLast; idx++)
        {
            uint16_t subdivLevel = arrayTypedGetV(input->meshTriangleSubdivLevels, idx);
            uint8_t  flag        = 0;

            // skip degenerate triangles
            if(meshIsTriangleDegenerate(topoUtil.getTriangleVertices(uint32_t(idx))))
            {
                arrayTypedSetV(output->meshTrianglePrimitiveFlags, idx, 0);
                continue;
            }

            // compare against neighbors
            Vector_uint32_3 triangleEdges = topoUtil.getTriangleEdges(uint32_t(idx));
            for(uint32_t e = 0; e < 3; e++)
            {
                uint32_t        edgeTrianglesCount;
                const uint32_t* edgeTriangles = topoUtil.getEdgeTriangles((&triangleEdges.x)[e], edgeTrianglesCount);
                for(uint32_t t = 0; t < edgeTrianglesCount; t++)
                {
                    uint32_t edgeTri    = topoUtil.getEdgeTriangle(edgeTriangles, t);
                    uint16_t otherLevel = arrayTypedGetV(input->meshTriangleSubdivLevels, edgeTri);
                    if(otherLevel == subdivLevel - 1)
                    {
                        flag |= 1 << e;
                    }
                    else if(otherLevel < subdivLevel)
                    {
                        errorIdx = uint32_t(idx);
                    }
                }
            }
            arrayTypedSetV(output->meshTrianglePrimitiveFlags, idx, flag);
        }
    });

    if(errorIdx != INVALID_INDEX)
    {
        LOGE(ctx, "found triangle with neighbor of less than (subvision level - 1) %d", errorIdx);
        return Result::eFailure;
    }

    return Result::eSuccess;
}

MICROMESH_API Result MICROMESH_CALL micromeshOpSanitizeSubdivLevels(OpContext                           ctx,
                                                                    const OpSanitizeSubdivLevels_input* input,
                                                                    OpSanitizeSubdivLevels_output*      output)
{
    CHECK_CTX_NONNULL(ctx);
    CHECK_NONNULL(ctx, input);
    CHECK_NONNULL(ctx, input->meshTopo);
    CHECK_NONNULL(ctx, output);
    CHECK_ARRAYVALIDTYPED(ctx, input->meshTriangleSubdivLevels);
    CHECK_ARRAYVALIDTYPED(ctx, output->meshTriangleSubdivLevels);
    CHECK_CTX_BEGIN(ctx);

    if(input->meshTriangleSubdivLevels.count != output->meshTriangleSubdivLevels.count)
    {
        LOGE(ctx, "input.meshTriangleSubdivLevels.count and output->meshTriangleSubdivLevels.count mismatch");
        return Result::eInvalidRange;
    }

    if(input->meshTriangleSubdivLevels.count != input->meshTopo->triangleVertices.count)
    {
        LOGE(ctx, "input.meshTriangleSubdivLevels.count and input->meshTopo->triangleVertices.count mismatch");
        return Result::eInvalidRange;
    }

    container::vector<uint16_t> maxLevel;

    if(arrayIsValid(input->meshTriangleMappings) && !arrayIsEmpty(input->meshTriangleMappings))
    {
        if(input->meshTriangleSubdivLevels.count != input->meshTriangleMappings.count)
        {
            LOGE(ctx, "input.meshTriangleSubdivLevels.count and input->meshTriangleMappings.count mismatch");
            return Result::eInvalidRange;
        }

        // ensure meshTriangleSubdivLevels is equal for all triangles mapping to same output

        maxLevel.resize(input->meshTriangleMappings.count, 0);

        for(uint64_t i = 0; i < input->meshTriangleMappings.count; i++)
        {
            uint16_t level   = arrayGetV<uint16_t>(input->meshTriangleSubdivLevels, i);
            uint32_t mapping = arrayGetV<uint32_t>(input->meshTriangleMappings, i);
            if(mapping != INVALID_INDEX)
            {
                maxLevel[mapping] = std::max(maxLevel[mapping], level);
            }
        }

        for(uint64_t i = 0; i < input->meshTriangleMappings.count; i++)
        {
            uint32_t mapping = arrayGetV<uint32_t>(input->meshTriangleMappings, i);
            uint16_t level;
            if(mapping != INVALID_INDEX)
            {
                level = maxLevel[mapping];
            }
            else
            {
                level = arrayGetV<uint16_t>(input->meshTriangleSubdivLevels, i);
            }

            arraySetV(output->meshTriangleSubdivLevels, i, level);
        }
    }
    else if(!arrayIsEqual(input->meshTriangleSubdivLevels, output->meshTriangleSubdivLevels))
    {
        ctx->arrayCopy<uint16_t>(output->meshTriangleSubdivLevels, input->meshTriangleSubdivLevels);
    }

    MeshTopologyUtil topoUtil(*input->meshTopo);

    container::vector<uint32_t> threadMinLevels(ctx->getThreadCount(), ~0);

    for(uint32_t lvl = 0; lvl < input->maxSubdivLevel - std::min(input->maxSubdivLevel, 1u); lvl++)
    {
        uint32_t testLevel = input->maxSubdivLevel - lvl;
        ctx->parallel_item_ranges(input->meshTriangleSubdivLevels.count, [&](uint64_t idxFirst, uint64_t idxLast,
                                                                             uint32_t threadIndex, void* userData) {
            for(uint64_t idx = idxFirst; idx < idxLast; idx++)
            {
                // skip degenerate triangles
                if(meshIsTriangleDegenerate(topoUtil.getTriangleVertices(uint32_t(idx))))
                {
                    continue;
                }

                uint32_t subdivLevel = arrayGetV<uint16_t>(output->meshTriangleSubdivLevels, idx);
                if(testLevel == subdivLevel)
                {
                    uint16_t minLevel = uint16_t(testLevel - 1);
                    // compare against neighbors
                    Vector_uint32_3 triangleEdges = topoUtil.getTriangleEdges(uint32_t(idx));

                    for(uint32_t e = 0; e < 3; e++)
                    {
                        uint32_t        edgeTrianglesCount;
                        const uint32_t* edgeTriangles = topoUtil.getEdgeTriangles((&triangleEdges.x)[e], edgeTrianglesCount);
                        for(uint32_t t = 0; t < edgeTrianglesCount; t++)
                        {
                            uint32_t edgeTri = topoUtil.getEdgeTriangle(edgeTriangles, t);

                            if(arrayGetV<uint16_t>(output->meshTriangleSubdivLevels, edgeTri) < minLevel)
                            {
                                // this write may be done by multiple threads, but guaranteed with same value
                                arraySetV(output->meshTriangleSubdivLevels, edgeTri, minLevel);
                                threadMinLevels[threadIndex] = std::min(threadMinLevels[threadIndex], uint32_t(minLevel));
                            }
                        }
                    }
                }
            }
        });
    }

    output->minSubdivLevel = ~0;
    for(uint32_t i = 0; i < ctx->getThreadCount(); i++)
    {
        output->minSubdivLevel = std::min(output->minSubdivLevel, threadMinLevels[i]);
    }

    if(arrayIsValid(input->meshTriangleMappings) && !arrayIsEmpty(input->meshTriangleMappings))
    {
        for(uint16_t& v : maxLevel)
        {
            v = uint16_t(0xFFFF);
        }

        for(uint64_t i = 0; i < input->meshTriangleMappings.count; i++)
        {
            uint16_t level   = arrayGetV<uint16_t>(output->meshTriangleSubdivLevels, i);
            uint32_t mapping = arrayGetV<uint32_t>(input->meshTriangleMappings, i);
            if(mapping != INVALID_INDEX)
            {
                if(maxLevel[mapping] == 0xFFFF)
                {
                    maxLevel[mapping] = level;
                }
                if(maxLevel[mapping] != level)
                {
                    LOGE(ctx,
                         "after sanitization the mapping buffer causes inconsistent subdivision level for triangle %zd", i);
                    return Result::eFailure;
                }
            }
        }
    }

    return Result::eSuccess;
}

static inline BaryUV_uint16 getValueUV(const uint32_t subdivLevel, const Vector_uint32_3 triangleVertices, uint32_t vtx)
{
    uint16_t      baryMax = uint16_t(1 << subdivLevel);
    BaryUV_uint16 uv      = {0, 0};

    if(triangleVertices.x == vtx)
    {
        //  w = baryMax;
    }
    else if(triangleVertices.y == vtx)
    {
        uv.u = baryMax;
    }
    else if(triangleVertices.z == vtx)
    {
        uv.v = baryMax;
    }

    return uv;
}

static inline uint32_t getValueIdx(const MicromapLayout& layout, const uint32_t subdivLevel, const Vector_uint32_3 triangleVertices, uint32_t vtx)
{
    BaryUV_uint16 uv = getValueUV(subdivLevel, triangleVertices, vtx);

    return layout.pfnGetMicroVertexIndex(uv.u, uv.v, subdivLevel, layout.userData);
}

static inline BaryUV_uint16 getValueUV(const uint32_t        subdivLevel,
                                       const Vector_uint32_3 triangleVertices,
                                       const Vector_uint32_2 edgeVertices,
                                       uint32_t              idx)
{
    uint32_t baryMax = (1 << subdivLevel);
    uint32_t u       = 0;
    uint32_t v       = 0;

    // each triangle vertex can be either
    // the startpoint of the edge ...

    if(triangleVertices.x == edgeVertices.x)
    {
        //  w = baryMax - idx;
    }
    else if(triangleVertices.y == edgeVertices.x)
    {
        u = baryMax - idx;
    }
    else if(triangleVertices.z == edgeVertices.x)
    {
        v = baryMax - idx;
    }

    // ... or the endpoint of the edge
    // but not both

    if(triangleVertices.x == edgeVertices.y)
    {
        //  w = idx;
    }
    else if(triangleVertices.y == edgeVertices.y)
    {
        u = idx;
    }
    else if(triangleVertices.z == edgeVertices.y)
    {
        v = idx;
    }

    return {uint16_t(u), uint16_t(v)};
}

static inline uint32_t getValueIdx(const MicromapLayout& layout,
                                   const uint32_t        subdivLevel,
                                   const Vector_uint32_3 triangleVertices,
                                   const Vector_uint32_2 edgeVertices,
                                   uint32_t              idx)
{
    BaryUV_uint16 uv = getValueUV(subdivLevel, triangleVertices, edgeVertices, idx);

    return layout.pfnGetMicroVertexIndex(uv.u, uv.v, subdivLevel, layout.userData);
}

MICROMESH_API Result MICROMESH_CALL micromeshOpSanitizeEdgeValues(OpContext ctx, const OpSanitizeEdgeValues_input* input, Micromap* modified)
{
    CHECK_CTX_NONNULL(ctx);
    CHECK_NONNULL(ctx, input);
    CHECK_NONNULL(ctx, input->meshTopology);
    CHECK_NONNULL(ctx, modified);
    CHECK_CTX_BEGIN(ctx);

    const MeshTopology* topology = input->meshTopology;

    FormatInfo formatInfo;
    Result     result = micromeshFormatGetInfo(modified->values.format, &formatInfo);

    if(result != Result::eSuccess || formatInfo.channelType != ChannelType::eSfloat || formatInfo.isCompressedOrPacked)
    {
        LOGE(ctx, "Micromap format was not uncompressed SFLOAT");
        return Result::eInvalidFormat;
    }

    if(modified->frequency != Frequency::ePerMicroVertex)
    {
        LOGE(ctx, "Micromap frequency must be ePerMicroVertex");
        return Result::eInvalidFrequency;
    }

    bool useMapping = arrayIsValid(input->meshTriangleMappings);
    if(useMapping && input->meshTriangleMappings.count != topology->triangleVertices.count)
    {
        LOGE(ctx, "input meshTriangleMappings mismatches topology triangle count");
        return Result::eInvalidRange;
    }

    MeshTopologyUtil topoUtil(*topology);
    uint32_t         channelCount = formatInfo.channelCount;

    if(topology->maxVertexTriangleValence > 0xFFFF)
    {
        LOGW(ctx, "input mesh has very high maxVertexTriangleValence");
    }

    container::vector<float*> scratchPointers(
        ctx->m_config.threadCount * std::max(topology->maxVertexTriangleValence, topology->maxEdgeTriangleValence));

    // sanitize all depths along corner vertices and shared edges
    //
    // over all vertices
    ctx->parallel_items(topology->vertexTriangleRanges.count, [&](uint64_t workIdx, uint32_t threadIdx, void* userData) {
        uint32_t vertexIndex = uint32_t(workIdx);

        uint32_t        vertexTrianglesCount;
        const uint32_t* vertexTriangles = topoUtil.getVertexTriangles(vertexIndex, vertexTrianglesCount);

        // skip simple
        if(vertexTrianglesCount == 1)
            return;

        float**  valuePointers      = scratchPointers.data() + (threadIdx * topology->maxVertexTriangleValence);
        uint32_t valuePointersCount = 0;

        MicromapValue value;

        // unify over values from other triangles
        for(uint32_t t = 0; t < vertexTrianglesCount; t++)
        {
            uint32_t        topoTri         = topoUtil.getVertexTriangle(vertexTriangles, t);
            Vector_uint32_3 topoTriVertices = topoUtil.getTriangleVertices(topoTri);

            if(meshIsTriangleDegenerate(topoTriVertices))
                continue;

            uint32_t micromapTri = meshGetTriangleMapping(input->meshTriangleMappings, topoTri);
            if(micromapTri == INVALID_INDEX)
                continue;

            uint32_t subdivLevel = arrayGetV<uint16_t>(modified->triangleSubdivLevels, micromapTri);
            // finds which of the 3 indices matches vtxIndex, and computes barycentric
            // coord accordingly, and then gets the value index from the barycentric map
            // level.
            uint32_t valIdx = getValueIdx(modified->layout, subdivLevel, topoTriVertices, vertexIndex);
            float*   valPtr = micromapGetTriangleValue<float>(*modified, micromapTri, valIdx);

            valuePointers[valuePointersCount++] = valPtr;
            // avoid extra loop and take 4 channel max
            value.value_float[0] += valPtr[0 % channelCount];
            value.value_float[1] += valPtr[1 % channelCount];
            value.value_float[2] += valPtr[2 % channelCount];
            value.value_float[3] += valPtr[3 % channelCount];
        }

        if(valuePointersCount)
        {
            // average
            float norm = float(valuePointersCount);

            value.value_float[0] /= norm;
            value.value_float[1] /= norm;
            value.value_float[2] /= norm;
            value.value_float[3] /= norm;

            // write back average to all shared triangles
            for(uint32_t i = 0; i < valuePointersCount; i++)
            {
                float* valPtr = valuePointers[i];

                valPtr[0 % channelCount] = value.value_float[0 % channelCount];
                valPtr[1 % channelCount] = value.value_float[1 % channelCount];
                valPtr[2 % channelCount] = value.value_float[2 % channelCount];
                valPtr[3 % channelCount] = value.value_float[3 % channelCount];
            }
        }
    });

    // over all edges
    ctx->parallel_items(topology->edgeVertices.count, [&](uint64_t workIdx, uint32_t threadIdx, void* userData) {
        uint32_t edgeIndex = uint32_t(workIdx);

        Vector_uint32_2 edgeVertices = topoUtil.getEdgeVertices(edgeIndex);

        uint32_t        edgeTrianglesCount;
        const uint32_t* edgeTriangles = topoUtil.getEdgeTriangles(edgeIndex, edgeTrianglesCount);

        // skip simple
        if(edgeTrianglesCount == 1)
            return;

        uint32_t refLevel = 0;

        // need to find highest subdivision
        // that becomes our reference stepping rate along the edge
        for(uint32_t t = 0; t < edgeTrianglesCount; t++)
        {
            uint32_t        topoTri         = topoUtil.getEdgeTriangle(edgeTriangles, t);
            Vector_uint32_3 topoTriVertices = topoUtil.getTriangleVertices(topoTri);

            if(meshIsTriangleDegenerate(topoTriVertices))
                continue;

            uint32_t micromapTri = meshGetTriangleMapping(input->meshTriangleMappings, topoTri);
            if(micromapTri == INVALID_INDEX)
                continue;
            uint32_t subdivLevel = arrayGetV<uint16_t>(modified->triangleSubdivLevels, micromapTri);
            refLevel             = std::max(refLevel, subdivLevel);
        }
        uint32_t refCount = 1 << refLevel;

        float** valuePointers = scratchPointers.data() + (threadIdx * topology->maxEdgeTriangleValence);

        // Walk along reference edge in the direction
        // of edgeIndices[0] to edgeIndices[1].
        // Skip first and last, as these are corners
        for(uint32_t i = 1; i < refCount; i++)
        {
            MicromapValue value;

            uint32_t valuePointersCount = 0;

            for(uint32_t t = 0; t < edgeTrianglesCount; t++)
            {
                uint32_t        topoTri         = topoUtil.getEdgeTriangle(edgeTriangles, t);
                Vector_uint32_3 topoTriVertices = topoUtil.getTriangleVertices(topoTri);

                if(meshIsTriangleDegenerate(topoTriVertices))
                    continue;

                uint32_t micromapTri = meshGetTriangleMapping(input->meshTriangleMappings, topoTri);
                if(micromapTri == INVALID_INDEX)
                    continue;

                uint32_t subdivLevel = arrayGetV<uint16_t>(modified->triangleSubdivLevels, micromapTri);

                // only every idxDiv is valid if subdivision levels are different,
                // as the reference would make too many iterations compared to the lower
                // resolution triangle.
                // Also need to divide i along edge for a triangle of lower subdivision
                // (e.g. if triangle has half the segments, need to divide i by 2)
                uint32_t levelDelta = refLevel - subdivLevel;
                uint32_t iDiv       = 1 << levelDelta;
                if(i % iDiv == 0)
                {
                    // find ith value along the provided edge in the target triangle
                    // computes barycentric and looks up value index in barycentric map level.
                    uint32_t valIdx = getValueIdx(modified->layout, subdivLevel, topoTriVertices, edgeVertices, i / iDiv);
                    float*   valPtr = micromapGetTriangleValue<float>(*modified, micromapTri, valIdx);

                    valuePointers[valuePointersCount++] = valPtr;

                    value.value_float[0] += valPtr[0 % channelCount];
                    value.value_float[1] += valPtr[1 % channelCount];
                    value.value_float[2] += valPtr[2 % channelCount];
                    value.value_float[3] += valPtr[3 % channelCount];
                }
            }

            if(valuePointersCount)
            {
                // average
                float norm = float(valuePointersCount);

                value.value_float[0] /= norm;
                value.value_float[1] /= norm;
                value.value_float[2] /= norm;
                value.value_float[3] /= norm;

                // write back average to all shared triangles
                for(uint32_t i = 0; i < valuePointersCount; i++)
                {
                    float* valPtr = valuePointers[i];

                    valPtr[0 % channelCount] = value.value_float[0 % channelCount];
                    valPtr[1 % channelCount] = value.value_float[1 % channelCount];
                    valPtr[2 % channelCount] = value.value_float[2 % channelCount];
                    valPtr[3 % channelCount] = value.value_float[3 % channelCount];
                }
            }
        }
    });

    return Result::eSuccess;
}


MICROMESH_API uint32_t MICROMESH_CALL micromeshMeshTopologyGetVertexSanitizationList(const MeshTopology* meshTopo,
                                                                                     const ArrayInfo_uint16* triangleSubdivLevels,
                                                                                     const ArrayInfo_uint32* meshTriangleMappings,
                                                                                     MicroVertexInfo queryVertex,
                                                                                     uint32_t        outputReserveCount,
                                                                                     MicroVertexInfo* outputVertices)
{
    assert(meshTopo && triangleSubdivLevels);

    MeshTopologyUtil topo(*meshTopo);

    uint32_t triangleMapIndex = meshTriangleMappings ? arrayGetV<uint32_t>(*meshTriangleMappings, queryVertex.triangleIndex) :
                                                       queryVertex.triangleIndex;
    uint32_t triangleSubdiv = arrayGetV<uint16_t>(*triangleSubdivLevels, triangleMapIndex);

    Vector_uint32_3 triangleVertices = topo.getTriangleVertices(queryVertex.triangleIndex);

    BaryWUV_uint16 vertexWUV = baryUVtoWUV_uint(queryVertex.vertexUV, triangleSubdiv);
    uint32_t       maxCoord  = 1 << triangleSubdiv;

    uint32_t vertexIndex = INVALID_INDEX;
    uint32_t resultCount = 0;

    if(vertexWUV.w == maxCoord)
    {
        vertexIndex = triangleVertices.x;
    }
    else if(vertexWUV.u == maxCoord)
    {
        vertexIndex = triangleVertices.y;
    }
    else if(vertexWUV.v == maxCoord)
    {
        vertexIndex = triangleVertices.z;
    }

    if(vertexIndex != INVALID_INDEX)
    {
        // we are on the corner

        const uint32_t* vertexTriangles = topo.getVertexTriangles(vertexIndex, resultCount);

        uint32_t count = resultCount < outputReserveCount ? resultCount : outputReserveCount;
        for(uint32_t i = 0; i < count; i++)
        {
            uint32_t otherTriangle = topo.getVertexTriangle(vertexTriangles, i);
            uint32_t otherMapIndex = meshTriangleMappings ? arrayGetV<uint32_t>(*meshTriangleMappings, otherTriangle) : otherTriangle;
            uint32_t        otherSubdiv  = arrayGetV<uint16_t>(*triangleSubdivLevels, otherMapIndex);
            Vector_uint32_3 otherIndices = topo.getTriangleVertices(otherTriangle);
            BaryUV_uint16   otherUV      = getValueUV(otherSubdiv, otherIndices, vertexIndex);

            outputVertices[i].triangleIndex = otherTriangle;
            outputVertices[i].vertexUV      = otherUV;
        }

        return resultCount;
    }
    else
    {
        uint32_t        edgeIndex     = INVALID_INDEX;
        uint32_t        edgeSub       = 0;
        uint32_t        edgeFirst     = 0;
        Vector_uint32_3 triangleEdges = topo.getTriangleEdges(queryVertex.triangleIndex);

        // we are on the edge
        // iterate all triangles on the edge
        if(vertexWUV.v == 0)
        {
            // w to u
            edgeIndex = triangleEdges.x;
            edgeSub   = vertexWUV.u;
            edgeFirst = triangleVertices.x;
        }
        else if(vertexWUV.w == 0)
        {
            // u to v
            edgeIndex = triangleEdges.y;
            edgeSub   = vertexWUV.v;
            edgeFirst = triangleVertices.y;
        }
        else if(vertexWUV.u == 0)
        {
            // v to w
            edgeIndex = triangleEdges.z;
            edgeSub   = vertexWUV.w;
            edgeFirst = triangleVertices.z;
        }
        else
        {
            // not on corner, nor edge
            return 0;
        }


        assert(edgeIndex != INVALID_INDEX);

        uint32_t        count;
        const uint32_t* edgeTriangles = topo.getEdgeTriangles(edgeIndex, count);
        Vector_uint32_2 edgeVertices  = topo.getEdgeVertices(edgeIndex);

        // the edge might be stored opposite to the local triangle
        if(edgeVertices.x != edgeFirst)
        {
            assert(edgeVertices.y == edgeFirst);
            // reverse order
            edgeSub = maxCoord - edgeSub;
        }

        for(uint32_t i = 0; i < count; i++)
        {
            uint32_t otherTriangle = topo.getEdgeTriangle(edgeTriangles, i);
            uint32_t otherMapIndex = meshTriangleMappings ? arrayGetV<uint32_t>(*meshTriangleMappings, otherTriangle) : otherTriangle;
            uint32_t        otherSubdiv  = arrayGetV<uint16_t>(*triangleSubdivLevels, otherMapIndex);
            Vector_uint32_3 otherIndices = topo.getTriangleVertices(otherTriangle);
            uint32_t        otherSub     = INVALID_INDEX;

            if(otherSubdiv >= triangleSubdiv)
            {
                // we are equal/ smaller resolution triangle and therefore
                // can be always turned into other's coordinate frame

                uint32_t mul = 1 << (otherSubdiv - triangleSubdiv);
                otherSub     = edgeSub * mul;
            }
            else
            {
                // we are the higher resolution triangle and therefore
                // the vertex may not exist in the lower-res triangle at all
                uint32_t div = 1 << (triangleSubdiv - otherSubdiv);
                if(edgeSub % div == 0)
                {
                    otherSub = edgeSub / div;
                }
                else
                {
                    continue;
                }
            }

            if(resultCount < outputReserveCount)
            {
                BaryUV_uint16 otherUV = getValueUV(otherSubdiv, otherIndices, edgeVertices, otherSub);

                outputVertices[resultCount].triangleIndex = otherTriangle;
                outputVertices[resultCount].vertexUV      = otherUV;
            }

            resultCount++;
        }

        return resultCount;
    }
}

}  // namespace micromesh
