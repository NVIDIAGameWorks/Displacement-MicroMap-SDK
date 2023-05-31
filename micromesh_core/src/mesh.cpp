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
#include <micromesh_internal/micromesh_math.h>

#include <limits>
#include <string.h>

namespace micromesh
{
namespace
{
class BuildMeshTopoPayload
{
  public:
    struct EdgeInfo
    {
        Vector_uint32_2 edge;
        uint32_t        triangleCount;
    };

    BuildMeshTopoPayload(MeshTopology* output)
        : m_topo(output)
    {
        Range32 rzero = {0, 0};
        arrayFill(output->vertexEdgeRanges, rzero);
        arrayFill(output->vertexTriangleRanges, rzero);

        uint32_t vertexCount = uint32_t(output->vertexEdgeRanges.count);
        m_edges.reserve(vertexCount * 2);
        m_map.reserve(vertexCount * 2);
    }

    MeshTopology* m_topo;

    uint64_t m_vertexEdgeConnections     = 0;
    uint64_t m_vertexTriangleConnections = 0;
    uint64_t m_edgeTriangleConnections   = 0;

    uint32_t m_maxVertexTriangleValence = 0;
    uint32_t m_maxVertexEdgeValence     = 0;
    uint32_t m_maxEdgeTriangleValence   = 0;

    bool m_isNonManifold = false;

    container::unordered_map<uint64_t, uint32_t> m_map;
    container::vector<EdgeInfo>                  m_edges;

    void countVertexTriangle(uint32_t v)
    {
        Range32* vertexTriangleRanges = arrayGet<Range32>(m_topo->vertexTriangleRanges, v);
        vertexTriangleRanges->count++;
        m_vertexTriangleConnections++;
    }

    void countVertexEdge(uint32_t v)
    {
        Range32* vertexEdgeRanges = arrayGet<Range32>(m_topo->vertexEdgeRanges, v);
        vertexEdgeRanges->count++;
        m_vertexEdgeConnections++;
    }

    uint32_t addEdge(uint32_t a, uint32_t b)
    {
        assert(a != b);

        // they way we fill in things every edge is used with one triangle at least
        m_edgeTriangleConnections++;

        uint64_t edge64;
        if(b < a)
        {
            edge64 = uint64_t(b) | (uint64_t(a) << 32);
        }
        else
        {
            edge64 = uint64_t(a) | (uint64_t(b) << 32);
        }

        auto it = m_map.find(edge64);
        if(it != m_map.end())
        {
            uint32_t idx = it->second;

            if(m_edges[idx].edge.x == a)
            {
                // in theory an edge is used by up to two triangles
                // and with opposite vertex ordering, if the winding
                // is consistent
                m_isNonManifold = true;
            }

            m_edges[idx].triangleCount++;
            return idx;
        }

        countVertexEdge(a);
        countVertexEdge(b);

        uint32_t idx = uint32_t(m_edges.size());
        m_edges.push_back({{a, b}, 1});
        m_map.insert({edge64, idx});

        return idx;
    }

    void addTriangle(uint32_t triIndex)
    {
        const Vector_uint32_3* triVertices = arrayGet<Vector_uint32_3>((const ArrayInfo&)m_topo->triangleVertices, triIndex);
        Vector_uint32_3*       triEdges = arrayGet<Vector_uint32_3>((ArrayInfo&)m_topo->triangleEdges, triIndex);

        uint32_t a = triVertices->x;
        uint32_t b = triVertices->y;
        uint32_t c = triVertices->z;

        if(meshIsTriangleDegenerate(*triVertices))
        {
            // skip degenerated triangles
            triEdges->x = INVALID_INDEX;
            triEdges->y = INVALID_INDEX;
            triEdges->z = INVALID_INDEX;
        }
        else
        {
            triEdges->x = addEdge(a, b);
            triEdges->y = addEdge(b, c);
            triEdges->z = addEdge(c, a);

            countVertexTriangle(a);
            countVertexTriangle(b);
            countVertexTriangle(c);
        }
    }

    void fillVertexEdge(uint32_t vertexIndex, uint32_t edgeIndex)
    {
        Range32*  vertexEdgeRanges = arrayGet<Range32>(m_topo->vertexEdgeRanges, vertexIndex);
        uint32_t* vertexTriangle =
            arrayGet<uint32_t>(m_topo->vertexEdgeConnections, vertexEdgeRanges->first + vertexEdgeRanges->count);

        *vertexTriangle = edgeIndex;

        vertexEdgeRanges->count++;
        m_maxVertexEdgeValence = std::max(m_maxVertexEdgeValence, vertexEdgeRanges->count);
    }

    void fillVertexTriangle(uint32_t vertexIndex, uint32_t triIndex)
    {
        Range32*  vertexTriangleRanges = arrayGet<Range32>(m_topo->vertexTriangleRanges, vertexIndex);
        uint32_t* vertexTriangle =
            arrayGet<uint32_t>(m_topo->vertexTriangleConnections, vertexTriangleRanges->first + vertexTriangleRanges->count);
        *vertexTriangle = triIndex;

        vertexTriangleRanges->count++;
        m_maxVertexTriangleValence = std::max(m_maxVertexTriangleValence, vertexTriangleRanges->count);
    }

    void fillEdgeTriangle(uint32_t edgeIndex, uint32_t triIndex)
    {
        Range32*  edgeTriangleRanges = arrayGet<Range32>(m_topo->edgeTriangleRanges, edgeIndex);
        uint32_t* edgeTriangle =
            arrayGet<uint32_t>(m_topo->edgeTriangleConnections, edgeTriangleRanges->first + edgeTriangleRanges->count);
        *edgeTriangle = triIndex;

        edgeTriangleRanges->count++;
        m_maxEdgeTriangleValence = std::max(m_maxEdgeTriangleValence, edgeTriangleRanges->count);
    }

    void fillTriangle(uint32_t triIndex)
    {
        const Vector_uint32_3* triVertices = arrayGet<Vector_uint32_3>((const ArrayInfo&)m_topo->triangleVertices, triIndex);
        const Vector_uint32_3* triEdges = arrayGet<Vector_uint32_3>((const ArrayInfo&)m_topo->triangleEdges, triIndex);

        uint32_t a = triVertices->x;
        uint32_t b = triVertices->y;
        uint32_t c = triVertices->z;

        // skip degenerated triangles
        if(meshIsTriangleDegenerate(*triVertices))
            return;

        fillEdgeTriangle(triEdges->x, triIndex);
        fillEdgeTriangle(triEdges->y, triIndex);
        fillEdgeTriangle(triEdges->z, triIndex);

        fillVertexTriangle(triVertices->x, triIndex);
        fillVertexTriangle(triVertices->y, triIndex);
        fillVertexTriangle(triVertices->z, triIndex);
    }

    void fillEdges()
    {
        for(size_t i = 0; i < m_edges.size(); i++)
        {
            Vector_uint32_2* edgeVertices = arrayGet<Vector_uint32_2>((ArrayInfo&)m_topo->edgeVertices, i);
            *edgeVertices                 = m_edges[i].edge;

            fillVertexEdge(m_edges[i].edge.x, uint32_t(i));
            fillVertexEdge(m_edges[i].edge.y, uint32_t(i));
        }
    }

    static void deleter(void* data) { delete reinterpret_cast<BuildMeshTopoPayload*>(data); }
};
}  // namespace

MICROMESH_API Result MICROMESH_CALL micromeshOpBuildMeshTopologyBegin(OpContext ctx, MeshTopology* output)
{
    CHECK_CTX_NONNULL(ctx);
    CHECK_NONNULL(ctx, output);
    CHECK_CTX_BEGIN(ctx);
    CHECK_ARRAYVALIDTYPED(ctx, output->triangleVertices);
    CHECK_ARRAYVALIDTYPED(ctx, output->triangleEdges);
    CHECK_NONNULL(ctx, output->vertexEdgeRanges.data);
    CHECK_NONNULL(ctx, output->vertexTriangleRanges.data);

    if(output->vertexEdgeRanges.count > MAX_UINT32_COUNT || output->triangleVertices.count > MAX_UINT32_COUNT)
    {
        LOGE(ctx, "vertexCount and triangleCount must be within uint32_t limits");
        return Result::eInvalidRange;
    }

    uint32_t vertexCount   = uint32_t(output->vertexEdgeRanges.count);
    uint32_t triangleCount = uint32_t(output->triangleVertices.count);

    BuildMeshTopoPayload* payload = new BuildMeshTopoPayload(output);

    for(uint32_t i = 0; i < triangleCount; i++)
    {
        const Vector_uint32_3* triIndices = arrayGet<Vector_uint32_3>((const ArrayInfo&)output->triangleVertices, i);
        payload->addTriangle(i);
    }

    output->vertexTriangleConnections.count = payload->m_vertexTriangleConnections;
    output->vertexEdgeConnections.count     = payload->m_vertexEdgeConnections;
    output->edgeVertices.count              = payload->m_edges.size();
    output->edgeTriangleRanges.count        = payload->m_edges.size();
    output->edgeTriangleConnections.count   = payload->m_edgeTriangleConnections;

    if(output->edgeVertices.count > MAX_UINT32_COUNT || output->vertexTriangleConnections.count > MAX_UINT32_COUNT
       || output->vertexEdgeConnections.count > MAX_UINT32_COUNT || output->edgeTriangleConnections.count > MAX_UINT32_COUNT)
    {
        delete payload;

        LOGE(ctx, "resulting arrays must be within uint32_t limits");
        return Result::eInvalidRange;
    }

    ctx->setPayload(payload, BuildMeshTopoPayload::deleter);
    ctx->setNextSequenceFn(&micromeshOpBuildMeshTopologyEnd);

    return Result::eSuccess;
}
// output must contain content of the arrays computed in preprocess
MICROMESH_API Result MICROMESH_CALL micromeshOpBuildMeshTopologyEnd(OpContext ctx, MeshTopology* output)
{
    CHECK_CTX_NONNULL(ctx);
    CHECK_NONNULL(ctx, output);
    CHECK_CTX_END(ctx, &micromeshOpBuildMeshTopologyEnd);

    BuildMeshTopoPayload* payload = reinterpret_cast<BuildMeshTopoPayload*>(ctx->m_opPayload);

    assert(payload->m_topo == output);

    const uint64_t vertexCount   = output->vertexEdgeRanges.count;
    const uint32_t triangleCount = uint32_t(output->triangleVertices.count);
    const uint64_t edgeCount     = output->edgeVertices.count;

    {
        uint32_t vertexTriangleOffset = 0;
        uint32_t vertexEdgeOffset     = 0;
        for(uint64_t i = 0; i < vertexCount; i++)
        {
            Range32* vertexTriangleRanges = arrayGet<Range32>(output->vertexTriangleRanges, i);
            Range32* vertexEdgeRanges     = arrayGet<Range32>(output->vertexEdgeRanges, i);
            vertexTriangleRanges->first   = vertexTriangleOffset;
            vertexTriangleOffset += vertexTriangleRanges->count;
            vertexEdgeRanges->first = vertexEdgeOffset;
            vertexEdgeOffset += vertexEdgeRanges->count;
            // incremented again during the filling phase
            vertexTriangleRanges->count = 0;
            vertexEdgeRanges->count     = 0;
        }

        assert(uint64_t(vertexTriangleOffset) == output->vertexTriangleConnections.count);
        assert(uint64_t(vertexEdgeOffset) == output->vertexEdgeConnections.count);

        uint32_t edgeTriangleOffset = 0;
        for(uint64_t i = 0; i < edgeCount; i++)
        {
            Range32* edgeTriangleRanges = arrayGet<Range32>(output->edgeTriangleRanges, i);
            edgeTriangleRanges->count   = payload->m_edges[i].triangleCount;
            edgeTriangleRanges->first   = edgeTriangleOffset;
            edgeTriangleOffset += edgeTriangleRanges->count;
            // incremented again during the filling phase
            edgeTriangleRanges->count = 0;
        }
        assert(uint64_t(edgeTriangleOffset) == output->edgeTriangleConnections.count);
    }

    for(uint32_t i = 0; i < triangleCount; i++)
    {
        payload->fillTriangle(i);
    }

    payload->fillEdges();

    output->isNonManifold = (payload->m_isNonManifold || payload->m_maxEdgeTriangleValence > 2) ? true : false;

    output->maxEdgeTriangleValence   = payload->m_maxEdgeTriangleValence;
    output->maxVertexEdgeValence     = payload->m_maxVertexEdgeValence;
    output->maxVertexTriangleValence = payload->m_maxVertexTriangleValence;

    ctx->resetSequence();

    return Result::eSuccess;
}

//////////////////////////////////////////////////////////////////////////

struct OpGrowTriangleSelectionPayload
{
    OpGrowTriangleSelectionPayload(const MeshTopology& topo)
        : topoUtil(topo)
    {
    }

    OpGrowTriangleSelection_input  input;
    OpGrowTriangleSelection_output output;
    MeshTopologyUtil               topoUtil;
    bool                           useSelection = false;

    std::atomic_uint32_t outputCounter = 0;

    struct Entry
    {
        Entry() = default;
        Entry(const Entry& h)
            : mask(h.mask.load())
        {
        }

        std::atomic_uint32_t mask = 0;
    };

    container::vector<Entry> selection;

    inline void select(uint32_t idx) { selection[idx / 32].mask |= (1u << (idx % 32)); }

    inline void processVertex(uint32_t vertexIndex)
    {
        uint32_t        vertexTrianglesCount;
        const uint32_t* vertexTriangles = topoUtil.getVertexTriangles(vertexIndex, vertexTrianglesCount);

        // iterate over all, ignore the fact that we will select the originating
        // triangle multiple times

        for(uint32_t t = 0; t < vertexTrianglesCount; t++)
        {
            uint32_t topoTri = topoUtil.getVertexTriangle(vertexTriangles, t);
            select(topoTri);
        }
    }
};


MICROMESH_API Result MICROMESH_CALL micromeshOpGrowTriangleSelection(OpContext                            ctx,
                                                                     const OpGrowTriangleSelection_input* input,
                                                                     OpGrowTriangleSelection_output*      output)
{
    CHECK_CTX_NONNULL(ctx);
    CHECK_NONNULL(ctx, input);
    CHECK_NONNULL(ctx, input->topology);
    CHECK_NONNULL(ctx, output);
    CHECK_CTX_BEGIN(ctx);

    if(output->triangleSelection.count != input->topology->triangleVertices.count)
    {
        LOGE(ctx, "output->triangleSelection.count != input->topology->triangleVertices.count");
        return Result::eInvalidRange;
    }

    OpGrowTriangleSelectionPayload payload(*input->topology);
    payload.input        = *input;
    payload.output       = *output;
    payload.useSelection = arrayIsValid(input->triangleSelection) && !arrayIsEmpty(input->triangleSelection);

    uint64_t triangleCount = payload.useSelection ? input->triangleSelection.count : input->triangleCount;

    uint32_t topoTriangleBitMaskCount = uint32_t((input->topology->triangleVertices.count + 31) / 32);
    payload.selection.resize(topoTriangleBitMaskCount);

    ctx->parallel_item_ranges(
        triangleCount,
        [](uint64_t idxFirst, uint64_t idxLast, uint32_t threadIndex, void* userData) {
            OpGrowTriangleSelectionPayload* payload = reinterpret_cast<OpGrowTriangleSelectionPayload*>(userData);

            for(uint64_t idx = idxFirst; idx < idxLast; idx++)
            {
                uint32_t tri = payload->useSelection ? arrayGetV<uint32_t>(payload->input.triangleSelection, idx) :
                                                       (uint32_t(idx) + payload->input.triangleFirst);
                Vector_uint32_3 triVertices = payload->topoUtil.getTriangleVertices(tri);

                // iterate connected triangles, skipping degenerates
                if(!meshIsTriangleDegenerate(triVertices))
                {
                    payload->processVertex(triVertices.x);
                    payload->processVertex(triVertices.y);
                    payload->processVertex(triVertices.z);
                }
            }
        },
        &payload);


    if(input->sortedOutput)
    {
        // serial processing gives us sorted selection
        uint32_t outputCounter = 0;
        for(uint32_t i = 0; i < topoTriangleBitMaskCount; i++)
        {
            uint32_t selectedBits = payload.selection[i].mask;
            if(!selectedBits)
                continue;

            for(uint32_t bit = 0; bit < 32; bit++)
            {
                if(selectedBits & (1u << bit))
                {
                    arraySetV<uint32_t>(output->triangleSelection, outputCounter++, i * 32 + bit);
                }
            }
        }

        output->triangleSelectionCount = outputCounter;
    }
    else
    {
        ctx->parallel_item_ranges(
            topoTriangleBitMaskCount,
            [](uint64_t idxFirst, uint64_t idxLast, uint32_t threadIndex, void* userData) {
                OpGrowTriangleSelectionPayload* payload = reinterpret_cast<OpGrowTriangleSelectionPayload*>(userData);

                for(uint64_t i = idxFirst; i < idxLast; i++)
                {
                    uint32_t selectedBits = payload->selection[i].mask;
                    if(!selectedBits)
                        continue;

                    uint32_t localCount = 0;
                    for(uint32_t bit = 0; bit < 32; bit++)
                    {
                        if(selectedBits & (1u << bit))
                        {
                            localCount++;
                        }
                    }

                    uint32_t offset = payload->outputCounter.fetch_add(localCount);

                    for(uint32_t bit = 0; bit < 32; bit++)
                    {
                        if(selectedBits & (1u << bit))
                        {
                            arraySetV<uint32_t>(payload->output.triangleSelection, offset++, uint32_t(i) * 32 + bit);
                        }
                    }
                }
            },
            &payload);

        output->triangleSelectionCount = payload.outputCounter;
    }


    return Result::eSuccess;
}

//////////////////////////////////////////////////////////////////////////

namespace
{
class DeduplicatorMap;
}
struct VertexDedup_s
{
    DeduplicatorMap* dedupMap = nullptr;
    uint32_t         index    = INVALID_INDEX;
    uint32_t         hashVal  = 0;
    uint32_t         checksum = 0;
};

namespace
{
// Thread-safe deduplication based on spatial hashing
class DeduplicatorMap
{
  public:
    // Initialize the hash map. For best results entriesCount should be
    // roughly 2x the expected number of entries
    void initialize(size_t entriesCount)
    {
        m_entries.resize(entriesCount);

        uniqueHashes.store(0);
    }

    size_t getSizeInBytes() const { return m_entries.size() * sizeof(DeduplicatorEntry); }

    uint32_t addUniqueVertex() { return uniqueHashes++; }

    uint32_t getUniqueVertexCount() const { return uniqueHashes.load(); }

    // For the vertex at index in input a hash key hashVal is computed
    // from the vertex coordinates (and potential attributes), as well as another
    // hash used as a checksum.
    // 1 - If the hash map entry at hashVal is empty the function stores
    // the vertex index at this location along with the checksum, and
    // the function returns the value of index.
    // 2 - If another vertex has the same hash key and checksum, the function return
    // the index stored at hashVal.
    // 3 - If another vertex has the same hash key but a different checksum the function
    // mitigates the collision be generating another value for hashVal by rehashing, and returns to 1.
    //
    // input: mesh that is being generated
    // index: index of the vertex to insert/test in the hash map
    // ignoreAttributes: if true only the vertex position is considered. Otherwise, attributes such as
    // normals and texture coordinates are also taken into account in the hashing process
    //
    // Note: this function only merges vertices with the *exact* same floating-point coordinates (and other attributes).
    // Those values are obtained through interpolation of the attributes at the base triangle vertices. Since neighboring
    // triangles on the base mesh may be enumerated in a different order, an order-independent interpolation function
    // such as getInterpolatedSorted must be used to ensure watertightness of the resulting mesh.
    uint32_t deduplicate(const VertexDedup_s& dedupState)
    {
        assert(dedupState.dedupMap == this);

        bool     useInputIndex = dedupState.index != INVALID_INDEX;
        uint32_t actualIndex   = 0;

        bool found = false;

        // Hash index for the vertex, used as a location in the hash map
        uint32_t hashVal = dedupState.hashVal % uint32_t(m_entries.size());
        // Checksum for collision detection
        // Prevent checksum==0 since this is the special value to indicate the entry is free
        uint32_t checksum = std::max(1u, dedupState.checksum);

        while(!found)
        {
            uint32_t currentChecksum = 0u;
            // Atomically replace the checksum value at hashVal if its value is 0
            if(m_entries[hashVal].checksum.compare_exchange_strong(currentChecksum, checksum))
            {
                // If the replacement succeeded, finalize the entry by storing the index of the vertex
                if(useInputIndex)
                {
                    actualIndex = dedupState.index;
                }
                else
                {
                    actualIndex = uniqueHashes++;
                }
                m_entries[hashVal].referenceVertex.exchange(actualIndex);
                found = true;
            }
            else
            {
                // If a vertex with same entry is already present, fetch the vertex index stored in the hash map
                if(currentChecksum == checksum)
                {
                    // Spinlock to ensure the entry's reference vertex had time to be written
                    while(m_entries[hashVal].referenceVertex.load() == INVALID_INDEX)
                        ;
                    // Fetch the vertex index from the hash map
                    actualIndex = m_entries[hashVal].referenceVertex.load();
                    found       = true;
                }
                else
                {
                    // A vertex is present at hashVal, but with a different checksum. This indicates a collision,
                    // which is mitigated by rehashing
                    hashVal = (hashVal + 1) % uint32_t(m_entries.size());
                }
            }
        }

        return actualIndex;
    }

    // Wang hash for the computation of the hash indices
    static inline uint32_t wangHash(uint32_t n)
    {
        n = (n ^ 61) ^ (n >> 16);
        n *= 9;
        n = n ^ (n >> 4);
        n *= 0x27d4eb2d;
        n = n ^ (n >> 15);
        return n;
    }
    // XOR-shift hash for the computation of the checksums
    static inline uint32_t xorshift32(uint32_t n)
    {
        n ^= n << 13;
        n ^= n >> 7;
        n ^= n << 17;
        return n;
    }

    // A name for a uint32_t void(uint32_t v) function, abstracting e.g.
    // wangHash and xorshift32
    typedef uint32_t (*HashFunction)(uint32_t);

    // Spatial hashing of vector types with componentCount components,
    // using the integer hashing function H. This function successively hashes
    // the components of the vector to obtain a single integer hash index.
    template <HashFunction H, uint32_t componentCount>
    static inline uint32_t hashVec(uint32_t previousHash, const uint32_t* vec)
    {
        uint32_t res = previousHash;
        for(uint32_t i = 0; i < componentCount; i++)
            res = H(res + vec[i]);
        return res;
    }

  private:
    // Hash map entry, containing a checksum for collision detection and
    // the index of the vertex corresponding to that entry.
    struct DeduplicatorEntry
    {
        DeduplicatorEntry() = default;
        DeduplicatorEntry(const DeduplicatorEntry& h)
            : checksum(h.checksum.load())
            , referenceVertex(h.referenceVertex.load())
        {
        }

        std::atomic_uint32_t checksum{0};
        std::atomic_uint32_t referenceVertex{INVALID_INDEX};
    };

    // Hash map storage
    container::vector<DeduplicatorEntry> m_entries;

    // Usage statistics
    std::atomic_uint32_t uniqueHashes{0};
};

VertexDedup_s makeVertexDedup(DeduplicatorMap* dedup, const OpBuildMeshTopologyIndices_input* input, uint32_t index)
{
    VertexDedup_s dedupState;
    dedupState.dedupMap = dedup;
    dedupState.index    = index;

    const uint32_t* pos = arrayGet<uint32_t>((const ArrayInfo&)input->meshVertexPositions, index);

    dedupState.hashVal  = DeduplicatorMap::hashVec<DeduplicatorMap::wangHash, 3>(dedupState.hashVal, pos);
    dedupState.checksum = DeduplicatorMap::hashVec<DeduplicatorMap::xorshift32, 3>(dedupState.checksum, pos);

    if(!arrayIsEmpty((const ArrayInfo&)input->meshVertexDirections))
    {
        const uint32_t* dir = arrayGet<uint32_t>((const ArrayInfo&)input->meshVertexDirections, index);
        dedupState.hashVal  = DeduplicatorMap::hashVec<DeduplicatorMap::wangHash, 3>(dedupState.hashVal, dir);
        dedupState.checksum = DeduplicatorMap::hashVec<DeduplicatorMap::xorshift32, 3>(dedupState.checksum, dir);
    }

    return dedupState;
}
}  // namespace

MICROMESH_API void MICROMESH_CALL micromeshVertexDedupAppendAttribute(VertexDedup dedupState, uint32_t dataSize, const void* data)
{
    union
    {
        uint32_t tailU32;
        uint8_t  tailU8[4];
    };

    uint32_t numTail = dataSize % 4;
    uint32_t num32   = dataSize / 4;

    const uint32_t* dataU32 = reinterpret_cast<const uint32_t*>(data);
    const uint8_t*  dataU8  = reinterpret_cast<const uint8_t*>(data);

    if(numTail)
    {
        tailU32 = 0;
        switch(numTail)
        {
        case 3:
            tailU8[2] = dataU8[num32 * 4 + 2];
            [[fallthrough]];
        case 2:
            tailU8[1] = dataU8[num32 * 4 + 1];
            [[fallthrough]];
        case 1:
            tailU8[0] = dataU8[num32 * 4 + 0];
        }
    }

    for(uint32_t i = 0; i < num32; i++)
        dedupState->hashVal = DeduplicatorMap::wangHash(dedupState->hashVal + dataU32[i]);
    if(numTail)
        dedupState->hashVal = DeduplicatorMap::wangHash(dedupState->hashVal + tailU32);

    for(uint32_t i = 0; i < num32; i++)
        dedupState->checksum = DeduplicatorMap::xorshift32(dedupState->checksum + dataU32[i]);
    if(numTail)
        dedupState->checksum = DeduplicatorMap::xorshift32(dedupState->checksum + tailU32);
}

MICROMESH_API uint32_t MICROMESH_CALL micromeshVertexDedupGetIndex(VertexDedup dedupState)
{
    assert(dedupState->dedupMap);

    // never called hash
    if(dedupState->hashVal == 0 && dedupState->checksum == 0)
        return dedupState->dedupMap->addUniqueVertex();

    return dedupState->dedupMap->deduplicate(*dedupState);
}

MICROMESH_API Result MICROMESH_CALL micromeshOpBuildMeshTopologyIndices(OpContext                               ctx,
                                                                        const OpBuildMeshTopologyIndices_input* input,
                                                                        OpBuildMeshTopologyIndices_output*      output)
{
    CHECK_CTX_NONNULL(ctx);
    CHECK_NONNULL(ctx, input);
    CHECK_NONNULL(ctx, output);
    CHECK_CTX_BEGIN(ctx);

    if(input->meshVertexPositions.count > MAX_UINT32_COUNT || input->meshTriangleVertices.count > MAX_UINT32_COUNT)
    {
        LOGE(ctx, "vertexCount and triangleCount must be within uint32_t limits");
        return Result::eInvalidRange;
    }

    uint32_t vertexCount   = uint32_t(input->meshVertexPositions.count);
    uint32_t triangleCount = uint32_t(input->meshTriangleVertices.count);

    DeduplicatorMap dedup;
    dedup.initialize(vertexCount * 2);

    ctx->parallel_item_ranges(triangleCount, [&](uint64_t idxFirst, uint64_t idxLast, uint32_t threadIndex, void* userData) {
        for(uint64_t idx = idxFirst; idx < idxLast; idx++)
        {
            const Vector_uint32_3* triVertices = arrayGet<Vector_uint32_3>((const ArrayInfo&)input->meshTriangleVertices, idx);
            Vector_uint32_3* triDedupVertices = arrayGet<Vector_uint32_3>((ArrayInfo&)output->meshTopologyTriangleVertices, idx);
            // not sure if doing this in parallel is actually fast
            VertexDedup_s dedupState0 = makeVertexDedup(&dedup, input, triVertices->x);
            VertexDedup_s dedupState1 = makeVertexDedup(&dedup, input, triVertices->y);
            VertexDedup_s dedupState2 = makeVertexDedup(&dedup, input, triVertices->z);

            triDedupVertices->x = dedup.deduplicate(dedupState0);
            triDedupVertices->y = dedup.deduplicate(dedupState1);
            triDedupVertices->z = dedup.deduplicate(dedupState2);
        }
    });

    return Result::eSuccess;
}

// output must have all information set, only values be written
MICROMESH_API Result MICROMESH_CALL micromeshOpSampleFromMesh(OpContext ctx, const OpSampleFromMesh_input* input, Micromap* output)
{
    CHECK_CTX_NONNULL(ctx);
    CHECK_NONNULL(ctx, input);
    CHECK_NONNULL(ctx, output);
    CHECK_CTX_BEGIN(ctx);

    bool hasTexCoords = arrayTypedIsValid(input->meshVertexTexcoords) && input->meshVertexTexcoords.count
                        && arrayTypedIsValid(input->meshTriangleVertices) && input->meshTriangleVertices.count;

    ctx->parallel_items(input->meshTriangleCount, [&](uint64_t meshTri, uint32_t threadIndex, void* userData) {
        uint32_t mapTri = meshGetTriangleMapping(input->meshTriangleMappings, meshTri);

        uint32_t subdivLevel = arrayTypedGetV(output->triangleSubdivLevels, mapTri);

        uint32_t numSegmentsPerEdge = subdivLevelGetSegmentCount(subdivLevel);
        uint32_t numVtxPerEdge      = subdivLevelGetSegmentCount(subdivLevel) + 1;

        float baryRcp = 1.0f / float(numSegmentsPerEdge);

        Vector_float_2 tex0 = {0, 0};
        Vector_float_2 tex1 = {0, 0};
        Vector_float_2 tex2 = {0, 0};

        if(hasTexCoords)
        {
            Vector_uint32_3 triVertices = arrayTypedGetV(input->meshTriangleVertices, meshTri);
            tex0                        = arrayTypedGetV(input->meshVertexTexcoords, triVertices.x);
            tex1                        = arrayTypedGetV(input->meshVertexTexcoords, triVertices.y);
            tex2                        = arrayTypedGetV(input->meshVertexTexcoords, triVertices.z);
        }

        void* beginResult = nullptr;
        if(input->pfnBeginTriangle)
        {
            beginResult = input->pfnBeginTriangle(uint32_t(meshTri), mapTri, threadIndex, input->userData);
        }

        for(uint32_t u = 0; u < numVtxPerEdge; u++)
        {
            for(uint32_t v = 0; v < numVtxPerEdge - u; v++)
            {
                uint32_t valueIdx  = output->layout.pfnGetMicroVertexIndex(u, v, subdivLevel, output->layout.userData);
                void*    valueData = micromapGetTriangleValue<void>(*output, mapTri, valueIdx);

                VertexSampleInfo sample;
                sample.meshTriangleIndex     = uint32_t(meshTri);
                sample.micromapTriangleIndex = mapTri;

                sample.subdivLevel = subdivLevel;
                sample.vertexUV.u  = uint16_t(u);
                sample.vertexUV.v  = uint16_t(v);

                uint32_t w              = (1u << subdivLevel) - u - v;
                sample.vertexWUVfloat.u = float(u) * baryRcp;
                sample.vertexWUVfloat.v = float(v) * baryRcp;
                sample.vertexWUVfloat.w = float(w) * baryRcp;

                sample.vertexTexCoord =
                    (tex0 * sample.vertexWUVfloat.w) + (tex1 * sample.vertexWUVfloat.u) + (tex2 * sample.vertexWUVfloat.v);

                input->pfnSampleVertex(&sample, valueIdx, valueData, threadIndex, beginResult, input->userData);
            }
        }

        if(input->pfnEndTriangle)
        {
            input->pfnEndTriangle(uint32_t(meshTri), mapTri, threadIndex, beginResult, input->userData);
        }
    });

    return Result::eSuccess;
}

MICROMESH_API Result MICROMESH_CALL micromeshOpSmoothMeshDirections(OpContext                           ctx,
                                                                    const OpSmoothMeshDirections_input* input,
                                                                    OpSmoothMeshDirections_output*      output)
{
    CHECK_CTX_NONNULL(ctx);
    CHECK_NONNULL(ctx, input);
    CHECK_NONNULL(ctx, output);
    CHECK_CTX_BEGIN(ctx);

    float areaWeight = input->triangleAreaWeight;

    Vector_float_3 zero = {0, 0, 0};
    arrayFill(output->meshVertexDirections, zero);

    ctx->parallel_item_ranges(input->meshTriangleVertices.count,
                              [&](uint64_t idxFirst, uint64_t idxLast, uint32_t threadIndex, void* userData) {
                                  for(uint64_t idx = idxFirst; idx < idxLast; idx++)
                                  {
                                      Vector_uint32_3 triVertices = arrayTypedGetV(input->meshTriangleVertices, idx);

                                      Vector_float_3 v0 = arrayTypedGetV(input->meshVertexPositions, triVertices.x);
                                      Vector_float_3 v1 = arrayTypedGetV(input->meshVertexPositions, triVertices.y);
                                      Vector_float_3 v2 = arrayTypedGetV(input->meshVertexPositions, triVertices.z);

                                      Vector_float_3 d0 = arrayTypedGetV(output->meshVertexDirections, triVertices.x);
                                      Vector_float_3 d1 = arrayTypedGetV(output->meshVertexDirections, triVertices.y);
                                      Vector_float_3 d2 = arrayTypedGetV(output->meshVertexDirections, triVertices.z);

                                      Vector_float_3 e0 = v1 - v0;
                                      Vector_float_3 e1 = v2 - v0;

                                      Vector_float_3 nrm = math::cross(e0, e1);
                                      float          len = math::length(nrm);
                                      len                = (0.5f * areaWeight) + (1.0f - areaWeight) * len;
                                      nrm                = nrm / len;

                                      arrayTypedSetV(output->meshVertexDirections, triVertices.x, d0 + nrm);
                                      arrayTypedSetV(output->meshVertexDirections, triVertices.y, d1 + nrm);
                                      arrayTypedSetV(output->meshVertexDirections, triVertices.z, d2 + nrm);
                                  }
                              });

    ctx->parallel_item_ranges(output->meshVertexDirections.count,
                              [&](uint64_t idxFirst, uint64_t idxLast, uint32_t threadIndex, void* userData) {
                                  for(uint64_t idx = idxFirst; idx < idxLast; idx++)
                                  {
                                      arrayTypedSetV(output->meshVertexDirections, idx,
                                                     math::normalize(arrayTypedGetV(output->meshVertexDirections, idx)));
                                  }
                              });

    return Result::eSuccess;
}

struct OpTessellateMeshPayload
{
    struct TriangleOutput
    {
        uint32_t vertexOffset;
        uint32_t triangleOffset;
    };

    struct ThreadConfig
    {
        uint32_t subdivLevel   = ~0;
        uint32_t edgeFlag      = ~0;
        uint32_t vertexCount   = 0;
        uint32_t triangleCount = 0;

        ArrayInfo_uint32_3 trianglesInfo;
        ArrayInfo_uint16_2 verticesInfo;
        Vector_uint32_3*   triangles     = nullptr;
        BaryUV_uint16*     vertices      = nullptr;
        uint32_t*          vertexIndices = nullptr;
    };
    MicromapLayout layout;

    bool                              useEdgeFlags = false;
    bool                              useMappings  = false;
    container::vector<TriangleOutput> triangles;

    const OpTessellateMesh_input* input  = nullptr;
    OpTessellateMesh_output*      output = nullptr;

    container::vector<BaryUV_uint16>   tempVertices;
    container::vector<uint32_t>        tempVertexIndices;
    container::vector<Vector_uint32_3> tempTriangles;
    container::vector<ThreadConfig>    configs;

    DeduplicatorMap dedupMap;

    static void deleter(void* data) { delete reinterpret_cast<OpTessellateMeshPayload*>(data); }
};

MICROMESH_API Result MICROMESH_CALL micromeshOpTessellateMeshBegin(OpContext                     ctx,
                                                                   const OpTessellateMesh_input* input,
                                                                   OpTessellateMesh_output*      output)
{
    CHECK_CTX_NONNULL(ctx);
    CHECK_NONNULL(ctx, input);
    CHECK_NONNULL(ctx, output);
    CHECK_CTX_BEGIN(ctx);

    // compute accurate triangle index
    // compute worst-case vertex count

    bool useMappings = arrayIsValid(input->meshTriangleMappings) && !arrayIsEmpty(input->meshTriangleMappings)
                       && arrayIsValid(input->micromapTriangleSubdivLevels)
                       && !arrayIsEmpty(input->micromapTriangleSubdivLevels);

    bool useEdgeFlags = arrayIsValid(input->meshTrianglePrimitiveFlags) && !arrayIsEmpty(input->meshTrianglePrimitiveFlags);

    uint64_t triangleCount = useMappings ? input->meshTriangleMappings.count : input->meshTriangleSubdivLevels.count;

    uint64_t outTriangleCount = 0;
    uint64_t outVertexCount   = 0;

    std::unique_ptr<OpTessellateMeshPayload> payload = std::make_unique<OpTessellateMeshPayload>();
    payload->triangles.resize(triangleCount + 1);
    payload->output = output;
    payload->input  = input;

    payload->useMappings  = useMappings;
    payload->useEdgeFlags = useEdgeFlags;
    micromeshLayoutInitStandard(&payload->layout, StandardLayoutType::eUmajor);

    payload->tempVertices.resize(subdivLevelGetVertexCount(input->maxSubdivLevel) * ctx->getThreadCount());
    payload->tempVertexIndices.resize(subdivLevelGetVertexCount(input->maxSubdivLevel) * ctx->getThreadCount());
    payload->tempTriangles.resize(subdivLevelGetTriangleCount(input->maxSubdivLevel) * ctx->getThreadCount());
    payload->configs.resize(ctx->getThreadCount());

    for(uint32_t i = 0; i < ctx->getThreadCount(); i++)
    {
        payload->configs[i].verticesInfo.data = &payload->tempVertices[subdivLevelGetVertexCount(input->maxSubdivLevel) * i];
        payload->configs[i].trianglesInfo.data = &payload->tempTriangles[subdivLevelGetTriangleCount(input->maxSubdivLevel) * i];
        payload->configs[i].vertices  = &payload->tempVertices[subdivLevelGetVertexCount(input->maxSubdivLevel) * i];
        payload->configs[i].triangles = &payload->tempTriangles[subdivLevelGetTriangleCount(input->maxSubdivLevel) * i];
        payload->configs[i].vertexIndices = &payload->tempVertexIndices[subdivLevelGetVertexCount(input->maxSubdivLevel) * i];
    }

    for(uint32_t meshTri = 0; meshTri < uint32_t(triangleCount); meshTri++)
    {
        uint32_t mapTri      = useMappings ? arrayGetV<uint32_t>(input->meshTriangleMappings, meshTri) : meshTri;
        uint16_t subdivLevel = useMappings ? arrayGetV<uint16_t>(input->micromapTriangleSubdivLevels, mapTri) :
                                             arrayGetV<uint16_t>(input->meshTriangleSubdivLevels, meshTri);
        uint16_t edgeFlag = useEdgeFlags ? arrayGetV<uint8_t>(input->meshTrianglePrimitiveFlags, meshTri) : uint8_t(0);

        OpTessellateMeshPayload::TriangleOutput& triOutput = payload->triangles[meshTri];
        triOutput.vertexOffset                             = uint32_t(outVertexCount);
        triOutput.triangleOffset                           = uint32_t(outTriangleCount);

        outTriangleCount += subdivLevelGetTriangleCount(subdivLevel, edgeFlag);
        outVertexCount += subdivLevelGetVertexCount(subdivLevel, edgeFlag);
    }

    if(outVertexCount > std::numeric_limits<uint32_t>::max())
    {
        LOGE(ctx, "Tessellating this mesh would produce %zu vertices, which is greater than 2^32-1.", outVertexCount);
        return Result::eInvalidValue;
    }
    if(outTriangleCount > std::numeric_limits<uint32_t>::max())
    {
        LOGE(ctx, "Tessellating this mesh would produce %zu triangles, which is greater than 2^32-1.", outTriangleCount);
        return Result::eInvalidValue;
    }

    {
        OpTessellateMeshPayload::TriangleOutput& triOutput = payload->triangles[triangleCount];
        triOutput.vertexOffset                             = uint32_t(outVertexCount);
        triOutput.triangleOffset                           = uint32_t(outTriangleCount);
    }

    output->vertexCount                = uint32_t(outVertexCount);
    output->meshTriangleVertices.count = uint32_t(outTriangleCount);

    if(outVertexCount > 0xFFFFFFFFull || triangleCount > 0xFFFFFFFFull)
    {
        LOGE(ctx, "output vertex or triangle out of uint32 bounds");
        return Result::eInvalidRange;
    }

    ctx->setNextSequenceFn(&micromeshOpTessellateMeshEnd);
    ctx->setPayload(payload.release(), OpTessellateMeshPayload::deleter);

    return Result::eSuccess;
}


MICROMESH_API Result MICROMESH_CALL micromeshOpTessellateMeshEnd(OpContext                     ctx,
                                                                 const OpTessellateMesh_input* input,
                                                                 OpTessellateMesh_output*      output)
{
    CHECK_CTX_NONNULL(ctx);
    CHECK_NONNULL(ctx, input);
    CHECK_NONNULL(ctx, output);
    CHECK_CTX_END(ctx, &micromeshOpTessellateMeshEnd);

    OpTessellateMeshPayload* payload = reinterpret_cast<OpTessellateMeshPayload*>(ctx->m_opPayload);
    assert(payload->input == input);
    assert(payload->output == output);

    if(payload->input->useVertexDeduplication)
    {
        payload->dedupMap.initialize(payload->output->vertexCount * 2);
    }

    ctx->parallel_items(
        payload->triangles.size() - 1,
        [](uint64_t itemIndex, uint32_t threadIndex, void* userData) {
            OpTessellateMeshPayload*               payload = reinterpret_cast<OpTessellateMeshPayload*>(userData);
            OpTessellateMeshPayload::ThreadConfig& config  = payload->configs[threadIndex];
            OpTessellateMesh_output*               output  = payload->output;
            const OpTessellateMesh_input*          input   = payload->input;


            bool useMappings  = payload->useMappings;
            bool useEdgeFlags = payload->useEdgeFlags;
            bool useDedup     = input->useVertexDeduplication;

            uint32_t meshTri     = uint32_t(itemIndex);
            uint32_t mapTri      = useMappings ? arrayGetV<uint32_t>(input->meshTriangleMappings, meshTri) : meshTri;
            uint16_t subdivLevel = useMappings ? arrayGetV<uint16_t>(input->micromapTriangleSubdivLevels, mapTri) :
                                                 arrayGetV<uint16_t>(input->meshTriangleSubdivLevels, meshTri);
            uint16_t edgeFlag = useEdgeFlags ? arrayGetV<uint8_t>(input->meshTrianglePrimitiveFlags, meshTri) : uint8_t(0);

            const OpTessellateMeshPayload::TriangleOutput& triOutput = payload->triangles[meshTri];

            void* beginResult = nullptr;
            if(input->pfnBeginTriangle)
            {
                beginResult = input->pfnBeginTriangle(meshTri, mapTri, threadIndex, input->userData);
            }

            if(config.edgeFlag != edgeFlag || config.subdivLevel != subdivLevel)
            {
                // recalculate vertices and triangles

                config.edgeFlag    = edgeFlag;
                config.subdivLevel = subdivLevel;

                uint32_t triangleCount = subdivLevelGetTriangleCount(subdivLevel);
                uint32_t vertexCount   = subdivLevelGetVertexCount(subdivLevel);
                // generate mesh
                config.trianglesInfo.count = triangleCount;
                config.verticesInfo.count  = vertexCount;

                micromeshLayoutBuildUVMesh(&payload->layout, &config.verticesInfo, &config.trianglesInfo, subdivLevel, edgeFlag);

                if(edgeFlag)
                {
                    // mark used vertices
                    memset(config.vertexIndices, 0, sizeof(uint32_t) * vertexCount);

                    for(uint32_t i = 0; i < triangleCount; i++)
                    {
                        Vector_uint32_3 tri = config.triangles[i];
                        if(!meshIsTriangleDegenerate(tri))
                        {
                            config.vertexIndices[tri.x] = 1;
                            config.vertexIndices[tri.y] = 1;
                            config.vertexIndices[tri.z] = 1;
                        }
                    }
                    config.vertexCount = 0;
                    // compact vertices
                    for(uint32_t i = 0; i < vertexCount; i++)
                    {
                        BaryUV_uint16 uv = config.vertices[i];
                        if(config.vertexIndices[i])
                        {
                            uint32_t compactIndex         = config.vertexCount++;
                            config.vertexIndices[i]       = compactIndex;
                            config.vertices[compactIndex] = uv;
                        }
                    }
                    config.triangleCount = 0;
                    // compacted triangles with compacted indices
                    for(uint32_t i = 0; i < triangleCount; i++)
                    {
                        Vector_uint32_3 tri = config.triangles[i];
                        if(!meshIsTriangleDegenerate(tri))
                        {
                            // re-index
                            tri.x = config.vertexIndices[tri.x];
                            tri.y = config.vertexIndices[tri.y];
                            tri.z = config.vertexIndices[tri.z];

                            uint32_t compactIndex          = config.triangleCount++;
                            config.triangles[compactIndex] = tri;
                        }
                    }
                }
                else
                {
                    config.triangleCount = triangleCount;
                    config.vertexCount   = vertexCount;
                }
            }

            assert(config.vertexCount == (payload->triangles[meshTri + 1].vertexOffset - triOutput.vertexOffset));
            assert(config.triangleCount == (payload->triangles[meshTri + 1].triangleOffset - triOutput.triangleOffset));

            // generate vertices
            for(uint32_t i = 0; i < config.vertexCount; i++)
            {
                VertexDedup_s dedupState;

                VertexGenerateInfo vertex;
                vertex.subdivLevel           = subdivLevel;
                vertex.vertexUV              = config.vertices[i];
                vertex.vertexWUVfloat        = baryUVtoWUV_float(vertex.vertexUV, subdivLevel);
                vertex.meshTriangleIndex     = meshTri;
                vertex.micromapTriangleIndex = mapTri;

                if(useDedup)
                {
                    BaryWUV_uint16 coord = baryUVtoWUV_uint(vertex.vertexUV, subdivLevel);
                    // only trigger dedup for on-edge vertices
                    // otherwise provide unique vertex
                    bool isOnEdge        = baryWUVisOnEdge(coord);
                    vertex.nonDedupIndex = !isOnEdge ? payload->dedupMap.addUniqueVertex() : 0;
                    dedupState.dedupMap  = isOnEdge ? &payload->dedupMap : nullptr;
                }
                else
                {
                    vertex.nonDedupIndex = triOutput.vertexOffset + i;
                }

                // let user generate the vertex and give us the vertex index
                config.vertexIndices[i] = input->pfnGenerateVertex(&vertex, dedupState.dedupMap ? &dedupState : nullptr,
                                                                   threadIndex, beginResult, input->userData);
            }

            // generate triangles
            for(uint32_t i = 0; i < config.triangleCount; i++)
            {
                Vector_uint32_3 tri = config.triangles[i];
                // re-index
                tri.x = config.vertexIndices[tri.x];
                tri.y = config.vertexIndices[tri.y];
                tri.z = config.vertexIndices[tri.z];
                arraySetV<Vector_uint32_3>(output->meshTriangleVertices, triOutput.triangleOffset + i, tri);
            }

            if(input->pfnEndTriangle)
            {
                input->pfnEndTriangle(meshTri, mapTri, threadIndex, beginResult, input->userData);
            }
        },
        ctx->m_opPayload);

    if(payload->input->useVertexDeduplication)
    {
        payload->output->vertexCount = payload->dedupMap.getUniqueVertexCount();
    }

    ctx->resetSequence();

    return Result::eSuccess;
}

struct OpAdaptiveSubdivisionPayload
{
    OpAdaptiveSubdivision_input  input;
    OpAdaptiveSubdivision_output output;
    bool                         useArea = false;

    std::atomic_uint32_t maxSizeUI = 0;
    float                maxSize   = 0;

    container::vector<float>    triangleSizes;
    container::vector<uint32_t> threadMinLevels;
    container::vector<uint32_t> threadMaxLevels;
};

template <bool useUV, bool useRelative>
void processAdapativeSubdivision(uint64_t idxFirst, uint64_t idxLast, uint32_t threadIndex, void* userData)
{
    OpAdaptiveSubdivisionPayload*      payload         = reinterpret_cast<OpAdaptiveSubdivisionPayload*>(userData);
    uint32_t*                          threadMinLevels = payload->threadMinLevels.data();
    uint32_t*                          threadMaxLevels = payload->threadMaxLevels.data();
    const OpAdaptiveSubdivision_input& input           = payload->input;
    OpAdaptiveSubdivision_output&      output          = payload->output;
    bool                               useArea         = payload->useArea;

    for(uint64_t idx = idxFirst; idx < idxLast; idx++)
    {
        Vector_uint32_3 tri = arrayGetV<Vector_uint32_3>(input.meshTriangleVertices, idx);

        Vector_float_3 va;
        Vector_float_3 vb;
        Vector_float_3 vc;

        if(useUV)
        {
            va = makeVector_float_3(arrayGetV<Vector_float_2>(input.meshVertexTexcoords, tri.x) * input.texResolution);
            vb = makeVector_float_3(arrayGetV<Vector_float_2>(input.meshVertexTexcoords, tri.y) * input.texResolution);
            vc = makeVector_float_3(arrayGetV<Vector_float_2>(input.meshVertexTexcoords, tri.z) * input.texResolution);
        }
        else
        {
            va = arrayGetV<Vector_float_3>(input.meshVertexPositions, tri.x) * input.positionScale;
            vb = arrayGetV<Vector_float_3>(input.meshVertexPositions, tri.y) * input.positionScale;
            vc = arrayGetV<Vector_float_3>(input.meshVertexPositions, tri.z) * input.positionScale;
        }

        union
        {
            float    size;
            uint32_t sizeU32;
        };
        if(useArea)
        {
            // take square root of uv area to get subdivlevel
            // so that number of triangles ~= number of texels
            size = sqrtf(math::length(math::cross(vb - va, vc - va)) * 0.5f);
        }
        else
        {
            size = std::max(std::max(math::length(va - vb), math::length(va - vc)), math::length(vb - vc));
        }

        if(useRelative)
        {
            uint32_t prevValue = payload->maxSizeUI;
            while(prevValue < sizeU32 && !payload->maxSizeUI.compare_exchange_weak(prevValue, sizeU32))
            {
            }

            if(payload->triangleSizes.size())
            {
                payload->triangleSizes[idx] = size;
            }
        }
        else
        {
            // Calculate the subdiv level that would produce the desired edge tessellation
            size                 = std::ceil(std::log2(std::max(1.0f, size)));
            size                 = std::max(0.0f, size + input.subdivLevelBias);
            uint16_t subdivLevel = std::min(uint16_t(input.maxSubdivLevel), uint16_t(size));
            arraySetV<uint16_t>(output.meshTriangleSubdivLevels, idx, subdivLevel);

            threadMinLevels[threadIndex] = std::min(threadMinLevels[threadIndex], uint32_t(subdivLevel));
            threadMaxLevels[threadIndex] = std::max(threadMaxLevels[threadIndex], uint32_t(subdivLevel));
        }
    }
}

MICROMESH_API Result MICROMESH_CALL micromeshOpAdaptiveSubdivision(OpContext                          ctx,
                                                                   const OpAdaptiveSubdivision_input* input,
                                                                   OpAdaptiveSubdivision_output*      output)
{
    CHECK_CTX_NONNULL(ctx);
    CHECK_NONNULL(ctx, input);
    CHECK_NONNULL(ctx, output);
    CHECK_CTX_BEGIN(ctx);

    bool useUV        = arrayIsValid(input->meshVertexTexcoords) && !arrayIsEmpty(input->meshVertexTexcoords);
    bool hasPositions = arrayIsValid(input->meshVertexPositions) && !arrayIsEmpty(input->meshVertexPositions);
    bool useArea      = input->useArea;
    bool useRelative  = input->useRelativeValues || input->onlyComputeRelativeMaxValue;

    if(!useUV && !hasPositions)
    {
        LOGE(ctx, "Generating subdivision levels requires vertex positions or texture coordinates. None given.");
        return Result::eInvalidValue;
    }

    OpAdaptiveSubdivisionPayload payload;
    payload.input   = *input;
    payload.output  = *output;
    payload.useArea = useArea;
    payload.maxSize = 0;
    payload.threadMinLevels.resize(ctx->getThreadCount(), ~0);
    payload.threadMaxLevels.resize(ctx->getThreadCount(), 0);

    if(useRelative && !input->onlyComputeRelativeMaxValue)
    {
        payload.triangleSizes.resize(input->meshTriangleVertices.count);
    }

    if(useUV && !useRelative)
    {
        ctx->parallel_item_ranges(input->meshTriangleVertices.count, processAdapativeSubdivision<true, false>, &payload);
    }
    else if(!useUV && !useRelative)
    {
        ctx->parallel_item_ranges(input->meshTriangleVertices.count, processAdapativeSubdivision<false, false>, &payload);
    }
    else if(useUV && useRelative)
    {
        ctx->parallel_item_ranges(input->meshTriangleVertices.count, processAdapativeSubdivision<true, true>, &payload);
    }
    else
    {
        ctx->parallel_item_ranges(input->meshTriangleVertices.count, processAdapativeSubdivision<false, true>, &payload);
    }

    output->relativeMaxValue = 0;
    if(useRelative)
    {
        uint32_t sizeUI          = payload.maxSizeUI;
        output->relativeMaxValue = *reinterpret_cast<float*>(&sizeUI);
    }

    if(payload.triangleSizes.size())
    {
        assert(useRelative);

        payload.maxSize = input->useRelativeMaxValueOverride ? input->relativeMaxValueOverride : output->relativeMaxValue;
        payload.maxSize *= input->relativeWeight;

        // seed initial subdivision target
        ctx->parallel_item_ranges(
            input->meshTriangleVertices.count,
            [](uint64_t idxFirst, uint64_t idxLast, uint32_t threadIndex, void* userData) {
                OpAdaptiveSubdivisionPayload*      payload = reinterpret_cast<OpAdaptiveSubdivisionPayload*>(userData);
                uint32_t*                          threadMinLevels = payload->threadMinLevels.data();
                uint32_t*                          threadMaxLevels = payload->threadMaxLevels.data();
                const OpAdaptiveSubdivision_input& input           = payload->input;
                OpAdaptiveSubdivision_output&      output          = payload->output;
                float                              maxSize         = payload->maxSize;
                uint32_t                           maxSubdivLevel  = input.maxSubdivLevel;

                for(uint64_t idx = idxFirst; idx < idxLast; idx++)
                {
                    // fraction relative to reference  1, 2, 3, 4, etc
                    float    fraction       = std::max(maxSize / payload->triangleSizes[idx], 1.0f);
                    uint32_t subdivFraction = (uint32_t)(std::log2f(fraction));
                    uint16_t subdivLevel    = uint16_t(maxSubdivLevel - std::min(maxSubdivLevel, subdivFraction));
                    arraySetV<uint16_t>(output.meshTriangleSubdivLevels, idx, subdivLevel);

                    threadMinLevels[threadIndex] = std::min(threadMinLevels[threadIndex], uint32_t(subdivLevel));
                    threadMaxLevels[threadIndex] = std::max(threadMaxLevels[threadIndex], uint32_t(subdivLevel));
                }
            },
            &payload);
    }

    if(!input->onlyComputeRelativeMaxValue)
    {
        output->minSubdivLevel = ~0;
        output->maxSubdivLevel = 0;

        for(uint32_t i = 0; i < ctx->getThreadCount(); i++)
        {
            output->minSubdivLevel = std::min(output->minSubdivLevel, payload.threadMinLevels[i]);
            output->maxSubdivLevel = std::max(output->maxSubdivLevel, payload.threadMaxLevels[i]);
        }
    }

    return Result::eSuccess;
}

}  // namespace micromesh
