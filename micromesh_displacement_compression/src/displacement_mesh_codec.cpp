/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <cinttypes>
#include <mutex>

#include "micromesh_internal/micromesh_context.h"
#include "displacement_block_codec.h"
#include "displacement_configs.h"
#include "displacement_mesh_codec.h"

namespace micromesh
{
namespace dispenc
{
///////////////////////////////////////////////////////////////////////////////
// General utilities

float computePSNRGeneral(double l2Err, double numElems)
{
    if(l2Err == 0.0 || numElems == 0.0)
        return INFINITY;

    const double maxValue = (double)(1u << config_decoderWordSizeInBits);
    const double MSE      = l2Err / numElems;
    const double PSNR     = 10.0 * std::log10(maxValue * maxValue / MSE);
    return (float)PSNR;
}

template <class T>
T roundUpToMultipleOf128(T v)
{
    return (v + (128 - 1)) & ~(128 - 1);
}

///////////////////////////////////////////////////////////////////////////////
// Triangle

MeshEncoder::Triangle::Triangle(const Micromap& micromap, uint64_t micromapTriangleIdx, std::array<float, 3> vtxImportance)
{
    m_baseSubdivLevel = static_cast<uint8_t>(arrayGetV<uint16_t>(micromap.triangleSubdivLevels, micromapTriangleIdx));
    m_subdivLevel     = m_baseSubdivLevel;
    m_baseTriangleIdx = micromapTriangleIdx;

    // w vertex
    m_vertices[0] = makeBaryUV_uint16(0, 0);
    // u vertex
    m_vertices[1] = makeBaryUV_uint16(getNumSegments(), 0);
    // v vertex
    m_vertices[2] = makeBaryUV_uint16(0, getNumSegments());

    m_vtxImportance = vtxImportance;

    for(uint32_t i = 0; i < 3; i++)
        m_edges[i] = computeEdgeInfo(i);

    allocateMem();

    const uint32_t valueIndexOffset = arrayGetV<uint32_t>(micromap.triangleValueIndexOffsets, micromapTriangleIdx);
    const uint32_t numVtxPerEdge    = 1u + getNumSegments();

    MicromapLayout layout = micromap.layout;
    if(!micromapLayoutIsValid(layout))
    {
        micromeshLayoutInitStandard(&layout, StandardLayoutType::eUmajor);
    }

    for(uint32_t u = 0; u < numVtxPerEdge; u++)
    {
        for(uint32_t v = 0; v < numVtxPerEdge - u; v++)
        {
            const uint32_t valueIdx  = layout.pfnGetMicroVertexIndex(u, v, m_baseSubdivLevel, layout.userData);
            const uint16_t dispValue = arrayGetV<uint16_t>(micromap.values, valueIndexOffset + valueIdx);
            m_reference[addr(u, v)]  = dispValue;
        }
    }
}

MeshEncoder::Triangle::Triangle(uint8_t subTriangleIdx, const Triangle& parent)
{
    m_subdivLevel      = parent.m_subdivLevel - 1;
    m_baseTriangleIdx  = parent.m_baseTriangleIdx;
    m_baseSubdivLevel  = parent.m_baseSubdivLevel;
    m_isWindingFlipped = parent.m_isWindingFlipped;
#ifdef ENABLE_SUBTRIANGLE_IDX
    m_subTriangleIdx = subTriangleIdx;
#endif

    // The post-encoding check flags must be propagated from parent to children.
    // subTriangle 1 doesn't have outer edges, but 0, 2 and 3 have two each.
    if(subTriangleIdx == 0)
    {
        m_postEncodingEdgeCheck[0] = parent.m_postEncodingEdgeCheck[0];
        m_postEncodingEdgeCheck[2] = parent.m_postEncodingEdgeCheck[2];
    }
    else if(subTriangleIdx == 2)
    {
        m_postEncodingEdgeCheck[0] = parent.m_postEncodingEdgeCheck[0];
        m_postEncodingEdgeCheck[1] = parent.m_postEncodingEdgeCheck[1];
    }
    else if(subTriangleIdx == 3)
    {
        // The winding order of this subtriangle is the opposite of its parent,
        // thus the edge flags must be swapped.
        m_postEncodingEdgeCheck[1] = parent.m_postEncodingEdgeCheck[2];
        m_postEncodingEdgeCheck[2] = parent.m_postEncodingEdgeCheck[1];
    }

    const BaryUV_uint16 w = parent.m_vertices[0];
    const BaryUV_uint16 u = parent.m_vertices[1];
    const BaryUV_uint16 v = parent.m_vertices[2];

    // Generate edges barys wrt the base triangle
    std::array<BaryUV_uint16, 3> mid_vertices{};
    for(uint32_t i = 0; i < 3; i++)
    {
        uint32_t j        = (1 + i) % 3;
        mid_vertices[i].u = (parent.m_vertices[i].u + parent.m_vertices[j].u) / 2;
        mid_vertices[i].v = (parent.m_vertices[i].v + parent.m_vertices[j].v) / 2;
    }

    const BaryUV_uint16 uw = mid_vertices[0];
    const BaryUV_uint16 uv = mid_vertices[1];
    const BaryUV_uint16 vw = mid_vertices[2];

    const std::array<float, 3> p_i = parent.m_vtxImportance;
    // UW, UV, VW
    const std::array<float, 3> mid_i{(p_i[0] + p_i[1]) / 2.0f, (p_i[1] + p_i[2]) / 2.0f, (p_i[2] + p_i[0]) / 2.0f};

    // This switch statement below implements the bird curve rule,
    // which determines the order of the sub-triangles and the order of the anchor vertices of each sub-triangle.

    // The curve enters at W and exits at V, visiting the sub-triangles 0, 1, 2 and 3 (in this order).
    // The last vertex/anchor of a sub-triangle is the first of the next sub-triangle (i.e. the new W vertex of the sub-triangle)
    // Note that the anchors of sub-triangles 0 and 2 are stored in counter-clockwise order, while 1 and 3 are in clockwise order.
    // ************************
    // *           V          *
    // *          /\          *
    // *         /  \         *
    // *        / 3  \        *
    // *       /      \       *
    // *   VW /________\ UV   *
    // *     /\        /\     *
    // *    /  \  1   /  \    *
    // *   /    \    /    \   *
    // *  /  0   \  /  2   \  *
    // * /        \/        \ *
    // *W ________UW________ U*
    // ************************

    std::array<uint32_t, 2> src_w_offset{}, src_step_u{}, src_step_v{};
    switch(subTriangleIdx)
    {
    case(0):
        // Select corners/anchors of sub-triangle
        m_vertices[0] = w;
        m_vertices[1] = uw;
        m_vertices[2] = vw;
        // Also interpolate the parent's vertex importances to get this triangle's vertex importances
        m_vtxImportance = {p_i[0], mid_i[0], mid_i[2]};
        // Initialize offset associated to the new w vertex in the bary space of the parent triangle
        // That's the starting point for copying data from the parent triangle to the sub-triangle
        src_w_offset[0] = 0;
        src_w_offset[1] = 0;
        // Generate vectors necessary to scan the parent triangle as we copy data to the sub-triangle
        // The role of these vectors is to rotate and/or flip the the copied data to account for the new anchors ordering
        src_step_u[0] = 1;
        src_step_u[1] = 0;
        src_step_v[0] = 0;
        src_step_v[1] = 1;
        break;
    case(1):
        m_vertices[0]      = vw;
        m_vertices[1]      = uv;
        m_vertices[2]      = uw;
        m_vtxImportance    = {mid_i[2], mid_i[1], mid_i[0]};
        src_w_offset[0]    = 0;
        src_w_offset[1]    = (int32_t)getNumSegments();
        src_step_u[0]      = 1;
        src_step_u[1]      = 0;
        src_step_v[0]      = 1;
        src_step_v[1]      = -1;
        m_isWindingFlipped = !m_isWindingFlipped;
        break;
    case(2):
        m_vertices[0]   = uw;
        m_vertices[1]   = u;
        m_vertices[2]   = uv;
        m_vtxImportance = {mid_i[0], p_i[1], mid_i[1]};
        src_w_offset[0] = (int32_t)getNumSegments();
        src_w_offset[1] = 0;
        src_step_u[0]   = 1;
        src_step_u[1]   = 0;
        src_step_v[0]   = 0;
        src_step_v[1]   = 1;
        break;
    case(3):
        m_vertices[0]      = uv;
        m_vertices[1]      = vw;
        m_vertices[2]      = v;
        m_vtxImportance    = {mid_i[1], mid_i[2], p_i[2]};
        src_w_offset[0]    = (int32_t)getNumSegments();
        src_w_offset[1]    = (int32_t)getNumSegments();
        src_step_u[0]      = -1;
        src_step_u[1]      = 0;
        src_step_v[0]      = -1;
        src_step_v[1]      = 1;
        m_isWindingFlipped = !m_isWindingFlipped;
        break;
    default:
        assert(0);
    }

    // Generate per-edge information necessary for propagating and validating edges
    for(uint32_t i = 0; i < 3; i++)
        m_edges[i] = computeEdgeInfo(i);

    // Copy reference data from the parent node.
    // Note that we don't re-sample the reference data from the original displacement map
    // because values along triangle edges might have been modified by previous encoding passes.
    allocateMem();
    const uint32_t numVtxPerEdge = 1u + getNumSegments();
    for(uint32_t dst_u = 0; dst_u < numVtxPerEdge; dst_u++)
    {
        for(uint32_t dst_v = 0; dst_v < numVtxPerEdge - dst_u; dst_v++)
        {
            int32_t src_u = src_w_offset[0];
            int32_t src_v = src_w_offset[1];

            src_u += (int32_t)dst_u * src_step_u[0];
            src_v += (int32_t)dst_u * src_step_u[1];

            src_u += (int32_t)dst_v * src_step_v[0];
            src_v += (int32_t)dst_v * src_step_v[1];

            m_reference[addr(dst_u, dst_v, m_subdivLevel)] = parent.m_reference[addr(src_u, src_v, parent.m_subdivLevel)];
        }
    }
}

void MeshEncoder::Triangle::bin(container::vector<container::vector<Triangle*>>& bins)
{
    // Nothing to do
    if(m_isEncoded)
        return;

    // Traverse hierarchy until we find a leaf node that has not been encoded yet
    if(!isLeafNode())
    {
        for(uint32_t i = 0; i < 4; i++)
        {
            m_children[i]->bin(bins);
        }
    }
    else
    {
        // Set the mask associated to the subdivision level of the base triangle associated to this subtriangle.
        // We will use it later to encode subtriangles in subdvision level order (for a given micromesh type).
        bins[m_baseSubdivLevel].push_back(this);
    }
}

bool MeshEncoder::Triangle::encode(const BlockFormatDispC1 fmt, bool isLastFormat, float minPSNR)
{
    // Nothing to do
    if(m_isEncoded)
        return true;

    // Split this node if it is too large to be encoded using the current micromesh configuration
    const auto& microMeshConfig = getMicromeshConfig(fmt);
    if(isLeafNode() && getNumSegments() > microMeshConfig.numSegments)
    {
        // Each instantiated child node gets its subset of data from the parent node
        for(uint8_t i = 0; i < 4; i++)
        {
            m_children[i] = std::make_unique<Triangle>(i, *this);
        }

        // Release parent node data
        deallocateMem();
    }

    // Traverse hierarchy until we find a leaf node that has not been encoded yet
    if(!isLeafNode())
    {
        bool allChildrenEncoded = true;
        for(uint32_t i = 0; i < 4; i++)
        {
            allChildrenEncoded = m_children[i]->encode(fmt, isLastFormat, minPSNR) && allChildrenEncoded;
        }

        // Only required with DISABLE_MIXED_FORMATS
        if(!allChildrenEncoded)
        {
            // Not all the subtriangles passed the quality requirements! Clear
            // their encoded information so we try again with the next setting.
            clearEncoded();
        }

        m_isEncoded = allChildrenEncoded;

        return allChildrenEncoded;
    }
    else if(getNumSegments() == microMeshConfig.numSegments || isLastFormat)
    {
        blockEncode(fmt, m_subdivLevel, m_reference, m_intermediate);

        // Verify whether encoding succeeded

        // Also, only check encoding quality if we have more micromesh configs available.
        // Otherwise just accept the result, no matter what.
        if(isLastFormat)
        {
            m_isEncoded = true;
        }
        else
        {
            m_isEncoded = satisfiesCompressionControl(fmt, minPSNR);
        }

        if(m_isEncoded)
        {
            // Update bounds
            m_displacementBounds         = {0xFFFF, 0};
            const uint32_t numVtxPerEdge = 1 + getNumSegments();
            for(uint32_t u = 0; u < numVtxPerEdge; u++)
            {
                for(uint32_t v = 0; v < numVtxPerEdge - u; v++)
                {
                    m_displacementBounds.x = std::min(m_displacementBounds.x, uint16_t(m_intermediate.m_decoded[addr(u, v)]));
                    m_displacementBounds.y = std::max(m_displacementBounds.y, uint16_t(m_intermediate.m_decoded[addr(u, v)]));
                }
            }

            m_encodedBlockFormat = fmt;
        }

        return m_isEncoded;
    }

    return m_isEncoded;
}

void MeshEncoder::Triangle::computeL2Err(double& l2Err, double& numElems) const
{
    if(!isLeafNode())
    {
        for(uint32_t i = 0; i < 4; i++)
            m_children[i]->computeL2Err(l2Err, numElems);
    }
    else
    {
        const auto     decoded       = m_intermediate.m_decoded;
        const auto     reference     = m_reference;
        const uint32_t numVtxPerEdge = 1u + getNumSegments();
        for(uint32_t u = 0; u < numVtxPerEdge; u++)
        {
            for(uint32_t v = 0; v < numVtxPerEdge - u; v++)
            {
                // We skip 50% of the edge displacements to avoid double counting (i.e. we assume most edges are shared)
                const double faceWeight = (v == 0 || u == 0 || (numVtxPerEdge - v - u == 0)) ? 0.5 : 1.0;

                // Interpolate vertex importance for this value. Note that we don't multiply this into the numElems sum!
                const double importance = m_vtxImportance[0] * float(u) / float(getNumSegments())
                                          + m_vtxImportance[1] * float(v) / float(getNumSegments())
                                          + m_vtxImportance[2] * (1.0f - float(u + v) / float(getNumSegments()));

                const uint32_t i     = addr(u, v);
                const double   delta = (double)(reference[i] - decoded[i]);
                l2Err += faceWeight * importance * delta * delta;
                numElems += faceWeight;
            }
        }
    }
}

float MeshEncoder::Triangle::computePSNR() const
{
    double l2Err = 0, numElems = 0;
    computeL2Err(l2Err, numElems);
    return computePSNRGeneral(l2Err, numElems);
}

void MeshEncoder::Triangle::getLocalEdgeBarys(uint32_t localEdgeIdx, BaryUV_uint16& a, BaryUV_uint16& b) const
{
    switch(localEdgeIdx)
    {
    case(0):
        a = {0, 0};
        b = {(uint16_t)getNumSegments(), 0};
        break;
    case(1):
        a = {(uint16_t)getNumSegments(), 0};
        b = {0, (uint16_t)getNumSegments()};
        break;
    case(2):
        a = {0, (uint16_t)getNumSegments()};
        b = {0, 0};
        break;
    }
}

template <typename T>
void MeshEncoder::Triangle::getEdgeData(const T* data, uint32_t localEdgeIdx, container::vector<T>& edge) const
{
    // Get edge endpoints in the local sub-triangle barycentric space
    BaryUV_uint16 a, b;
    getLocalEdgeBarys(localEdgeIdx, a, b);

    // Return a copy of the edge data
    std::array<int32_t, 2> coords = {a.u, a.v};
    const std::array<int32_t, 2> delta = {(b.u - a.u) / (int32_t)getNumSegments(), (b.v - a.v) / (int32_t)getNumSegments()};
    for(uint32_t i = 0; i <= getNumSegments(); i++)
    {
        edge[i] = data[addr(coords[0], coords[1])];
        coords[0] += delta[0];
        coords[1] += delta[1];
    }
}

void MeshEncoder::Triangle::getReferenceEdge(uint32_t edgeIdx, container::vector<uint16_t>& edge) const
{
    getEdgeData(m_reference, edgeIdx, edge);
}

void MeshEncoder::Triangle::getDecodedEdge(uint32_t edgeIdx, container::vector<uint16_t>& edge) const
{
    getEdgeData(m_intermediate.m_decoded, edgeIdx, edge);
}

void MeshEncoder::Triangle::getEncodedFormatCountInfo(BlockFormatDispC1& fmt, uint64_t& numLeafTriangles) const
{
    // Find the first leaf triangle:
    const Triangle* firstLeafTriangle = this;
    while(!firstLeafTriangle->isLeafNode())
    {
        firstLeafTriangle = firstLeafTriangle->m_children[0].get();
    }

    assert(firstLeafTriangle->getIsEncoded());

    fmt = firstLeafTriangle->m_encodedBlockFormat;
    // Assuming uniform block format, compute the number of leaf triangles
    // from the subdivision levels:
    numLeafTriangles = 1ULL << (2 * (getBaseSubdivLevel() - firstLeafTriangle->getSubdivLevel()));
}

void MeshEncoder::Triangle::appendLeafTriangles(container::vector<const Triangle*>& triangles) const
{
    if(!isLeafNode())
    {
        // Looking for leaf nodes..
        for(uint32_t i = 0; i < 4; i++)
            m_children[i]->appendLeafTriangles(triangles);
    }
    else
    {
        triangles.push_back(this);
    }
}

bool MeshEncoder::Triangle::processEdge(EdgeMode mode, const EdgeInfo& srcEdgeInfo, const container::vector<uint16_t>& srcEdgeData)
{
    const uint32_t numVertices = (uint32_t)srcEdgeData.size();
    const uint32_t numSegments = numVertices - 1u;

    if(!isLeafNode())
    {
        if(getNumSegments() > numSegments)
        {
            for(uint32_t i = 0; i < 4; i++)
            {
                if(!m_children[i]->processEdge(mode, srcEdgeInfo, srcEdgeData))
                    return false;
            }
        }
    }
    else
    {
        const bool setReference     = mode == EdgeMode::eSetReference && !m_isEncoded;
        const bool preEncValidation = mode == EdgeMode::ePreEncodeValidation && getNumSegments() == numSegments;
        const bool postEncValidation = mode == EdgeMode::ePostEncodeValidation && m_isEncoded && getNumSegments() == numSegments;
        if(setReference || preEncValidation || postEncValidation)
        {
            // We process micromesh types in compression order (from most to least compressed) and if we reach a unencoded leaf node that doesn't
            // have the right number of segments then we have not processed triangles in the right order.
            // The general rule is that mid-encoding all unencoded leaf nodes must have the same number of segments.
            assert(getNumSegments() == numSegments
                   && "This should not be possible. If the number of segments doesn't match something went wrong.");

            // We need to check whether this is the right edge to process.
            // First, we get the starting/ending point of this edge, then we check whether it matches the ending/starting point of the other edge.
            // If it does, this is edge we were looking for and we update its reference values.
            for(uint32_t localEdgeIdx = 0; localEdgeIdx < 3u; localEdgeIdx++)
            {
                const BaryUV_uint16 a = m_vertices[localEdgeIdx];
                const BaryUV_uint16 b = m_vertices[(1 + localEdgeIdx) % 3];

                bool match = doEndpointsMatch(a, srcEdgeInfo.barys[0], b, srcEdgeInfo.barys[1]);
                bool chtam = doEndpointsMatch(b, srcEdgeInfo.barys[0], a, srcEdgeInfo.barys[1]);

                if(match || chtam)
                {
                    if(setReference)
                    {
                        // Get edge endpoints in the local sub-triangle barycentric space
                        BaryUV_uint16 c, d;
                        getLocalEdgeBarys(localEdgeIdx, c, d);

                        // Copy reference edge
                        std::array<int32_t, 2>       coords{c.u, c.v};
                        const std::array<int32_t, 2> delta = {(d.u - c.u) / int32_t(getNumSegments()),
                                                              (d.v - c.v) / int32_t(getNumSegments())};

                        for(uint32_t i = 0; i < numVertices; i++)
                        {
                            m_reference[addr((uint32_t)coords[0], (uint32_t)coords[1])] =
                                match ? srcEdgeData[i] : srcEdgeData[numSegments - i];
                            coords[0] += delta[0];
                            coords[1] += delta[1];
                        }
                        return true;
                    }
                    else
                    {
                        container::vector<uint16_t> dstEdgeData(numVertices);
                        if(preEncValidation)
                            getReferenceEdge(localEdgeIdx, dstEdgeData);
                        else
                            getDecodedEdge(localEdgeIdx, dstEdgeData);

                        for(uint32_t i = 0; i < numVertices; i++)
                        {
                            // Check whether individual displacement values along edges match
                            if(match ? dstEdgeData[i] != srcEdgeData[i] : dstEdgeData[i] != srcEdgeData[numSegments - i])
                            {
                                assert(0 && "Edges don't match!");
                                //log::fatal("Decoded edges don't match!");
                                return false;
                            }
                        }
                        return true;
                    }
                }
            }
        }
    }

    return true;
}

bool MeshEncoder::Triangle::processOuterEdge(EdgeMode                           mode,
                                             const EdgeInfo&                    srcEdgeInfo,
                                             const container::vector<uint16_t>& srcEdgeData,
                                             uint32_t                           dstLocalEdgeIdx,
                                             bool                               areEdgesConcordant)
{
    const int32_t  numVertices = (int32_t)srcEdgeData.size();
    const uint32_t numSegments = (uint32_t)(numVertices - 1);

    if(!isLeafNode())
    {
        if(getNumSegments() > numSegments)
        {
            for(uint32_t i = 0; i < 4; i++)
            {
                if(!m_children[i]->processOuterEdge(mode, srcEdgeInfo, srcEdgeData, dstLocalEdgeIdx, areEdgesConcordant))
                    return false;
            }
        }
    }
    else
    {
        const bool match = getNumSegments() == numSegments;  // do src and dst edges have the same number of segments?
        const uint32_t dst2x = getBaseNumSegments() == 2u * srcEdgeInfo.numBaseSegments ? 1u : 0u;  // does the dst edge have 2x the resolution of the src edge?

        const bool setReference      = mode == EdgeMode::eSetReference && !m_isEncoded;
        const bool preEncValidation  = mode == EdgeMode::ePreEncodeValidation && (match || dst2x);
        const bool postEncValidation = mode == EdgeMode::ePostEncodeValidation && m_isEncoded && (match || dst2x);

        if(setReference || preEncValidation || postEncValidation)
        {
            // We process micromesh types in compression order (from most to least compressed) and if we reach a unencoded leaf node that doesn't
            // have the right number of segments then we have not processed triangles in the right order.
            assert((match || dst2x) && "Unexpected number of segments. Did two adjacent triangles have subdivision levels that differed by more than 1?");

            for(uint32_t dstEdgeIdx = 0; dstEdgeIdx < 3u; dstEdgeIdx++)
            {
                // If this is the right edge to process..
                const auto& dstEdgeInfo = getEdgeInfo(dstEdgeIdx);
                if(dstEdgeInfo.isOuterEdge && dstLocalEdgeIdx == dstEdgeInfo.baseEdgeIdx
                   && srcEdgeInfo.numBaseSegments <= getBaseNumSegments())
                {
                    // To copy or compare data between the two edges we first determine the endpoints of their intervals **with respect to the space of the dst edge**.
                    // Note that the dst edge can have twice the resolution of the src edge, in which case we "stretch" by 2x the src edge interval.
                    int32_t srcStart = srcEdgeInfo.outerEdgeStart << dst2x;
                    int32_t srcStop  = srcEdgeInfo.outerEdgeStop << dst2x;
                    // In the common case (2-valence edge of a manifold mesh) the data coming from the src edge is ordered in the opposite direction wrt the dst edge,
                    // which would normally require to "flip" the src edge interval endpoints.
                    // When areEdgesConcordant is true the edges data are sorted in the same direction and we don't need to flip the src edge interval.
                    srcStart = areEdgesConcordant ? srcStart : (int32_t)getBaseNumSegments() - srcStart;
                    srcStop  = areEdgesConcordant ? srcStop : (int32_t)getBaseNumSegments() - srcStop;
                    // Dst edge interval endpoints
                    const auto dstStart = dstEdgeInfo.outerEdgeStart;
                    const auto dstStop  = dstEdgeInfo.outerEdgeStop;

                    // In the inner loop we walk over the dst edge vertices. We do this along the edge interval..
                    const int32_t dstDelta = (dstStop - dstStart > 0 ? 1 : -1) * (1 << dst2x);
                    // ..and also in the barycentric space of the dst triangle.
                    BaryUV_uint16 a, b;
                    getLocalEdgeBarys(dstEdgeIdx, a, b);
                    const std::array<int32_t, 2> dstBaryDelta = {(b.u - a.u) / (int32_t)(getNumSegments() >> dst2x),
                                                                 (b.v - a.v) / (int32_t)(getNumSegments() >> dst2x)};

                    // Loop over dst edge interval
                    bool                   done = false;
                    std::array<int32_t, 2> dstBarys{a.u, a.v};
                    int32_t                dstPos = dstStart;
                    while(!done)
                    {
                        // If this the last iteration we are done
                        done = dstPos == dstStop;

                        // If the dst vertex is located inside the src edge interval..
                        if((dstPos >= srcStart && dstPos <= srcStop) || (dstPos >= srcStop && dstPos <= srcStart))
                        {
                            // Compute index into src edge data
                            const size_t   srcDataIdx  = (size_t)std::abs(dstPos - srcStart) >> dst2x;
                            const uint32_t dstBaryAddr = addr((uint32_t)dstBarys[0], (uint32_t)dstBarys[1]);

                            if(setReference)
                            {
                                m_reference[dstBaryAddr] = srcEdgeData[srcDataIdx];
                                // A reference edge from a 2x lower res triangle is not always representable by a micromesh type
                                // that is "more compressed" than the one used to encode the source edge in the first place.
                                // To avoid cracks we set this per-edge flag to remind us at encoding time to check
                                // if every other displacement value along this edge exactly matches the reference.
                                // If it doesn't we mark the triangle as "not encoded" and try again later with a less compressed micromesh type.
                                m_postEncodingEdgeCheck[dstEdgeIdx] = m_postEncodingEdgeCheck[dstEdgeIdx] || (dst2x > 0);
                            }
                            else
                            {
                                if(preEncValidation ? m_reference[dstBaryAddr] != srcEdgeData[srcDataIdx] :
                                                      m_intermediate.m_decoded[dstBaryAddr] != srcEdgeData[srcDataIdx])
                                {
                                    assert(0 && "Edges don't match!");
                                    return false;
                                }
                            }
                        }

                        // Update dst edge barycentric & interval positions
                        dstBarys[0] += dstBaryDelta[0];
                        dstBarys[1] += dstBaryDelta[1];
                        dstPos += dstDelta;
                    }
                    return true;
                }
            }
        }
    }
    return true;
}

bool MeshEncoder::Triangle::areLowerResNeighborsEncoded(uint32_t dstLocalEdgeIdx, uint32_t numBaseSegments)
{
    if(!isLeafNode())
    {
        for(uint32_t i = 0; i < 4; i++)
        {
            if(!m_children[i]->areLowerResNeighborsEncoded(dstLocalEdgeIdx, numBaseSegments))
                return false;
        }
    }
    else
    {
        for(uint32_t dstEdgeIdx = 0; dstEdgeIdx < 3u; dstEdgeIdx++)
        {
            const auto& dstEdgeInfo = getEdgeInfo(dstEdgeIdx);
            // We can't encode a triangle that shares an outer edge with an not-yet-encoded subtriangle of lower resolution.
            // Lower res subtriangles must be encoded first to avoid cracks. We'll skip it for now and try again later.
            if(dstEdgeInfo.isOuterEdge && dstLocalEdgeIdx == dstEdgeInfo.baseEdgeIdx
               && numBaseSegments > getBaseNumSegments() && (!m_isEncoded))
                return false;
        }
    }

    return true;
}

void MeshEncoder::Triangle::appendEncodedTriangles(container::vector<const Triangle*>& triangles) const
{
    if(!isLeafNode())
    {
        // Looking for leaf nodes..
        for(uint32_t i = 0; i < 4; i++)
            m_children[i]->appendEncodedTriangles(triangles);
    }
    else if(m_isEncoded)
    {
        triangles.push_back(this);
    }
}

void MeshEncoder::Triangle::clearEncoded()
{
    m_isEncoded = false;

    if(!isLeafNode())
    {
        for(uint32_t i = 0; i < 4; i++)
        {
            m_children[i]->clearEncoded();
        }
    }
}

// Returns the sign of the direction from a to b: either +1, 0, or -1.
int32_t directionSign(uint16_t a, uint16_t b)
{
    if(b > a)
    {
        return 1;
    }
    if(b < a)
    {
        return -1;
    }
    return 0;
}

void MeshEncoder::Triangle::writeMip(void* output, uint16_t mipSubdivLevel, const MicromapLayout& layout) const
{
    assert(m_isEncoded);

    if(!isLeafNode())
    {
        for(uint32_t i = 0; i < 4; i++)
        {
            m_children[i]->writeMip(output, mipSubdivLevel, layout);
        }
    }
    else
    {
        // The base triangle's coordinates are scaled down by this factor to
        // get the coordinates of the mip.
        const uint32_t decimationFactor = 1ULL << uint32_t(m_baseSubdivLevel - mipSubdivLevel);
        // For now, we use a fairly brute-force approach, iterating over all
        // values and copying those whose u and v coordinates (in the
        // coordinate space of the base triangle) are 0 mod decimationFactor.
        // Start by computing the local -> base triangle transform.
        const std::array<int32_t, 2> origin{m_vertices[0].u, m_vertices[0].v};
        const std::array<int32_t, 2> stepU{directionSign(m_vertices[0].u, m_vertices[1].u),
                                           directionSign(m_vertices[0].v, m_vertices[1].v)};
        const std::array<int32_t, 2> stepV{directionSign(m_vertices[0].u, m_vertices[2].u),
                                           directionSign(m_vertices[0].v, m_vertices[2].v)};
        const int32_t                numVtxPerEdge = 1 + int32_t(getNumSegments());
        for(int32_t u = 0; u < numVtxPerEdge; u++)
        {
            for(int32_t v = 0; v < numVtxPerEdge - u; v++)
            {
                const int32_t baseU = origin[0] + stepU[0] * u + stepV[0] * v;
                const int32_t baseV = origin[1] + stepU[1] * u + stepV[1] * v;
                if((baseU % decimationFactor) == 0 && (baseV % decimationFactor) == 0)
                {
                    const uint16_t decompressedValue = m_intermediate.m_decoded[addr(u, v)];
                    const uint32_t outputIndex = layout.pfnGetMicroVertexIndex(baseU / decimationFactor, baseV / decimationFactor,
                                                                               mipSubdivLevel, layout.userData);
                    packedWriteR11UnormPackedAlign32(output, outputIndex, decompressedValue);
                }
            }
        }
    }
}

void MeshEncoder::Triangle::writePackedBlock(void* packedData) const
{
    blockPackData(m_encodedBlockFormat, m_subdivLevel, m_intermediate, packedData);
}


void MeshEncoder::Triangle::allocateMem()
{
    const uint32_t elementArraySize = subdivLevelGetVertexCount(m_baseSubdivLevel);

    m_allocated                  = std::make_unique<uint16_t[]>(elementArraySize * 3);
    m_reference                  = m_allocated.get();
    m_intermediate.m_decoded     = m_reference + elementArraySize;
    m_intermediate.m_corrections = reinterpret_cast<int16_t*>(m_intermediate.m_decoded + elementArraySize);
}

void MeshEncoder::Triangle::deallocateMem()
{
    m_allocated                  = nullptr;
    m_reference                  = nullptr;
    m_intermediate.m_decoded     = nullptr;
    m_intermediate.m_corrections = nullptr;
}

bool MeshEncoder::Triangle::doEndpointsMatch(BaryUV_uint16 a_start, BaryUV_uint16 b_start, BaryUV_uint16 a_end, BaryUV_uint16 b_end) const
{
    return a_start.u == b_start.u && a_start.v == b_start.v && a_end.u == b_end.u && a_end.v == b_end.v;
}

Vector_uint16_2 MeshEncoder::Triangle::getCombinedDisplacementBounds() const
{
    auto bounds = m_displacementBounds;
    if(!isLeafNode())
    {
        for(uint32_t i = 0; i < 4; i++)
        {
            const auto childrenBounds = m_children[i]->getCombinedDisplacementBounds();
            bounds.x                  = std::min(bounds.x, childrenBounds.x);
            bounds.y                  = std::max(bounds.y, childrenBounds.y);
        }
    }
    return bounds;
}

MeshEncoder::Triangle::EdgeInfo MeshEncoder::Triangle::computeEdgeInfo(uint32_t localEdgeIdx) const
{
    MeshEncoder::Triangle::EdgeInfo info{};

    info.barys[0]        = m_vertices[localEdgeIdx];
    info.barys[1]        = m_vertices[(1 + localEdgeIdx) % 3];
    info.numBaseSegments = getBaseNumSegments();

    // We have an outer edge with respect to the base triangle
    // if one of the barycentric coords is always zero along the edge.
    uint32_t w[2] = {getBaseNumSegments() - info.barys[0].u - info.barys[0].v,
                     getBaseNumSegments() - info.barys[1].u - info.barys[1].v};

    // when v is zero we have an outer w-vertex --> u-vertex edge
    if(info.barys[0].v == 0 && info.barys[1].v == 0)
    {
        info.isOuterEdge    = true;
        info.baseEdgeIdx    = 0u;
        info.outerEdgeStart = info.barys[0].u;
        info.outerEdgeStop  = info.barys[1].u;
    }
    // when w is zero we have an outer u-vertex --> v-vertex edge
    else if(w[0] == 0 && w[1] == 0)
    {
        info.isOuterEdge    = true;
        info.baseEdgeIdx    = 1u;
        info.outerEdgeStart = info.barys[0].v;
        info.outerEdgeStop  = info.barys[1].v;
    }
    // when u is zero we have an outer v-vertex --> w-vertex edge
    else if(info.barys[0].u == 0 && info.barys[1].u == 0)
    {
        info.isOuterEdge    = true;
        info.baseEdgeIdx    = 2u;
        info.outerEdgeStart = (int32_t)getBaseNumSegments() - info.barys[0].v;
        info.outerEdgeStop  = (int32_t)getBaseNumSegments() - info.barys[1].v;
    }
    else
    {
        info.isOuterEdge    = false;
        info.baseEdgeIdx    = 0u;
        info.outerEdgeStart = 0u;
        info.outerEdgeStop  = 0u;
    }

    return info;
}

bool MeshEncoder::Triangle::satisfiesCompressionControl(const BlockFormatDispC1 fmt, const float minPSNR) const
{
    // Lossless micromeshes are always successfully encoded.
    const MicromeshConfig microMeshConfig = getMicromeshConfig(fmt);
    if(microMeshConfig.hasFlatEncoding())
        return true;

    const uint32_t numVtxPerEdge = 1u + getNumSegments();

    {
        const float PSNR = computePSNR();
        if(PSNR < minPSNR)
            return false;
    }

    for(uint32_t edgeIdx = 0; edgeIdx < 3u; edgeIdx++)
    {
        // If half of this edge reference values have been propagated from a 2x lower res triangle
        // we need to check whether we encoded them losslesly in order to avoid cracks.
        // These values are not always representable if the current triangle is being compressed
        // with a micromesh type that is "more compressed" than the one used to encode the 2x lower res triangle.
        // If the check fail we mark the subtriangle as not encoded and we'll try again later with a less compressed micromesh type.
        if(m_postEncodingEdgeCheck[edgeIdx])
        {
            // Get edge endpoints in the local sub-triangle barycentric space
            BaryUV_uint16 a, b;
            getLocalEdgeBarys(edgeIdx, a, b);

            // Copy reference edge
            std::array<int32_t, 2>       coords{a.u, a.v};
            const std::array<int32_t, 2> delta = {
                (b.u - a.u) / (int32_t)getNumSegments(),
                (b.v - a.v) / (int32_t)getNumSegments(),
            };

            for(uint32_t i = 0; i < numVtxPerEdge; i += 2)
            {
                const uint32_t coordAddr = addr((uint32_t)coords[0], (uint32_t)coords[1]);
                if (m_reference[coordAddr] != m_intermediate.m_decoded[coordAddr])
                    return false;
                // This multiplication by 2 instead of a left shift by 1 is
                // important: left shifting a negative integer is UB in C++.
                coords[0] += delta[0] * 2;
                coords[1] += delta[1] * 2;
            }
        }
    }

    return true;
}


///////////////////////////////////////////////////////////////////////////////
// MeshEncoder

static container::vector<BlockFormatDispC1> initBlockFormats(const uint32_t enabledBlockFormatBits)
{
    container::vector<BlockFormatDispC1> blockFormats;

    // To avoid cracks micromesh configurations must be added in this order (from highest to lowest compression ratio)
    if(enabledBlockFormatBits & (1 << uint32_t(BlockFormatDispC1::eR11_unorm_lvl5_pack1024)))
    {
        blockFormats.push_back(BlockFormatDispC1::eR11_unorm_lvl5_pack1024);
    }
    if(enabledBlockFormatBits & (1 << uint32_t(BlockFormatDispC1::eR11_unorm_lvl4_pack1024)))
    {
        blockFormats.push_back(BlockFormatDispC1::eR11_unorm_lvl4_pack1024);
    }
    if(enabledBlockFormatBits & (1 << uint32_t(BlockFormatDispC1::eR11_unorm_lvl3_pack512)))
    {
        blockFormats.push_back(BlockFormatDispC1::eR11_unorm_lvl3_pack512);
    }

    return blockFormats;
}

static bool validateBlockFormats(OpContext                                   ctx,
                                 const container::vector<BlockFormatDispC1>& blockFormats,
                                 const OpCompressDisplacement_settings*      settings)
{
    if(blockFormats.size() == 0)
    {
        LOGE(ctx, "No valid block formats were specified in enabledBlockFormatBits.");
        return false;
    }

    // NOTE: At the moment, I believe it's impossible to have an invalid block
    // configuration once we get here. So the code below could potentially
    // be removed (unless we decide to experiment with new formats again).

    // Validate individual correction bit width configurations
    for(size_t i = 0; i < blockFormats.size(); i++)
    {
        const auto& config = getMicromeshConfig(blockFormats[i]).numCorrBits;
        for(size_t j = 1; j < config.size(); j++)
        {
            if(config[j] > config[j - 1])
            {
                LOGE(ctx,
                     "Illegal configuration. To avoid cracks, the correction bit width for a given subdivision level "
                     "must not increase from the previous subdivision level.");
                return false;
            }
        }
    }

    // Validate the ensemble of correction configurations
    for(size_t i = 1; i < blockFormats.size(); i++)
    {
        const auto& curr = getMicromeshConfig(blockFormats[i]).numCorrBits;
        const auto& prev = getMicromeshConfig(blockFormats[i - 1]).numCorrBits;
        for(size_t j = 0; j < curr.size(); j++)
        {
            if(curr[j] < prev[j])
            {
                LOGE(ctx,
                     "Illegal configuration. To avoid cracks, the correction bit width for a given subdivision level "
                     "must not decrease when going from a finer/more compressed micromesh type to a coarser/less "
                     "compressed one.");
                return false;
            }
        }
    }

    // If `requireLosslessMeshEdges` was true, make sure we had a lossless
    // format. It's possible to construct a scenario where we could compress
    // them losslessly without the lossless format, but it's unlikely.
    if(settings->requireLosslessMeshEdges)
    {
        bool foundLosslessFormat = false;
        for(const BlockFormatDispC1& format : blockFormats)
        {
            if(getMicromeshConfig(format).hasFlatEncoding())
            {
                foundLosslessFormat = true;
                break;
            }
        }
        if(!foundLosslessFormat)
        {
            LOGE(ctx,
                 "The compressor was called with `requireLosslessMeshEdges == true`, but no lossless format was "
                 "enabled. Without a lossless format, it's unlikely that all edges can be losslessly encoded.");
            return false;
        }
    }

    return true;
}

Result MeshEncoder::batchEncode(OpContext ctx, const OpCompressDisplacement_settings* settings, const OpCompressDisplacement_input* inputUncompressed)
{
    // NOTE: Can we move format validation up to the API layer?
    const container::vector<BlockFormatDispC1> blockFormats = initBlockFormats(settings->enabledBlockFormatBits);
    if(!validateBlockFormats(ctx, blockFormats, settings))
    {
        return Result::eInvalidBlockFormat;  // eInvalidValue could also be OK
    }
    mipIgnoredSubdivLevel = settings->mipIgnoredSubdivLevel;

    // Create micromap triangle objects from the reference values.
    const Micromap& micromap             = *inputUncompressed->data;
    const uint64_t  numMicromapTriangles = micromap.triangleSubdivLevels.count;
    m_micromapTriangles.resize(numMicromapTriangles);
    // Iterate over the mesh triangle -> micromap triangle map, or the identity
    // if it exists.
    const bool hasMeshTriangleMap = !arrayIsEmpty(inputUncompressed->meshTriangleMappings);
    const uint64_t numMeshTriangles = (hasMeshTriangleMap ? inputUncompressed->meshTriangleMappings.count : numMicromapTriangles);

    // TODO: Support a many-to-one mesh-to-micromap function.
    // Since we don't support it yet, check for unintended behavior.
    {
        container::vector<bool> triReferencedAlready(numMicromapTriangles, false);
        for(uint64_t meshTriIdx = 0; meshTriIdx < numMeshTriangles; meshTriIdx++)
        {
            const uint64_t micromapTriIdx = meshGetTriangleMapping(inputUncompressed->meshTriangleMappings, meshTriIdx);

            if(micromapTriIdx > numMicromapTriangles)
            {
                LOGE(ctx,
                     "inputUncompressed->meshTriangleMappings maps mesh triangle %" PRIu64
                     " to micromap triangle %" PRIu64
                     ", which is greater than the number of micromap triangles (%" PRIu64 ").",
                     meshTriIdx, micromapTriIdx, numMicromapTriangles);
                return Result::eInvalidValue;
            }

            if(triReferencedAlready[micromapTriIdx])
            {
                LOGE(ctx,
                     "Multiple mesh triangles point to micromap triangle %" PRIu64
                     " via inputUncompressed->meshTriangleMappings. The encoder does not yet support a many-to-one "
                     "function here, but it's on the roadmap: please see the repository's issues for more information.",
                     micromapTriIdx);
                return Result::eInvalidValue;
            }

            triReferencedAlready[micromapTriIdx] = true;
        }
    }


    // Also construct the inverse map.
    container::vector<uint64_t> triangleMeshMappings(numMicromapTriangles, NO_MESH_TRIANGLE);
    ctx->parallel_items(numMeshTriangles, [&](uint64_t meshTriIdx, uint32_t, void*) {
        const uint64_t micromapTriIdx = meshGetTriangleMapping(inputUncompressed->meshTriangleMappings, meshTriIdx);

        std::array<float, 3> vtxImportance = {1.0f, 1.0f, 1.0f};
        if(!arrayIsEmpty(inputUncompressed->perVertexImportance))
        {
            const Vector_uint32_3 vtxIndices = arrayGetV<Vector_uint32_3>(inputUncompressed->topology->triangleVertices, meshTriIdx);
            vtxImportance[0] = arrayGetV<float>(inputUncompressed->perVertexImportance, vtxIndices.x);
            vtxImportance[1] = arrayGetV<float>(inputUncompressed->perVertexImportance, vtxIndices.y);
            vtxImportance[2] = arrayGetV<float>(inputUncompressed->perVertexImportance, vtxIndices.z);
        }

        triangleMeshMappings[micromapTriIdx] = meshTriIdx;

        m_micromapTriangles[micromapTriIdx] = std::make_unique<Triangle>(*inputUncompressed->data, micromapTriIdx, vtxImportance);

        // Handle `requireLosslessMeshEdges` by marking which edges are along
        // mesh edges, and thus must be perfectly encoded. We say an edge is
        // a mesh edge if it has only 1 neighboring triangle - so e.g. if it
        // has 3 neighboring triangles, it's not counted as a mesh edge.
        if(settings->requireLosslessMeshEdges)
        {
            const Vector_uint32_3 triangleEdgeIndices =
                arrayGetV<Vector_uint32_3>(inputUncompressed->topology->triangleEdges, meshTriIdx);
            for(uint32_t i = 0; i < 3; i++)
            {
                const uint32_t edgeIdx = triangleEdgeIndices[i];
                // The check for INVALID_INDEX here avoids indexing out of
                // bounds. edgeIdx can be INVALID_INDEX when triangle
                // meshTriIdx is adjacent to a degenerate triangle. When this
                // occurs, we probably want to ensure the edge is perfectly
                // encoded - otherwise, we could get a visual discontinuity,
                // like a visible border between mesh triangles (although
                // the mesh will still be watertight).
                if(edgeIdx == INVALID_INDEX
                   || arrayGetV<Range32>(inputUncompressed->topology->edgeTriangleRanges, edgeIdx).count < 2)
                {
                    m_micromapTriangles[micromapTriIdx]->postEncodingEdgeCheck(i) = true;
                }
            }
        }
    });

    if(settings->validateInputs)
    {
        // Validate edges before encoding
        if(!validateEdges(ctx, *inputUncompressed->topology, inputUncompressed->meshTriangleMappings, triangleMeshMappings, true))
        {
            // validateEdges will already have printed a message
            return Result::eMismatchingInputEdgeValues;
        }
    }

    // Bin not-yet-encoded triangles according their base resolution, ordered from low to high res
    container::vector<container::vector<Triangle*>> bins(config_maxSubdLevel);
    for(auto& triangle : m_micromapTriangles)
        triangle->bin(bins);

    // Encode triangles in base triangle resolution order, one bin at a time, starting from the lowest resolution ones.
    for(auto& bin : bins)
    {
        if(!bin.empty())
        {
            // Loop over micromesh type, from most to least compressed
            for(size_t fmtIdx = 0; fmtIdx < blockFormats.size(); fmtIdx++)
            {
                // Loop over subtriangle candidates for encoding
                container::vector<const Triangle*> encoded;
                // Since multiple threads attempt to append to `encoded`, we
                // lock on this to provide safe multithreading:
                std::mutex encodedMutex;

                ctx->parallel_items(bin.size(), [&](uint64_t triangleIdx, uint32_t threadIdx, void* userData) {
                    Triangle* triangle = bin[triangleIdx];
                    // Have to avoid trying to re-encode already encoded triangles
                    if(triangle->getIsEncoded())
                    {
                        return;
                    }

                    if(triangle->encode(blockFormats[fmtIdx], (fmtIdx == (blockFormats.size() - 1)), settings->minimumPSNR))
                    {
                        assert(triangle->getIsEncoded());
                        // With DISABLE_MIXED_FORMATS mode, this means that
                        // the whole triangle has been successfully encoded.
                        // So add all its subtriangles to the list:
                        const std::lock_guard<std::mutex> lock(encodedMutex);
                        triangle->appendEncodedTriangles(encoded);
                    }
                });

                // In order to avoid cracks *decoded edges* of encoded subtriangles are used to
                // update the *reference edges* of not-yet-encoded subtriangles.
                processEdges(ctx, EdgeMode::eSetReference, *inputUncompressed->topology, inputUncompressed->meshTriangleMappings,
                             triangleMeshMappings, encoded, getMicromeshConfig(blockFormats[fmtIdx]).numSegments);
            }
        }
    }

    if(settings->validateOutputs)
    {
        // Validate edges after encoding
        if(!validateEdges(ctx, *inputUncompressed->topology, inputUncompressed->meshTriangleMappings, triangleMeshMappings, false))
        {
            // validateEdges will already have printed a message
            return Result::eMismatchingOutputEdgeValues;
        }
    }

    return Result::eSuccess;
}

void MeshEncoder::fillEncodedSizes(MicromapCompressed& o) const
{
    // Fill in all members we can without looping.
    o.values.byteStride              = 1;
    o.values.format                  = Format::eDispC1_r11_unorm_block;
    o.triangleSubdivLevels.count     = m_micromapTriangles.size();
    o.triangleValueByteOffsets.count = m_micromapTriangles.size();
    o.triangleBlockFormats.count     = m_micromapTriangles.size();

    // Loop over micromap triangles to determine the maximum subdivision level
    // and the byte length of the values array.
    o.values.count   = 0;
    o.minSubdivLevel = ~0;
    o.maxSubdivLevel = 0;
    for(const auto& baseTriangle : m_micromapTriangles)
    {
        // Note that we can now assume all the leaf triangles' formats are
        // identical!
        uint32_t baseSubdiv = baseTriangle->getBaseSubdivLevel();
        o.minSubdivLevel    = std::min(o.minSubdivLevel, baseSubdiv);
        o.maxSubdivLevel    = std::max(o.maxSubdivLevel, baseSubdiv);

        BlockFormatDispC1 singleFormat;
        uint64_t          numLeafTriangles;
        baseTriangle->getEncodedFormatCountInfo(singleFormat, numLeafTriangles);
        const uint32_t blockSizeBytes = getMicromeshConfig(singleFormat).blockSizeInBits / 8;

        // If we have a 128-byte format, we must add padding to start at
        // a 128-byte interval.
        if(blockSizeBytes == 128)
        {
            o.values.count = roundUpToMultipleOf128(o.values.count);
        }
        o.values.count += blockSizeBytes * numLeafTriangles;
    }
}

void MeshEncoder::writeCompressedData(OpContext ctx, OpCompressDisplacement_output& outputCompressed) const
{
    MicromapCompressed& o = *outputCompressed.compressed;

    // Compute o.triangleValueByteOffsets.
    // NOTE: This could be parallelized as a prefix sum.
    const size_t numTriangles = m_micromapTriangles.size();
    {
        // We assume overflow has already been checked.
        uint32_t valueByteOffset = 0;
        for(size_t triIdx = 0; triIdx < numTriangles; triIdx++)
        {
            const auto& baseTriangle = m_micromapTriangles[triIdx];

            BlockFormatDispC1 singleFormat;
            uint64_t          numLeafTriangles;
            baseTriangle->getEncodedFormatCountInfo(singleFormat, numLeafTriangles);
            const uint32_t blockSizeBytes = getMicromeshConfig(singleFormat).blockSizeInBits / 8;

            // If we have a 128-byte format, we must add padding to start at
            // a 128-byte interval.
            if(blockSizeBytes == 128)
            {
                valueByteOffset = roundUpToMultipleOf128(valueByteOffset);
            }
            arraySetV<uint32_t>(o.triangleValueByteOffsets, triIdx, valueByteOffset);
            valueByteOffset += blockSizeBytes * uint32_t(numLeafTriangles);

            // Also write singleFormat, since we have it here.
            arraySetV<uint16_t>(o.triangleBlockFormats, triIdx, static_cast<uint16_t>(singleFormat));
        }
    }

    // Now write the remaining data in parallel.
    ctx->parallel_items(numTriangles, [&](uint64_t triIdx, uint32_t threadIdx, void* userData) {
        const auto& baseTriangle = m_micromapTriangles[triIdx];
        arraySetV<uint16_t>(o.triangleSubdivLevels, triIdx, baseTriangle->getBaseSubdivLevel());

        BlockFormatDispC1 singleFormat;
        uint64_t          numLeafTriangles_unused;
        baseTriangle->getEncodedFormatCountInfo(singleFormat, numLeafTriangles_unused);
        const uint32_t blockSizeBytes = getMicromeshConfig(singleFormat).blockSizeInBits / 8;

        // The memory at which to write the next block; always aligned to
        // the block size from the above padding.
        uint8_t* valueWritePtr = arrayGet<uint8_t>(o.values, arrayGetV<uint32_t>(o.triangleValueByteOffsets, triIdx));
        container::vector<const Triangle*> leafTriangles;
        baseTriangle->appendLeafTriangles(leafTriangles);

        for(const auto& leafTriangle : leafTriangles)
        {
            leafTriangle->writePackedBlock(valueWritePtr);
            valueWritePtr += blockSizeBytes;
        }
    });

    // Handle the optional `triangleMinMaxs` output.
    if(outputCompressed.triangleMinMaxs.data)
    {
        micromesh::ArrayInfo& triangleMinMaxs = outputCompressed.triangleMinMaxs;
        ctx->parallel_items(numTriangles, [&](uint64_t triIdx, uint32_t threadIdx, void* userData) {
            const auto&           baseTriangle = m_micromapTriangles[triIdx];
            const Vector_uint16_2 bounds       = baseTriangle->getCombinedDisplacementBounds();
            arraySetV<uint16_t>(triangleMinMaxs, 2 * triIdx + 0, bounds.x);
            arraySetV<uint16_t>(triangleMinMaxs, 2 * triIdx + 1, bounds.y);
        });
    }

    // Handle the optional `mipData` output.
    if(outputCompressed.mipData)
    {
        MicromapPacked& mipData = *outputCompressed.mipData;
        // Tracks whether any thread encountered a subdiv level less than the mip level.
        std::atomic<bool> invalidSubdivLevel(false);

        ctx->parallel_items(numTriangles, [&](uint64_t triIdx, uint32_t threadIdx, void* userData) {
            const auto&    baseTriangle    = m_micromapTriangles[triIdx];
            const uint32_t baseSubdivLevel = baseTriangle->getBaseSubdivLevel();
            const uint16_t mipSubdivLevel  = arrayTypedGetV(mipData.triangleSubdivLevels, triIdx);
            if(mipSubdivLevel == mipIgnoredSubdivLevel)
            {
                return;
            }
            if(mipSubdivLevel > baseSubdivLevel)
            {
                invalidSubdivLevel.store(true);
                return;
            }

            const uint32_t firstByte = arrayTypedGetV(mipData.triangleValueByteOffsets, triIdx);
            baseTriangle->writeMip(reinterpret_cast<uint8_t*>(mipData.values.data) + firstByte, /* output data */
                                   mipSubdivLevel,                                              /* mipSubdivLevel */
                                   mipData.layout);                                             /* layout */
        });
    }
}

float MeshEncoder::computePSNR() const
{
    double l2Err = 0, numElems = 0;
    for(const auto& baseTriangle : m_micromapTriangles)
    {
        baseTriangle->computeL2Err(l2Err, numElems);
    }

    return computePSNRGeneral(l2Err, numElems);
}


bool MeshEncoder::validateEdges(OpContext                          ctx,
                                const MeshTopology&                topology,
                                const ArrayInfo_uint32&            meshTriangleMappings,
                                const container::vector<uint64_t>& triangleMeshMappings,
                                bool                               preEncodingValidation)
{
    EdgeMode mode = preEncodingValidation ? EdgeMode::ePreEncodeValidation : EdgeMode::ePostEncodeValidation;

    for(auto& baseTriangle : m_micromapTriangles)
    {
        // Get all encoded sub-triangles within this base triangle
        container::vector<const Triangle*> triangles;
        if(preEncodingValidation)
            baseTriangle->appendLeafTriangles(triangles);
        else
            baseTriangle->appendEncodedTriangles(triangles);

        processEdges(ctx, mode, topology, meshTriangleMappings, triangleMeshMappings, triangles);
    }

    return true;
}

bool MeshEncoder::processEdges(OpContext                           ctx,
                               EdgeMode                            mode,
                               const MeshTopology&                 topology,
                               const ArrayInfo_uint32&             meshTriangleMappings,
                               const container::vector<uint64_t>&  triangleMeshMappings,
                               container::vector<const Triangle*>& triangles,
                               uint32_t                            numSegments)
{
    container::vector<uint16_t> srcEdgeData(1u + numSegments);

    for(uint64_t itemIdx = 0; itemIdx < triangles.size(); itemIdx++)
    {
        auto& triangle = triangles[itemIdx];

        if(mode != EdgeMode::eSetReference)
            srcEdgeData.resize(1u + triangle->getNumSegments());

        // Loop over its edges
        for(uint32_t srcLocalEdgeIdx = 0; srcLocalEdgeIdx < 3; srcLocalEdgeIdx++)
        {
            // Get all we need to know about this edge
            const auto& srcEdgeInfo = triangle->getEdgeInfo(srcLocalEdgeIdx);

            if(!srcEdgeInfo.isOuterEdge)
            {
                if(mode == EdgeMode::eSetReference || mode == EdgeMode::ePostEncodeValidation)
                    triangle->getDecodedEdge(srcLocalEdgeIdx, srcEdgeData);
                else
                    triangle->getReferenceEdge(srcLocalEdgeIdx, srcEdgeData);

                if(!m_micromapTriangles[triangle->getBaseTriangleIdx()]->processEdge(mode, srcEdgeInfo, srcEdgeData))
                {
                    return false;
                }
            }
            // If this is an outer edge then it can possibly be shared with a sub-triangle within a different base triangle
            else
            {
                if(mode == EdgeMode::eSetReference || mode == EdgeMode::ePostEncodeValidation)
                    triangle->getDecodedEdge(srcLocalEdgeIdx, srcEdgeData);
                else
                    triangle->getReferenceEdge(srcLocalEdgeIdx, srcEdgeData);

                // Get the edge indices of the base triangle associated to this sub-triangle
                const auto srcTopoFaceIdx = triangleMeshMappings[triangle->getBaseTriangleIdx()];
                const Vector_uint32_3 triangleEdgeIndices = arrayGetV<Vector_uint32_3>(topology.triangleEdges, srcTopoFaceIdx);
                const auto srcMeshEdgeIdx = triangleEdgeIndices[srcEdgeInfo.baseEdgeIdx];
                // Quick fix for an issue: MeshTopology allows setting the
                // triangle edge indices of degenerate triangles to
                // {INVALID_INDEX, INVALID_INDEX, INVALID_INDEX}. (Most likely,
                // the idea is that we can't guarantee perceptual
                // watertightness here). This means we can't propagate data
                // along edges of degenerate triangles. However, we've still
                // attempted to compress the data contained within a degenerate
                // triangle, so maybe we could skip compressing such data
                // altogether? TODO: Followup.
                if(srcMeshEdgeIdx == INVALID_INDEX)
                {
                    continue;
                }

                // Loop over the base triangles sharing this edge
                const Range32 edgeFaceIndicesRange = arrayGetV<Range32>(topology.edgeTriangleRanges, srcMeshEdgeIdx);
                for(uint32_t f = 0; f < edgeFaceIndicesRange.count; f++)
                {
                    // If this base triangle is not the one we started from..
                    const auto dstTopoFaceIdx =
                        arrayGetV<uint32_t>(topology.edgeTriangleConnections, edgeFaceIndicesRange.first + f);
                    if(dstTopoFaceIdx != srcTopoFaceIdx)
                    {
                        // Find the destination edge that matches the source edge
                        uint32_t   dstLocalEdgeIdx;
                        const auto dstGlobalEdges = arrayGetV<Vector_uint32_3>(topology.triangleEdges, dstTopoFaceIdx);
                        if(srcMeshEdgeIdx == dstGlobalEdges.x)
                            dstLocalEdgeIdx = 0u;
                        else if(srcMeshEdgeIdx == dstGlobalEdges.y)
                            dstLocalEdgeIdx = 1u;
                        else
                            dstLocalEdgeIdx = 2u;


                        bool areEdgesConcordant = false;
                        // In the non-manifold case the src and dst outer edges might be concordant (i.e. endpoints with identical ordering),
                        // in which case we need to inform processOuterEdge() to NOT swap the source edge endpoints.
                        if(topology.isNonManifold)
                        {
                            const auto& srcFaceVtxs = arrayGetV<Vector_uint32_3>(topology.triangleVertices, srcTopoFaceIdx);
                            const auto& dstFaceVtxs = arrayGetV<Vector_uint32_3>(topology.triangleVertices, dstTopoFaceIdx);
                            areEdgesConcordant = srcFaceVtxs[srcEdgeInfo.baseEdgeIdx] == dstFaceVtxs[dstLocalEdgeIdx]
                                                 && srcFaceVtxs[(1 + srcEdgeInfo.baseEdgeIdx) % 3]
                                                        == dstFaceVtxs[(1 + dstLocalEdgeIdx) % 3];
                        }

                        // Convert back from topology indices to input indices
                        const auto dstBaseTriangleIdx = meshGetTriangleMapping(meshTriangleMappings, dstTopoFaceIdx);
                        if(!m_micromapTriangles[dstBaseTriangleIdx]->processOuterEdge(mode, srcEdgeInfo, srcEdgeData,
                                                                                      dstLocalEdgeIdx, areEdgesConcordant))
                        {
                            return false;
                        }
                    }
                }
            }
        }
    }

    return true;
}


}  // namespace dispenc
}  // namespace micromesh