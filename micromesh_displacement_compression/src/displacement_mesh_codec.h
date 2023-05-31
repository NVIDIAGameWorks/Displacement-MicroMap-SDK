/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Internal interface for the mesh encoder and decoder, which encodes meshes
// while preserving watertightness
// For encoding, we have a MeshEncoder class which stores all encoding data,
// which we then retrieve later. We assume that all API preconditions are
// already checked in displacement_compression.cpp. Unlike the block encoder,
// we have format conversion logic inside MeshEncoder, since it's usually not
// called repeatedly.
// This can also expose functions for unit testing.

#pragma once

#ifndef NDEBUG
// Enable this to store data in Triangles that's useful for debugging.
#ifndef ENABLE_SUBTRIANGLE_IDX
#define ENABLE_SUBTRIANGLE_IDX
#endif
#endif

#include <memory>
#include <array>
#include <micromesh_internal/micromesh_containers.h>
#include <micromesh/micromesh_displacement_compression.h>
#include <micromesh/micromesh_utils.h>

namespace micromesh
{
namespace dispenc
{
// Use this class to encode an entire mesh. This can ensure watertigntness
// across edges. This class essentially stores all output values and all
// temporary data the encoder needs to modify during operations.
// A MeshEncoder must not be accessed by multiple threads at once.
class MeshEncoder
{
  public:
    MeshEncoder(){};

    // Implements the main encoding algorithm! Stores compressed data and
    // intermediate structures inside m_baseTriangles.
    Result batchEncode(OpContext ctx, const OpCompressDisplacement_settings* settings, const OpCompressDisplacement_input* inputUncompressed);

    // Sets the size fields of the output to the values needed to store this
    // MeshEncoder's compressed data, following the API rules for what to set.
    void fillEncodedSizes(MicromapCompressed& outputCompressed) const;

    // Once the MicromapCompressed's data has been allocated, call this to
    // store it!
    void writeCompressedData(OpContext ctx, OpCompressDisplacement_output& outputCompressed) const;

    // Returns the total peak signal-to-noise ratio for the encoded mesh.
    // This is 10 * log_10((maximum error per vertex)^2 / (mean square error)^2),
    // where we average the error over all microvertices (not microtriangles).
    // If there were no errors, returns +INFINITY.
    float computePSNR() const;

  private:
    // Delete copy constructor and assignment operators
    MeshEncoder(const MeshEncoder&) = delete;
    MeshEncoder& operator=(MeshEncoder other) = delete;

    //////////////////////////////////////////////////////////////////////////
    // Transient and final state of encoding process

    enum class EdgeMode
    {
        // Sets the reference edges of unencoded micromap triangles based on
        // the decoded edges of `triangles`.
        eSetReference = 0,
        // Verifies that reference edges match between adjacent triangles.
        ePreEncodeValidation,
        // Verifies that decoded edges match between adjacent triangles.
        ePostEncodeValidation,
    };

    // A Triangle represents a micromap triangle, or a sub-triangle of a Triangle.
    // That is, a Triangle can be a quadtree. The leaf nodes are Triangles
    // that are equivalent to one compressed block.
    // Each Triangle stores both uncompressed and compressed data in
    // an intermediate format.
    class Triangle
    {
      public:
        // Initial micromap triangle constructor
        Triangle(const Micromap& micromap, uint64_t micromapTriangleIdx, std::array<float, 3> vtxImportance);
        // Split sub-triangle constructor, used by encoding process
        Triangle(uint8_t subTriangleIdx, const Triangle& parent);

        // Delete copy constructor and assignment operators
        Triangle(const Triangle&) = delete;
        Triangle& operator=(Triangle other) = delete;

        void bin(container::vector<container::vector<Triangle*>>& bins);
        bool encode(const BlockFormatDispC1 fmt, bool isLastFormat, float minPSNR);

        void computeL2Err(double& l2Err, double& numElems) const;
        // Returns 10 * log_10((maximum possible error)^2/(mean square error)).
        float computePSNR() const;

        inline uint32_t getSubdivLevel() const { return m_subdivLevel; }
        inline uint32_t getBaseSubdivLevel() const { return m_baseSubdivLevel; }
        inline uint32_t getNumSegments() const { return subdivLevelGetSegmentCount(m_subdivLevel); }
        inline uint32_t getBaseNumSegments() const { return subdivLevelGetSegmentCount(m_baseSubdivLevel); }
        inline uint64_t getBaseTriangleIdx() const { return m_baseTriangleIdx; }
        inline bool     getIsEncoded() const { return m_isEncoded; }
        Vector_uint16_2 getCombinedDisplacementBounds() const;

        inline bool& postEncodingEdgeCheck(uint32_t i) { return m_postEncodingEdgeCheck[i]; }

        struct EdgeInfo
        {
            uint32_t numBaseSegments{};          // number of segments of the associated base triangle
            std::array<BaryUV_uint16, 2> barys;  // edge endpoints barycentric coordinates in the space of the base triangle
            int32_t outerEdgeStart{};     // [Only valid for outer edges] parametric starting point of the outer edge
            int32_t outerEdgeStop{};      // [Only valid for outer edges] parametric stopping point of the outer edge
            uint8_t baseEdgeIdx{};        // [Only valid for outer edges] edge index (i.e. 0 -> u, 1 -> v, 2 -> w)
            bool    isOuterEdge = false;  // is this edge (possibly) shared with another base triangle?
        };

        const EdgeInfo& getEdgeInfo(uint32_t localEdgeIdx) const { return m_edges[localEdgeIdx]; }
        void            getLocalEdgeBarys(uint32_t localEdgeIdx, BaryUV_uint16& a, BaryUV_uint16& b) const;
        template <typename T>
        void getEdgeData(const T* data, uint32_t edgeIdx, container::vector<T>& edge) const;
        void getReferenceEdge(uint32_t localEdgeIdx, container::vector<uint16_t>& edge) const;
        void getDecodedEdge(uint32_t localEdgeIdx, container::vector<uint16_t>& edge) const;

        bool processEdge(EdgeMode mode, const EdgeInfo& srcEdgeInfo, const container::vector<uint16_t>& srcEdgeData);
        bool processOuterEdge(EdgeMode                           mode,
                              const EdgeInfo&                    srcEdgeInfo,
                              const container::vector<uint16_t>& srcEdgeData,
                              uint32_t                           dstLocalEdgeIdx,
                              bool                               areEdgesConcordant);
        bool areLowerResNeighborsEncoded(uint32_t dstLocalEdgeIdx, uint32_t numBaseSegments);

        // Only use this if this is an encoded base triangle.
        void getEncodedFormatCountInfo(BlockFormatDispC1& fmt, uint64_t& numLeafTriangles) const;
        void appendLeafTriangles(container::vector<const Triangle*>& triangles) const;
        void appendEncodedTriangles(container::vector<const Triangle*>& triangles) const;
        void clearEncoded();

        // Writes a mip of decoded data from all child triangles to a mip in
        // R11_unorm format with a given layout. This works recursively;
        // output is the address of the 1st byte in the output for the
        // base triangle.
        void writeMip(void* output, uint16_t mipSubdivLevel, const MicromapLayout& layout) const;

        // Writes a leaf triangle's packed, encoded data to a location in memory.
        void writePackedBlock(void* packedData) const;

      private:
        ///////////////////////////////////////////////////////////////////////////
        // Local subdivision and orientation within top-level triangle

        uint64_t m_baseTriangleIdx = 0u;  // Index of triangle in Micromap input and output.
        uint8_t  m_baseSubdivLevel = 0u;
        uint8_t  m_subdivLevel     = 0u;
#ifdef ENABLE_SUBTRIANGLE_IDX
        uint8_t m_subTriangleIdx = 0u;  //  Required for debugging. Please don't remove.
#endif

        // Does this triangle use a reversed winding compared to regular?
        bool m_isWindingFlipped = false;

        // Per-vertex importance at each of the corners of the triangle.
        // Note that subtriangles get their importances interpolated from this!
        std::array<float, 3> m_vtxImportance = {1.0f, 1.0f, 1.0f};

        // Barycentric coordinates of the triangle vertices with respect to the
        // top-level triangle, ordered as u, v, and w vertices (i.e. these
        // coordinates are invariant with respect to "rotations" due to the
        // bird curve)
        std::array<BaryUV_uint16, 3> m_vertices;

        std::array<EdgeInfo, 3> m_edges;

        // Split (quadtree) hierarchy
        std::array<std::unique_ptr<Triangle>, 4> m_children = {nullptr, nullptr, nullptr, nullptr};

        ///////////////////////////////////////////////////////////////////////////
        // Encoding related

        // We use one allocation for both m_reference and m_intermediate's buffers.
        std::unique_ptr<uint16_t[]> m_allocated;
        Intermediate                m_intermediate;
        // reference displacement map (mutable reference/target for encoding)
        uint16_t* m_reference;
        // Indexes into m_reference and m_intermediate's blocks
        uint32_t addr(uint32_t u, uint32_t v) const { return umajorUVtoLinear(u, v, m_subdivLevel); }
        uint32_t addr(uint32_t u, uint32_t v, uint32_t subdiv) const { return umajorUVtoLinear(u, v, subdiv); }

        BlockFormatDispC1 m_encodedBlockFormat = BlockFormatDispC1::eInvalid;
        Vector_uint16_2   m_displacementBounds = {0xFFFF, 0};

        // For each edge, set this to `true` to check that the compressed edge
        // exactly matches the reference edge. This is used both for edge
        // propagation and for `requireLosslessMeshEdges`.
        std::array<bool, 3> m_postEncodingEdgeCheck = {false, false, false};
        // Whether the Triangle and all its children have been successfully encoded.
        bool m_isEncoded = false;

        ///////////////////////////////////////////////////////////////////////////

        void allocateMem();
        void deallocateMem();

        inline bool isLeafNode() const { return m_children[0] == nullptr; }
        inline bool doEndpointsMatch(BaryUV_uint16 a_start, BaryUV_uint16 b_start, BaryUV_uint16 a_end, BaryUV_uint16 b_end) const;

        EdgeInfo computeEdgeInfo(uint32_t localEdgeIdx) const;

        bool satisfiesCompressionControl(const BlockFormatDispC1 fmt, const float minPSNR) const;
    };

    // In the following two functions, triangleMeshMappings is an inverse for
    // meshTriangleMappings: triangleMeshMappings[micromap triangle index]
    // gives a mesh triangle index.

    // Checks whether values match up on the edges of adjacent faces. Prints
    // a message if there are mismatches. More complex than one might think
    // initially!
    bool validateEdges(OpContext                          ctx,
                       const MeshTopology&                topology,
                       const ArrayInfo_uint32&            meshTriangleMappings,
                       const container::vector<uint64_t>& triangleMeshMappings,
                       bool                               preEncodingValidation);

    // General function for processing the edges of triangles in `triangles`
    // and the edges of other triangles in the micromap. If mode is not
    // eSetReference and validation fails, prints a message.
    bool processEdges(OpContext                           ctx,
                      EdgeMode                            mode,
                      const MeshTopology&                 topology,
                      const ArrayInfo_uint32&             meshTriangleMappings,
                      const container::vector<uint64_t>&  triangleMeshMappings,
                      container::vector<const Triangle*>& triangles,
                      uint32_t                            numSegments = 0);

    // Previously known as m_baseTriangles.
    container::vector<std::unique_ptr<Triangle>> m_micromapTriangles;

    // Used for the inverse of the mesh-to-micromap function to indicate that
    // an element has no inverse.
    static constexpr uint64_t NO_MESH_TRIANGLE = ~0u;

    // Stored from the settings so we can reuse it in micromeshOpCompressDisplacementEnd().
    uint16_t mipIgnoredSubdivLevel = 0xFFFF;
};

}  // namespace dispenc
}  // namespace micromesh