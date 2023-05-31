//
// Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#pragma once

#include "micromesh_types.h"
#include "micromesh_api.h"
#include <cassert>

namespace micromesh
{
//////////////////////////////////////////////

// All major operations are done using a context.
// A context can work only at one operation step at a time.
// If an operation requires multiple steps to complete, these must be called
// in-order using the same context that started the operation.
// Such a sequence can be aborted with micromeshOpContextAbort.
// Contexts cannot be used concurrently unless specifically mentioned.

enum class OpContextType : uint32_t
{
    // this context will itself distribute work on multiple threads
    // std::thread is used as default implementation.
    eImmediateAutomaticThreading,

    // this context records the work for an operation step and
    // allows the user to distribute it across multiple threads manually.
    // micromeshOpContextJoin must be called to make forward progress
    // and is the only function allowed to be called concurrently.
    //   Not yet implemented
    //eDeferredManualJoin,
};

struct OpConfig
{
    OpContextType contextType = OpContextType::eImmediateAutomaticThreading;
    // if threadCount > 1 operations may use std::thread based multi-threading to speed things up
    uint32_t threadCount = 1;
};

// sets OpContextType::eImmediateAutomaticThreading and
// threadCount to std::thread::hardware_concurrency()
MICROMESH_API OpConfig MICROMESH_CALL micromeshGetDefaultOpConfig();

MICROMESH_API Result MICROMESH_CALL micromeshCreateOpContext(const OpConfig*            config,
                                                             OpContext*                 pContext,
                                                             const MessageCallbackInfo* messageCallback);
// legal to be called with nullptr
MICROMESH_API void MICROMESH_CALL micromeshDestroyOpContext(OpContext context);


// aborts the processing if an operation with multiple steps was started.
MICROMESH_API void MICROMESH_CALL micromeshOpContextAbort(OpContext context);

// only valid for OpContextType::eDeferredManualJoin
// MICROMESH_API JoinResult MICROMESH_CALL micromeshOpContextJoin(OpContext* context, uint32_t threadIdx);

MICROMESH_API OpConfig MICROMESH_CALL micromeshOpContextGetConfig(OpContext context);

MICROMESH_API void MICROMESH_CALL micromeshOpContextSetMessageCallback(OpContext context, const MessageCallbackInfo info);
MICROMESH_API MessageCallbackInfo MICROMESH_CALL micromeshOpContextGetMessageCallback(OpContext context);

//////////////////////////////////////////////

// ScopedOpContext is a utility helper
// create temporary contexts
struct ScopedOpContext
{
  public:
    ScopedOpContext(micromesh::OpConfig            config       = micromeshGetDefaultOpConfig(),
                    micromesh::MessageCallbackInfo callbackInfo = micromesh::MessageCallbackInfo())
    {
        [[maybe_unused]] micromesh::Result result = micromesh::micromeshCreateOpContext(&config, &m_context, &callbackInfo);
        assert(result == micromesh::Result::eSuccess);  // Errors have already been printed with MLOGE()
    }

    ScopedOpContext(uint32_t numThreads, micromesh::MessageCallbackInfo callbackInfo = micromesh::MessageCallbackInfo())
        : ScopedOpContext({micromesh::OpContextType::eImmediateAutomaticThreading, numThreads}, callbackInfo)
    {
    }

    ~ScopedOpContext() { micromeshDestroyOpContext(m_context); }

    operator micromesh::OpContext() const { return m_context; }

    // Disable copying
    ScopedOpContext(const ScopedOpContext& other)            = delete;
    ScopedOpContext& operator=(const ScopedOpContext& other) = delete;

  private:
    OpContext m_context = nullptr;
};

//////////////////////////////////////////////
// leverage the context's work distribution system, for any operation

typedef void (*PFN_genericSingleWorkload)(uint64_t itemIndex, uint32_t threadIndex, void* userData);
typedef void (*PFN_genericRangeWorkload)(uint64_t itemFirst, uint64_t itemLast, uint32_t threadIndex, void* userData);

struct OpDistributeWork_input
{
    // only one of them will be executed, will pick first non-null
    PFN_genericSingleWorkload pfnGenericSingleWorkload = nullptr;
    PFN_genericRangeWorkload  pfnGenericRangeWorkload  = nullptr;
    void*                     userData                 = nullptr;

    // the distributor partitions the totalWorkLoad in batches
    // of this size. One batch is kept on a single thread.
    // for a given batch:
    // - pfnGenericSingleWorkload is called up to `batchSize` many times on that thread
    // - pfnGenericRangeWorkload is called once per batch with up to `batchSize` elements
    uint64_t batchSize = 1024;
};

MICROMESH_API Result MICROMESH_CALL micromeshOpDistributeWork(OpContext context, const OpDistributeWork_input* input, uint64_t totalWorkload);

//////////////////////////////////////////////

struct OpBuildMeshTopologyIndices_input
{
    ArrayInfo_float_3  meshVertexPositions;
    ArrayInfo_uint32_3 meshTriangleVertices;

    // optional, taken into account to determine unique
    // vertices
    ArrayInfo_float_3 meshVertexDirections;
};

struct OpBuildMeshTopologyIndices_output
{
    // uses a subset of vertex indices for matching
    // meshVertexPositions (and meshVertexDirections)
    ArrayInfo_uint32_3 meshTopologyTriangleVertices;
};

MICROMESH_API Result MICROMESH_CALL micromeshOpBuildMeshTopologyIndices(OpContext                               ctx,
                                                                        const OpBuildMeshTopologyIndices_input* input,
                                                                        OpBuildMeshTopologyIndices_output*      output);


//////////////////////////////////////////////

// begins computations and returns partial results as well as rest of output allocation sizes
// content of input must stay valid for "end" operation
// output properties:
//   must be provided with appropriate content
//    - triangleVertices;
//   must be allocated and sized based on number of input triangles and vertices
//    - triangleEdges;
//    - vertexEdgeRanges;
//    - vertexTriangleRanges;
//   array count will be written by this operation
//    - edgeVertices
//    - edgeTriangleRanges
//    - edgeTriangleConnections
//    - vertexTriangleConnections
//    - vertexEdgeConnections
//   the number of triangles must be less than 2^32.
MICROMESH_API Result MICROMESH_CALL micromeshOpBuildMeshTopologyBegin(OpContext ctx, MeshTopology* output);

// output must contain array pointers for those computed in preprocess
MICROMESH_API Result MICROMESH_CALL micromeshOpBuildMeshTopologyEnd(OpContext ctx, MeshTopology* output);

//////////////////////////////////////////////

struct OpGrowTriangleSelection_input
{
    const MeshTopology* topology = nullptr;

    // either use a range of triangles to seed
    uint32_t triangleFirst = 0;
    uint32_t triangleCount = 0;

    // or a list of indices (used if valid / non-empty)
    ArrayInfo_uint32 triangleSelection;

    // output triangle Selection is sorted by ascending triangle indices
    bool sortedOutput = true;
};

struct OpGrowTriangleSelection_output
{
    // the reserved count must match topology.triangleVertices.count
    ArrayInfo_uint32 triangleSelection;
    // the actual written selection count
    uint64_t triangleSelectionCount = 0;
};

// adds the +1 ring connectivity to the input triangles
MICROMESH_API Result MICROMESH_CALL micromeshOpGrowTriangleSelection(OpContext                            ctx,
                                                                     const OpGrowTriangleSelection_input* input,
                                                                     OpGrowTriangleSelection_output*      output);

//////////////////////////////////////////////
// Sampling

struct VertexSampleInfo
{
    // input mesh
    uint32_t meshTriangleIndex;
    // output in micromap
    uint32_t micromapTriangleIndex;

    uint32_t      subdivLevel;
    BaryUV_uint16 vertexUV;
    BaryWUV_float vertexWUVfloat;
    // if texcoords are provided
    Vector_float_2 vertexTexCoord;
};

// function that should write into the target microvertex
// must be thread-safe
typedef void (*PFN_sampleVertex)(const VertexSampleInfo* sampleInfo,
                                 uint32_t                valueIdx,
                                 void*                   valueData,
                                 uint32_t                threadIndex,
                                 void*                   beginTriangleResult,
                                 void*                   userData);

// callbacks called prior/after the main action callback, triggered once per mesh triangle
// must be thread-safe
typedef void* (*PFN_beginTriangle)(uint32_t meshTriangleIndex, uint32_t micromapTriangleIndex, uint32_t threadIndex, void* userData);
typedef void (*PFN_endTriangle)(uint32_t meshTriangleIndex,
                                uint32_t micromapTriangleIndex,
                                uint32_t threadIndex,
                                void*    beginTriangleResult,
                                void*    userData);

struct OpSampleFromMesh_input
{
    uint32_t meshTriangleCount = 0;

    // per-mesh-triangle mapping index to a micromesh triangle
    // optional
    // if a micromapTriangle is used multiple times, the
    // sample function will be evoked multiple times, and this can
    // be concurrent
    ArrayInfo_uint32 meshTriangleMappings;

    // optionally provides interpolated texcoords (e.g. for texture sampling)
    ArrayInfo_uint32_3 meshTriangleVertices;
    ArrayInfo_float_2  meshVertexTexcoords;

    void* userData = nullptr;

    PFN_sampleVertex pfnSampleVertex = nullptr;

    // optional if developer wants to do some per-triangle operations prior sampling
    PFN_beginTriangle pfnBeginTriangle = nullptr;
    PFN_endTriangle   pfnEndTriangle   = nullptr;
};

// output must have all information set, only values be written
// typically followed by `micromeshOpSanitizeEdgeValues`
MICROMESH_API Result MICROMESH_CALL micromeshOpSampleFromMesh(OpContext ctx, const OpSampleFromMesh_input* input, Micromap* output);

//////////////////////////////////////////////

struct OpSmoothMeshDirections_input
{
    ArrayInfo_float_3  meshVertexPositions;
    ArrayInfo_uint32_3 meshTriangleVertices;

    float triangleAreaWeight = 0;
};

struct OpSmoothMeshDirections_output
{
    ArrayInfo_float_3 meshVertexDirections;
};

MICROMESH_API Result MICROMESH_CALL micromeshOpSmoothMeshDirections(OpContext                           ctx,
                                                                    const OpSmoothMeshDirections_input* input,
                                                                    OpSmoothMeshDirections_output*      output);

//////////////////////////////////////////////
// Tessellation

// Appends vertex attribute into hash state.
// The data is taken in its binary state and there is no float tolerance and such.
// Developer must ensure attributes are added in same order for all vertices.
MICROMESH_API void MICROMESH_CALL micromeshVertexDedupAppendAttribute(VertexDedup dedupState, uint32_t dataSize, const void* data);

// computes unique vertex index for the state provided
MICROMESH_API uint32_t MICROMESH_CALL micromeshVertexDedupGetIndex(VertexDedup dedupState);

// OpTessellateMesh
// tessellates the input mesh, turning the microtriangles and microvertices into a full output mesh.

struct VertexGenerateInfo
{
    uint32_t      subdivLevel;
    BaryUV_uint16 vertexUV;
    BaryWUV_float vertexWUVfloat;

    // If no vertex deduplication is performed, return this value.
    // If vertex deduplication is performed, but not required
    // as vertex is not on a triangle edge and `dedupState` is
    // null, return this value.
    uint32_t nonDedupIndex;

    // input mesh
    uint32_t meshTriangleIndex;
    // input micromap (if provided)
    uint32_t micromapTriangleIndex;
};

// function that computes a new vertex.
// returns index of vertex to be used by triangle vertex index buffer.
// must be thread-safe
//
// This function computes the new vertex with per-microvertex interpolated attributes.
// The attributes are typically the result of barycentric interpolation of the input
// mesh triangle's vertices and the provided barycentric coordinate in `vertexWUVfloat`.
//
// The developer has two choices, which must be handled uniformly for the entire
// tessellation process:
//
//  useVertexDeduplication == false:
//      return the canonical `nonDedupIndex`, which guarantees to fill the worst-case
//      `vertexCount` returned by the "begin" function in a conical ordering.
//
//  useVertexDeduplication == true:
//      return a deduplicated vertex index (indices returned can have random ordering)
//      by using `micromeshVertexDedupAppendAttribute` and then using the return
//      value of `micromeshVertexDedupGetIndex` as vertex index.
//      The `dedupState` pointer can be null for vertices that are not on the edge
//      of the triangle, in that case use `nonDedupIndex`
//
// The vertex deduplicator works on binary data alone, so it is sensitive to the order
// of float operations. As the winding of adjacent triangle changes it will happen that
// barycentric sum  of the `vertexAttribute * barycentricWeight` pairs are summed in
// different order. The developer must take care and ensure stable results are provided
// to the dedup function. `meshReorderStableInterpolation` can help provide a stable
// ordering of vertex indices and barycentric weights.
typedef uint32_t (*PFN_generateVertex)(const VertexGenerateInfo* vertexInfo,
                                       VertexDedup               dedupState,
                                       uint32_t                  threadIndex,
                                       void*                     beginTriangleResult,
                                       void*                     userData);

struct OpTessellateMesh_input
{
    bool     useVertexDeduplication = true;
    uint32_t maxSubdivLevel         = 0;

    // either use per-mesh triangle subdiv levels
    ArrayInfo_uint16 meshTriangleSubdivLevels;

    // or combiination of
    // per-mesh-triangle mapping index to a micromap triangle
    // and the micromap triangle
    ArrayInfo_uint32 meshTriangleMappings;
    ArrayInfo_uint16 micromapTriangleSubdivLevels;

    // optional if applicable
    ArrayInfo_uint8 meshTrianglePrimitiveFlags;

    void* userData = nullptr;

    PFN_generateVertex pfnGenerateVertex = nullptr;

    // optional, can aid decoding local data for specific triangles
    PFN_beginTriangle pfnBeginTriangle = nullptr;
    PFN_endTriangle   pfnEndTriangle   = nullptr;
};

struct OpTessellateMesh_output
{
    uint32_t           vertexCount = 0;
    ArrayInfo_uint32_3 meshTriangleVertices;
};

// The begin function computes the count for meshTriangleVertices for the provided input and
// provides an upper vertexCount value. The actual value used can be lower, as result of
// vertex deduplication performed in the end function.
// the number of output vertices and triangles must each be less than 2^32.
// writes:
//     meshTriangleVertices.count : accurate output size
//     vertexCount               : worst-case output size
MICROMESH_API Result MICROMESH_CALL micromeshOpTessellateMeshBegin(OpContext                     ctx,
                                                                   const OpTessellateMesh_input* input,
                                                                   OpTessellateMesh_output*      output);

// The end function tessellates the input mesh, turning the microtriangles and microvertices into a full output mesh.
// Vertex creation must be handled by the developer. The post-deduplication vertexCount is written.
// writes:
//      meshTriangleVertices.data : final triangle indices
//      vertexCount              : final vertex count
MICROMESH_API Result MICROMESH_CALL micromeshOpTessellateMeshEnd(OpContext                     ctx,
                                                                 const OpTessellateMesh_input* input,
                                                                 OpTessellateMesh_output*      output);
//////////////////////////////////////////////

struct OpAdaptiveSubdivision_input
{
    // highest subdivision level possible
    uint32_t maxSubdivLevel = 0;

    // base decision on area otherwise longest edge
    bool useArea = false;

    // if true we will base relative to largest
    //      triangle value multiplied by referenceWeight.
    // if false then we will use triangle value as is to
    //      derive subdiv level so that
    //      for area:
    //          area units count ~= triangle count
    //                sqrt(area) ~= triangle subdiv level
    //      for edge
    //          edge length ~= triangle subdiv level
    //
    //      always clamped by maxSubdivLevel.
    bool useRelativeValues = false;

    bool  useRelativeMaxValueOverride = false;
    float relativeMaxValueOverride    = 0;

    // if true we don't compute subdivlevels in this pass
    // but just return the `relativeMaxValue`
    bool onlyComputeRelativeMaxValue = false;

    // for techniques deriving subdivlevel relative
    // to the maximum triangle value computed, use
    // this value to influence which values can achieve
    // maxSubdivLevel (smaller number and more triangles can achieve
    // peak, higher number then less).
    //
    // relative_factor = triangle_value / (maximum_triangle_value * referenceWeight)
    float relativeWeight = 1.0f;

    // Manual adjustment of the output subdivision values. Only applied when
    // values are not relative, i.e. !useRelativeValues. This is typically
    // negative and simply added to the result.
    int32_t subdivLevelBias = 0;

    // triangle vertex indices
    ArrayInfo_uint32_3 meshTriangleVertices;

    // either provide meshVertexTexcoords
    // or meshVertexPositions

    // if provided we will base subdivlevel on UV area
    ArrayInfo_float_2 meshVertexTexcoords;
    // texcoords are multiplied by this factor prior calculating area
    Vector_float_2 texResolution = {1, 1};

    // otherwise uses longest edge
    // longest edge is multiplied by referenceWeight
    ArrayInfo_float_3 meshVertexPositions;
    // positions are multiplied by this scale factor
    Vector_float_3 positionScale = {1, 1, 1};
};

struct OpAdaptiveSubdivision_output
{
    ArrayInfo_uint16 meshTriangleSubdivLevels;
    uint32_t         maxSubdivLevel = 0;
    uint32_t         minSubdivLevel = 0;

    // updated if `useRelativeValues` is used
    float relativeMaxValue = 0;
};

// seeds subdivision levels per triangle based mesh properties
// requires micromeshOpSanitizeSubdivLevels to be run afterwards
MICROMESH_API Result MICROMESH_CALL micromeshOpAdaptiveSubdivision(OpContext                          ctx,
                                                                   const OpAdaptiveSubdivision_input* input,
                                                                   OpAdaptiveSubdivision_output*      output);

//////////////////////////////////////////////

struct OpSanitizeSubdivLevels_input
{
    const MeshTopology* meshTopo       = nullptr;
    uint32_t            maxSubdivLevel = 0;
    ArrayInfo_uint16    meshTriangleSubdivLevels;
    ArrayInfo_uint32    meshTriangleMappings;  // optional
};

struct OpSanitizeSubdivLevels_output
{
    // in-place modification possible
    ArrayInfo_uint16 meshTriangleSubdivLevels;
    uint32_t         minSubdivLevel = 0;
};

// Ensures subdivision levels between triangles of a shared edge only differ by up to one.
// Input and output meshTriangleSubdivLevels are allowed to match (in-place modification).
//
// If mapping is provided, also ensures all subdivisions levels of the mesh that point to
// same target will use a common value (highest subdiv of any of these triangles).
// After sanitization it could be that the mapping creates inconsistencies, like one mesh
// triangle is set to subdiv a and another one to subdiv b while both point to the same
// target mapping. This state will not be resolved and an error is thrown.
MICROMESH_API Result MICROMESH_CALL micromeshOpSanitizeSubdivLevels(OpContext                           ctx,
                                                                    const OpSanitizeSubdivLevels_input* input,
                                                                    OpSanitizeSubdivLevels_output*      output);

struct OpBuildPrimitiveFlags_input
{
    const MeshTopology* meshTopo = nullptr;
    ArrayInfo_uint16    meshTriangleSubdivLevels;
};

struct OpBuildPrimitiveFlags_output
{
    // in-place modification possible
    ArrayInfo_uint8 meshTrianglePrimitiveFlags;
};

// builds the primitive flags where each edge bit is set, if a neighboring mesh triangle of that
// edge has one less subdivision level.
// Throws error if there is a greater difference than one.
MICROMESH_API Result MICROMESH_CALL micromeshOpBuildPrimitiveFlags(OpContext                          ctx,
                                                                   const OpBuildPrimitiveFlags_input* input,
                                                                   OpBuildPrimitiveFlags_output*      output);

//////////////////////////////////////////////

// Micromap format must be uncompressed float and per-micro-vertex
// only its values will be modified.
// Does average the values for every shared micro-vertex.


struct OpSanitizeEdgeValues_input
{
    const MeshTopology* meshTopology = nullptr;
    ArrayInfo_uint32    meshTriangleMappings;  // optional
};

MICROMESH_API Result MICROMESH_CALL micromeshOpSanitizeEdgeValues(OpContext ctx, const OpSanitizeEdgeValues_input* input, Micromap* modified);

//////////////////////////////////////////////

// returns number of micro-vertices that are topological equivalent to this micro vertex.
// Used for finding matching micro-vertices along shared edges and triangles to handle
// sanitization for watertightness. Will return zero for inner micro-vertices.
// If provided, fills `outputVertices` up to `outputReserveCount`.
// You can use `max(meshTopo.maxEdgeTriangleValence, meshTopo.maxVertexTriangleValence) as
// guaranteed upper bound for `outputReserveCount`.
// The output vertexUV values will be adjusted for their triangle's winding and subdivision level.
// The list will have a stable ordering and will include the queried point as well.
// `meshTriangleMappings` can be null, then `triangleSubdivLevels` is equivalent
// to `meshTriangleSubdivLevels`, otherwise we will use the mapping indirection prior access it.
// thread-safe utility function
//
// `micromeshOpSanitizeEdgeValues` is preferred for sanitizing entire micromaps.
MICROMESH_API uint32_t MICROMESH_CALL micromeshMeshTopologyGetVertexSanitizationList(const MeshTopology* meshTopo,
                                                                                     const ArrayInfo_uint16* triangleSubdivLevels,
                                                                                     const ArrayInfo_uint32* meshTriangleMappings,
                                                                                     MicroVertexInfo queryVertex,
                                                                                     uint32_t        outputReserveCount,
                                                                                     MicroVertexInfo* outputVertices);

//////////////////////////////////////////////

struct OpComputeTriangleMinMaxs_output
{
    // both arrays can be be left empty
    // then only globals are computed
    ArrayInfo triangleMins;
    ArrayInfo triangleMaxs;

    MicromapValue globalMin;
    MicromapValue globalMax;
};

MICROMESH_API Result MICROMESH_CALL micromeshOpComputeTriangleMinMaxs(OpContext                        ctx,
                                                                      const Micromap*                  input,
                                                                      OpComputeTriangleMinMaxs_output* output);

//////////////////////////////////////////////

struct OpFloatToQuantized_input
{
    // only 32bit floats supported
    const Micromap* floatMicromap = nullptr;

    // scale less than this is treated as zero
    float scaleThreshold = 0.0000001f;

    // required to set up output MicromapValueFloatExpansion
    MicromapValue globalMin;
    MicromapValue globalMax;

    // if outputFormat channelType is SFLOAT
    // then we quantize to [-1,1] if this state is false
    // otherwise [0,1]
    bool outputUnsignedSfloat{};
};

// converts from float to UNORM [0,1], SNORM [-1,1], or SFLOAT [-1,1] or [0,1] range
// sets up output MicromapValueFloatExpansion accordingly.
// Output's arrays must be properly sized. Output values.format must
// be trivial uncompressed SLFOAT,UNORM or SNORM.
// all other properties will be written using values from input, unless array pointers match.
MICROMESH_API Result MICROMESH_CALL micromeshOpFloatToQuantized(OpContext ctx, const OpFloatToQuantized_input* input, Micromap* output);

MICROMESH_API Result MICROMESH_CALL micromeshOpFloatToQuantizedPacked(OpContext                       ctx,
                                                                      const OpFloatToQuantized_input* input,
                                                                      MicromapPacked*                 output);

// converts from float to UNORM [0,1], SNORM [-1,1], or SFLOAT [-1,1] or [0,1] range
// sets up output MicromapValueFloatExpansion accordingly.
// Output's arrays must be properly sized. Output values.format must
// be trivial uncompressed SLFOAT,UNORM or SNORM.
// settings.floatMicromap is ignored
// thread-safe, single-threaded, meant for single triangle processing
MICROMESH_API Result MICROMESH_CALL micromeshFloatToQuantizedValues(const OpFloatToQuantized_input* settings,
                                                                    const ArrayInfo*                floatInput,
                                                                    ArrayInfo*                      output,
                                                                    MicromapValueFloatExpansion*    outputExpansion,
                                                                    const MessageCallbackInfo*      callbacks);

//////////////////////////////////////////////

struct OpQuantizedToFloat_input
{
    const Micromap* quantizedMicromap = nullptr;

    // if true output float will be in [0,1] or [-1,1] range and
    // expansion will not be removed. Otherwise the output
    // values will have scale and bias applied and
    // output's expansion will be reset to default (bias 0 and scale 1).
    bool outputKeepFloatExpansion;
};

// Converts from UNORM [0,1], SNORM [-1,1], or SFLOAT [-1,1] or [0,1] range
// to SFLOAT. Output values can have the float expansion applied.
// Output's arrays must be properly setup. Output values.format must be trivial uncompressed SFLOAT.
// All other properties will be written using values from input, unless array pointers match.
// only 32bit floats supported
MICROMESH_API Result MICROMESH_CALL micromeshOpQuantizedToFloat(OpContext                       ctx,
                                                                const OpQuantizedToFloat_input* input,
                                                                // the output's arrays must be properly sized
                                                                // their contents will be filled
                                                                Micromap* output);

// in-place value conversion from quantized to float then float to quantized
// keeps expansion as is, only works if formats have same / compatible channel type
// all other properties will be written using values from input, unless array pointers match.
MICROMESH_API Result MICROMESH_CALL micromeshOpQuantizedToQuantized(OpContext ctx,
                                                                    Micromap* input,
                                                                    // the output's arrays must be properly sized
                                                                    // their contents will be filled
                                                                    Micromap* output);


// Converts from UNORM [0,1], SNORM [-1,1], or SFLOAT [-1,1] or [0,1] range
// to SFLOAT. Output values can have the float expansion applied.
// Output's arrays must be properly setup. Output values.format must be trivial uncompressed SFLOAT.
// All other properties will be written using values from input.
// only 32bit floats supported
// thread-safe, single-threaded, meant for single triangle processing
MICROMESH_API Result MICROMESH_CALL micromeshQuantizedToFloatValues(bool             outputKeepFloatExpansion,
                                                                    const ArrayInfo* quantizedInput,
                                                                    const MicromapValueFloatExpansion* inputExpansion,
                                                                    // the output's arrays must be properly sized
                                                                    // their contents will be filled
                                                                    ArrayInfo*                   output,
                                                                    MicromapValueFloatExpansion* outputExpansion,
                                                                    const MessageCallbackInfo*   callbacks);

struct OpQuantizedPackedToFloat_input
{
    const MicromapPacked* quantizedMicromap = nullptr;

    // if true output float will be in [0,1] or [-1,1] range and
    // expansion will not be removed. Otherwise the output
    // values will have scale and bias applied and
    // output's expansion will be reset to default (bias 0 and scale 1).
    bool outputKeepFloatExpansion;
};

// Converts from eR11_unorm_packed_align32 format to SFLOAT32.
MICROMESH_API Result MICROMESH_CALL micromeshOpQuantizedPackedToFloat(OpContext                             ctx,
                                                                      const OpQuantizedPackedToFloat_input* input,
                                                                      // the output's arrays must be properly sized
                                                                      // their contents will be filled
                                                                      Micromap* output);

// Like micromeshQuantizedToFloatValues(), this is a thread-safe,
// single-threaded function meant for single triangle processing.
MICROMESH_API Result MICROMESH_CALL micromeshQuantizedPackedToFloatValues(bool             outputKeepFloatExpansion,
                                                                          const ArrayInfo* quantizedInput,
                                                                          const MicromapValueFloatExpansion* inputExpansion,
                                                                          // the output's arrays must be properly sized
                                                                          // their contents will be filled
                                                                          ArrayInfo*                   output,
                                                                          MicromapValueFloatExpansion* outputExpansion,
                                                                          const MessageCallbackInfo*   callbacks);

//////////////////////////////////////////////

// changes output.values
// currently supported:
// - from eR11_unorm_pack16 to eR11_unorm_packed_align32
// - from eR16_unorm        to eR11_unorm_packed_align32
// use `micromeshMicromapPackedSetupValues` with active triangleValueBytesOffsets computation in advance on output
MICROMESH_API Result MICROMESH_CALL micromeshOpPack(OpContext ctx, const Micromap* input, MicromapPacked* output);

// changes output.values
// currently supported:
// - from eR11_unorm_packed_align32 to eR16_unorm
// - from eR11_unorm_packed_align32 to eR8_unorm
MICROMESH_API Result MICROMESH_CALL micromeshOpUnpack(OpContext ctx, const MicromapPacked* input, Micromap* output);

// changes output.values
// currently supported:
// - lowering bit precision for integer types
MICROMESH_API Result MICROMESH_CALL micromeshOpLowerBit(OpContext ctx, const Micromap* input, Micromap* output);

//////////////////////////////////////////////

// change layout of a micromap
// values are re-ordered in-place
MICROMESH_API Result MICROMESH_CALL micromeshOpChangeLayout(OpContext ctx, const MicromapLayout* newLayout, Micromap* modified);


//////////////////////////////////////////////

// swizzle values per-triangle
// values are re-ordered in-place

struct OpSwizzle_input
{
    // per-triangle see `TriangleSwizzle`
    ArrayInfo_uint8 triangleSwizzle;

    // values from above array are right-shifted
    // before interpreted (allows to pack swizzle
    // with edge decimation flags)
    uint32_t swizzleBitShift;
};

MICROMESH_API Result MICROMESH_CALL micromeshOpSwizzle(OpContext ctx, const struct OpSwizzle_input* input, Micromap* modified);

//////////////////////////////////////////////

MICROMESH_API uint32_t MICROMESH_CALL micromeshGetBlockFormatUsageReserveCount(const MicromapCompressed* compressed);

struct OpComputeBlockFormatUsages_input
{
    const MicromapCompressed* compressed = nullptr;

    // optional, if provided computes the instanced usages
    ArrayInfo_uint32 meshTriangleMappings;
};

struct OpComputeBlockFormatUsages_output
{
    // must be equal to micromeshGetBlockFormatUsageReserveCount(compressed);
    uint32_t reservedUsageCount = 0;
    // must be provided
    BlockFormatUsage* pUsages = nullptr;

    // will be written
    uint32_t usageCount = 0;
};

MICROMESH_API Result MICROMESH_CALL micromeshOpComputeBlockFormatUsages(OpContext                               ctx,
                                                                        const OpComputeBlockFormatUsages_input* input,
                                                                        OpComputeBlockFormatUsages_output*      output);

//////////////////////////////////////////////

}  // namespace micromesh
