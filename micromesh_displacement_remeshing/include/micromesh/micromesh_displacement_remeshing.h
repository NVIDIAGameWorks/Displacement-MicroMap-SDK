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
#include <micromesh/micromesh_gpu.h>

namespace micromesh
{
// Remesher operation mode
enum class RemeshingMode : uint32_t
{
    // Decimation using edge collapse
    eDecimate = 0
};

// Error values for the remesher
// Since remeshing is performed on GPU the error
// may result from previous remesher calls
enum class RemesherErrorState : uint32_t
{
    eRemesherErrorNone                 = 0x0,
    eRemesherErrorVertexHashNotFound   = 0x1,
    eRemesherErrorEdgeHashNotFound     = 0x2,
    eRemesherErrorDebug                = 0x3,
    eRemesherErrorOutOfEdgeStorage     = 0x5,
    eRemesherErrorNoTriangleFound      = 0x6,
    eRemesherErrorNoVertexHistoryFound = 0x7,
    eRemesherErrorInvalidConstantValue = 0x8
};

// Remesher operation settings
struct OpRemeshing_settings
{
    // Operation mode, decimation using edge collapse by default
    RemeshingMode mode = RemeshingMode::eDecimate;
    // Maximum error resulting from an edge collapse operation
    // This error is computed from the resulting edge length, vertex valence,
    // vertex importance and angle between adjacent triangles
    float errorThreshold = 100.f;

    // Weight applied to the vertex importance when computing the
    // error incurred by an edge collapse operation
    float vertexImportanceWeight = 200.f;
    // When collapsing an edge, offset the resulting vertex along
    // its geometric normal so it lies as close as possible to the original surface
    bool fitToOriginalSurface = true;

    // Stop decimation if the resulting triangles would cover more than
    // 4^clampDecimationLevel triangles or displacement map texels.
    // The default level 5 allows a decimated triangle to represent
    // up to 1024 original triangles or texels. If -1 clamping is ignored
    int32_t clampDecimationLevel = 5u;

    // Generate micromesh-related information, such as the subdivision
    // level per triangle and displacement bounds per vertex
    bool generateMicromeshInfo = true;

    // Optional resolution of a displacement map that could be applied to
    // the original mesh. This resolution is used to limit the decimation
    // level if needed (see above)
    Vector_uint32_2 dispMapResolution = {INVALID_INDEX, INVALID_INDEX};

    // Maximum number of triangles in the decimated mesh. If positive this value
    // supersedes errorThreshold. Note this value is only a hint.
    // Depending on the other parameters the decimation may not be able
    // to achieve the target count. Conversely, the resulting triangle count
    // may be lower than this maximum triangle count.
    int32_t maxTriangleCount = -1;

    // Maximum vertex valence resulting from an edge collapse. This limit
    // prevents the generation of large triangle fans whose elongated triangles
    // may create rendering issues
    int32_t maxVertexValence = 20;
    // Maximum importance of the vertices involved in an edge collapse operation
    // This can typically be used to enforce the preservation of some fine features
    float maxVertexImportance = 1.f;

    // Number of texture coordinates per vertex
    uint32_t texcoordCount = 0;
    // Texture coordinate index used for optional displacement mapping
    uint32_t texcoordIndex = INVALID_INDEX;

    // Additional scale to the direction bounds to guarantee they contain the surface.
    float directionBoundsFactor = 1.02f;
};

// Current state of the remeshed object
struct RemeshingCurrentState
{
    // Decimated triangle count
    uint32_t triangleCount;
    // Decimated vertex count
    uint32_t vertexCount;
    // Error state
    RemesherErrorState errorState;
    // Number of merge operations performed in the last iteration
    uint32_t mergeCount;
    // Debug values - for internal purposes only
    Vector_uint32_4 debug;
};


struct RemeshingVertexMergeInfo
{
    uint32_t vertexIndexA;
    uint32_t vertexIndexB;
    float    blendAtoB;

    /*
    To allow users their own vertex attribute storage
    this information is provided as recipe for the expected
    merging operations to be performed on the
    vertex buffers.
     
    Next to merging, we expect the user to provide updated
    hash & checksum values for the full set of vertex
    properties.
     
    if (vertexIndexA == INVALID_INDEX) return;
    
    hash     = 0
    checksum = 0
     
    // The meshVertexTexcoords buffer is expected to be updated here
    // as well. Only the meshVertexPositions buffer is managed by the remeshing
    // directly
    for every vertex attribute (texcoord, normals etc.)
    {
        attributeA = attribute[vertexIndexA];
        attributeB = attribute[vertexIndexB];
        merged = lerp(attributeA, attributeB, blentAtoB);
        write result back to one/both vertices
        attribute[vertexIndexA] = merged;
        if (mode == RemeshingMode::eDecimate) {
            attribute[vertexIndexB] = merged;
        }
     
        // using the provided hash algorithms
        hash     = appendHashFuncPrimary(hash, merged);
        checksum = appendHashFuncSecondary(checksum, merged);
    }
     
    // add position to hash (enough to read vertexIndexA only)
    // merging will have occured before
     
    pos      = positions[vertexIndexA];
    hash     = appendHashFuncPrimary(hash, pos);
    checksum = appendHashFuncSecondary(checksum, pos);
      
    // output hash
    
    vertexHashes[vertexIndexA] = (hash, checksum);
    if (mode == RemeshingMode::eDecimate) {
        vertexHashes[vertexIndexB] = (hash, checksum);
    }
     
    // optional if user intends to provide curvature information
    curvatureA = vertexCurvatures[vertexIndexA];
    curvatureB = vertexCurvatures[vertexIndexB];
    vertexCurvatures[vertexIndexA] = max(curvatureA, curvatureB);
    if (mode == RemeshingMode::eDecimate) {
        vertexCurvatures[vertexIndexB] = max(curvatureA, curvatureB);
    }
    */
};

// Performs the operations described in RemeshingVertexMergeInfo
// must be thread-safe, vertexMergeInfos is part of scratch data,
// the other arrays are provided within OpRemeshing_output.
//
// The thread manager inside OpContext will launch this
// function multiple times on one or more threads with a subset of
// the total work.
//
// Note: the developer is expected to apply custom strides for
// vertexAttributeHashes and vertexCurvatures if they chose to have
// such strides.
typedef void (*PFN_remeshingMergeVertexAttributes)(uint32_t                        vertexCount,
                                                   const RemeshingVertexMergeInfo* vertexMergeInfos,
                                                   Vector_uint32_2*                vertexHashes,
                                                   float*                          vertexCurvatures,
                                                   uint32_t                        threadIndex,
                                                   void*                           userData);

// Description of the input mesh
struct OpRemeshing_input
{
    // Input triangle count
    uint32_t meshTriangleCount = 0;
    // Input vertex count
    uint32_t meshVertexCount = 0;

    // User data, typically to store a pointer to the
    // mesh data on the app side
    void* userData = nullptr;

    // Callback function performing a vertex merge on the CPU. Not used for now.
    PFN_remeshingMergeVertexAttributes pfnMergeVertexAttributes = nullptr;
};

struct OpRemeshing_output
{
    // The mesh properties are modified during the remeshing process.
    // They must be seeded with the initial state up front.
    // The output counts after decimation may be less
    // and are provided with the `outputTriangleCount` and `outputVertexCount`.
    //
    // Additional custom vertex attributes are computed
    // by the developer in a separate pass using
    // the `pfnPassMergeAttributes` callback.
    // These also must start out with the initial state.


    // must match `OpRemeshing_input::meshVertexCount`
    ArrayInfo_float_3 meshVertexPositions;
    // must match `OpRemeshing_input::meshVertexCount`
    ArrayInfo_float_2 meshVertexTexcoords;
    // must match `OpRemeshing_input::meshVertexCount`
    // .x is primary hash, .y is checksum hash
    ArrayInfo_uint32_2 meshVertexHashes;

    // must match `OpRemeshing_input::meshTriangleCount`
    ArrayInfo_uint32_3 meshTriangleVertices;
    // optional
    // must match `OpRemeshing_input::meshTriangleCount`
    ArrayInfo_uint32 meshTriangleUserIDs;

    // optional
    // must match `OpRemeshing_input::meshVertexCount
    ArrayInfo_float meshVertexCurvatures;

    // following are only relevant for `RemeshingMode::eDecimate`
    //
    // target subdivision levels
    // each decimated triangle should use for displacement
    // must match `OpRemeshing_input::meshTriangleCount`
    // must be provided if `OpRemeshing_settings::generateDisplacementInfo`
    ArrayInfo_uint16 meshTriangleSubdivLevels;
    // if provided, filled with the target primitive flags
    // must match `OpRemeshing_input::meshTriangleCount`
    ArrayInfo_uint8 meshTrianglePrimitiveFlags;
    // if provided, fill in displacement directions
    // must match `OpRemeshing_input::meshVertexCount`
    ArrayInfo_float_3 meshVertexDirections;
    // if provided, fill in displacement direction bounds
    // you may manually apply direction bounds on the resulting
    // meshVertexPositions and meshVertexDirections
    // must match `OpRemeshing_input::meshVertexCount`
    ArrayInfo_float_2 meshVertexDirectionBounds;

    RemeshingCurrentState remeshingCurrentState;

    // must match `micromeshOpRemeshingGetScratchCount`
    uint32_t scratchCount;
    // computed during `micromeshOpRemeshingBegin`
    uint64_t* scratchSizes;
    // must be provided in `micromeshOpRemeshingEnd` with appropriate size
    void** scratchDatas;

    // written after end function
    uint32_t meshTriangleCount = 0;
    uint32_t meshVertexCount   = 0;
};

// Get the number of scratch buffers required by the remesher
MICROMESH_API uint32_t MICROMESH_CALL micromeshOpRemeshingGetScratchCount(OpContext ctx, const OpRemeshing_settings* settings);

// performs remeshing preparation.
// neither output arrays nor scratchData are used yet
// writes:
//     scratchSizes[]
MICROMESH_API Result MICROMESH_CALL micromeshOpRemeshingBegin(OpContext                   ctx,
                                                              const OpRemeshing_settings* settings,
                                                              const OpRemeshing_input*    input,
                                                              OpRemeshing_output*         output);

// performs remeshing operation
// must provide properly sized output arrays
// and scratchDatas
// writes:
//     fills various arrays
//     outputTriangleCount
//     outputVertexCount
MICROMESH_API Result MICROMESH_CALL micromeshOpRemeshingEnd(OpContext ctx, const OpRemeshing_input* input, OpRemeshing_output* output);


namespace gpu
{
typedef struct GpuRemeshing_s* GpuRemeshing;

static const uint32_t GpuRemeshingMajor = 0;
static const uint32_t GpuRemeshingMinor = 1;

struct GpuRemeshing_config
{
    // in-case dynamic linkage is used, want to ensure
    // everything is as expected
    uint32_t versionMajor = GpuRemeshingMajor;
    uint32_t versionMinor = GpuRemeshingMinor;

    // what version of the library's shader is requested
    gpu::ShaderCodeType codeType = eShaderCodeInvalid;

    // influences the configuration of the available pipelines

    uint32_t             supportedModeCount = 0;
    const RemeshingMode* supportedModes     = nullptr;

    bool useTriangleUserIDs = false;
    bool useVertexCurvature = false;
    bool useDebugBuffers    = false;
};


// Return the shader code types the API can provide
MICROMESH_API Result MICROMESH_CALL micromeshGpuRemeshingGetAvailableShaderCodeTypes(AvailableShaderCodeTypes* codeTypes);

// Initialize the internal remeshing data structures based on the input configuration
MICROMESH_API Result MICROMESH_CALL micromeshGpuRemeshingCreate(const GpuRemeshing_config* config,
                                                                GpuRemeshing*              pRemeshing,
                                                                const MessageCallbackInfo* messageCallback = nullptr);
// Destroy the internal remeshing data structures
MICROMESH_API Result MICROMESH_CALL micromeshGpuRemeshingDestroy(GpuRemeshing remeshing);

// The remeshing process involves both internal and user-side shaders, whose calls are scheduled from
// the remesher
enum GpuRemeshingUserPipeline : uint32_t
{
    // kernel must perform merging of all custom vertex attributes, which are not
    // visible to this api. See explanation of `RemeshingVertexMergeInfo`
    //
    // IN:              eGpuRemeshingMeshVertexMergeBuffer
    // MODIFY:          eGpuRemehsingMeshVertexHashBuffer
    //                  eGpuRemeshingMeshVertexCurvatureBuffer (optional)
    //                  all custom vertex attributes
    //
    // It does use GpuRemeshingUserMergeVerticesPushConstant
    eGpuRemeshingUserMergeVertices,
    eGpuRemeshingUserPipelineCount
};

// Write the necessary buffer and pipeline counts in the SetupInfo object
MICROMESH_API Result MICROMESH_CALL micromeshGpuRemeshingGetSetup(GpuRemeshing remeshing, gpu::SetupInfo* setup);

// Get the pipeline layout description for the pipelineLayoutIndex-th internal pipeline
MICROMESH_API Result MICROMESH_CALL micromeshGpuRemeshingGetPipelineLayout(GpuRemeshing             remeshing,
                                                                           uint32_t                 pipelineLayoutIndex,
                                                                           gpu::PipelineLayoutInfo* pipeline);

// Get the pipeline description for the pipelineIndex-th internal pipeline
MICROMESH_API Result MICROMESH_CALL micromeshGpuRemeshingGetPipeline(GpuRemeshing       remeshing,
                                                                     uint32_t           pipelineIndex,
                                                                     gpu::PipelineInfo* pipeline);
//MICROMESH_API Result MICROMESH_CALL micromeshGpuRemeshingGetUserPipeline(GpuRemeshing             remeshing,
//                                                                         GpuRemeshingUserPipeline userPipelineIndex,
//                                                                         gpu::UserPipelineInfo*   pipeline);

MICROMESH_API Result MICROMESH_CALL micromeshGpuRemeshingGetPersistent(GpuRemeshing remeshing, gpu::PersistentResourceInfo* persistent);

// task specific

enum GpuRemeshingResource : uint32_t
{
    // main modified buffers
    // -------------------------
    // must be pre-filled with appropriate data prior task begin
    // will be updated continuously during the process.
    // 4 x float per-vertex - 4th component is ignored
    eGpuRemeshingMeshVertexPositionsBuffer,
    // texcoordCount x 2 x float per-vertex
    eGpuRemeshingMeshVertexTexcoordsBuffer,
    // 2 x uint per-vertex
    eGpuRemeshingMeshVertexHashBuffer,
    // 3 x uint per-triangle
    eGpuRemeshingMeshTrianglesBuffer,
    // 1 x uint per-triangle (e.g. per-triangle component/material assignments etc.)
    // (optional `GpuRemeshing_config::useTriangleUserIDs`)
    eGpuRemeshingMeshTriangleUserIDsBuffer,

    // 1 x float16 per-vertex (optional `GpuRemeshing_config::useVertexCurvature`)
    eGpuRemeshingMeshVertexImportanceBuffer,

    // output buffers
    // -------------------------
    //
    // 1 x uint { uint16 subdivlevel, uint16 edgeflags} per-triangle
    // (optional `OpRemeshing_settings::generateDisplacementInfo`, only in eDecimate mode)
    eGpuRemeshingMeshTriangleSubdivisionInfoBuffer,
    // 4 x float16 per-vertex - 4th component unused
    // (optional `OpRemeshing_settings::generateDisplacementInfo`, only in eDecimate mode)
    eGpuRemeshingMeshVertexDirectionsBuffer,
    // 2 x float per-vertex
    // (optional `OpRemeshing_settings::generateDisplacementInfo`, only in eDecimate mode)
    eGpuRemeshingMeshVertexDirectionBoundsBuffer,

    // intermediate buffers used during process
    // ----------------------------------------
    // 3 x uint per-vertex as below
    // RemeshingVertexMergeInfo {
    //  uint32_t vertexIndexA;
    //  uint32_t vertexIndexB;
    //  float    blendAtoB;
    // }
    eGpuRemeshingMeshVertexMergeBuffer,

    // 1 x uint per-vertex
    eGpuRemeshingDebugVertexBuffer,
    // 1 x uint per-triangle
    eGpuRemeshingDebugTriangleBuffer,

    // 1 RemeshingCurrentState struct, used for feedback
    eGpuRemeshingCurrentStateBuffer,

    eGpuRemeshingScratchStart,
    // first come persistent count many resources
    // then task count many resources
};


struct GpuRemeshing_input
{
    // the remesher can attempt
    uint32_t deviceMemoryBudgetMegaBytes = 0;

    uint32_t meshTriangleCount = 0;
    uint32_t meshVertexCount   = 0;

    uint32_t maxDisplacementSubdivLevel;
};

struct GpuRemeshing_output
{
    // returns expected size information
    uint32_t           scratchTaskCount;
    ResourceAllocInfo* scratchTaskAllocs;

    // written after end function
    uint32_t meshTriangleCount = 0;
    uint32_t meshVertexCount   = 0;
};

typedef struct GpuRemeshingTask_s* GpuRemeshingTask;

// fills in GpuRemeshing_output::scratchTaskSizes
MICROMESH_API Result MICROMESH_CALL micromeshGpuRemeshingBeginTask(GpuRemeshing                remeshing,
                                                                   const OpRemeshing_settings* settings,
                                                                   const GpuRemeshing_input*   input,
                                                                   GpuRemeshing_output*        output,
                                                                   GpuRemeshingTask*           pTask);

// command buffer generation
// calls callback inside seqInfo
// returns Result::eContinue if multiple submits with potential readback inbetween are needed
MICROMESH_API Result MICROMESH_CALL micromeshGpuRemeshingContinueTask(GpuRemeshing     remeshing,
                                                                      GpuRemeshingTask task,
                                                                      gpu::CommandSequenceInfo<GpuRemeshingResource>* seqInfo);

// fills in GpuRemeshing_output::outputTriangleCount and GpuRemeshing_output::outputVertexCount
// task handle is invalid afterwards
MICROMESH_API Result MICROMESH_CALL micromeshGpuRemeshingEndTask(GpuRemeshing remeshing, GpuRemeshingTask task, GpuRemeshing_output* output);

}  // namespace gpu

}  // namespace micromesh
