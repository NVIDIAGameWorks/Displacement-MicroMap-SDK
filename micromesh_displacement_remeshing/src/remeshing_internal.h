
/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Internal interface for the block encoder and decoder. Ported from the
// old displacement encoder.
// The idea is that this exposes the interface the mesh encoder needs to be
// efficient: 11-bit UNORM values in u-major layout, all API preconditions
// already checked. Data transfer and conversion operations are left up to the
// higher-level API functions. Ideally, these would be templated so we could
// get good code generation.
// This also exposes the functions that are unit-tested.

#pragma once
#include "vulkan/vulkan_core.h"
#include "micromesh/micromesh_displacement_remeshing.h"
#include "shaders/remesh_host_device.h"
#include "shaders/remesh_bindings.h"
#include <string>
#include <array>

namespace micromesh
{
namespace remesher
{
enum ScratchBuffers
{
    eScratchIndexBuffer,
    eScratchTriangles,
    eScratchMetadata,
    eScratchHashMap,
    eScratchEdgeList,
    eScratchVertices,
    eScratchTrianglesDesc,
    eScratchVertexAliases,
    eScratchOriginalPos,
    eScratchTriangleSubdivisionInfoBackupBuffer,
    eScratchActiveVertices,
    eScratchBufferCount
};


enum Pipelines
{
    eInitializeQuadrics,
    eComputePerVertexQuadrics,
    eBuildEdgeList,
    eFinalizeEdgeList,
    eCollapseFlag,
    eCollapsePropagate,
    eCollapseResolve,
    eCompactIndices,
    eCompactVertices,
    eDeduplicate,
    eDeduplicateFinalize,
    eEdgeCostDistribute,
    eDeduplicateBase,
    eDeduplicateBaseFinalize,
    eClearHashMap,
    eClearEdgeList,
    eLinkHighLowVertices,
    eDeduplicateBaseSaveAliases,
    eApplyMinDisplacement,
    eGenerateSubdivisionInfo,
    eClearMicromeshData,
    ePipelineCount
};

enum ReadbackBuffers
{
    eReadbackMetadata     = 0,
    eReadbackCurrentState = 1,
    eReadbackCount
};

enum PipelineLayouts
{
    eInternalPipelines,
    ePipelineLayoutCount
};

enum DecimationState
{
    eDecimation,
    eVertexOptimization,
    eMicromeshGeneration,
    eFinalCompaction,
    eFinalReadback
};

}  // namespace remesher

namespace gpu
{
struct GpuRemeshing_s
{
    GpuRemeshing_config config;
    std::array<DescriptorRangeInfo, remesher::ScratchBuffers::eScratchBufferCount + gpu::GpuRemeshingResource::eGpuRemeshingScratchStart> descriptorRanges;
    std::array<gpu::PipelineInfo, remesher::ePipelineCount> pipelineInfo;
    std::array<std::string, remesher::ePipelineCount>       pipelineNames;

    std::array<DescriptorRangeInfo, 1> userDescriptorRanges;
    std::array<gpu::UserPipelineInfo, gpu::GpuRemeshingUserPipeline::eGpuRemeshingUserPipelineCount> userPipelineInfo;


    RemesherConstants    constants;
    OpRemeshing_settings settings;
    uint32_t             currentIndexCount;
};

struct GpuRemeshingTask_s
{
    uint32_t                  iterationIndex;
    uint32_t                  lastTriangleCount;
    uint32_t                  stableSince;
    bool                      decimationStarted;
    remesher::DecimationState decimationState;
};

}  // namespace gpu


}  // namespace micromesh