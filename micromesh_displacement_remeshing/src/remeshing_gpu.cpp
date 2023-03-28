/*
* Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <iostream>
#include <string>

#include <micromesh/micromesh_displacement_remeshing.h>

#include "remeshing_internal.h"

#include "_autogen/remesh_init_quadrics.comp.glsl.h"
#include "_autogen/remesh_quadrics.comp.glsl.h"
#include "_autogen/remesh_edge_list.comp.glsl.h"
#include "_autogen/remesh_edge_list_finalize.comp.glsl.h"
#include "_autogen/remesh_collapse_flag.comp.glsl.h"
#include "_autogen/remesh_collapse_propagate.comp.glsl.h"
#include "_autogen/remesh_collapse_resolve.comp.glsl.h"
#include "_autogen/remesh_deduplicate.comp.glsl.h"
#include "_autogen/remesh_deduplicate_finalize.comp.glsl.h"
#include "_autogen/remesh_edge_cost_distribute.comp.glsl.h"
#include "_autogen/remesh_deduplicate_base.comp.glsl.h"
#include "_autogen/remesh_deduplicate_finalize_base.comp.glsl.h"
#include "_autogen/remesh_compact_indices.comp.glsl.h"
#include "_autogen/remesh_compact_vertices.comp.glsl.h"
#include "_autogen/remesh_link_high_low_vertices.comp.glsl.h"
#include "_autogen/remesh_deduplicate_save_aliases_base.comp.glsl.h"
#include "_autogen/remesh_apply_min_displacement.comp.glsl.h"
#include "_autogen/remesh_generate_subdivision_info.comp.glsl.h"
#include "_autogen/remesh_clear_micromesh_data.comp.glsl.h"
#include "_autogen/remesh_clear_hash_map.comp.glsl.h"
#include "_autogen/remesh_clear_edge_list.comp.glsl.h"


static inline uint32_t getBlockCount(uint32_t targetThreadCount, uint32_t blockSize)
{
    return (targetThreadCount + blockSize - 1) / blockSize;
}


static inline size_t multipleOf(size_t minSize, size_t multiple)
{
    return ((minSize + multiple - 1) / multiple) * multiple;
}

namespace micromesh
{
namespace gpu
{
// data independent
MICROMESH_API Result MICROMESH_CALL micromeshGpuRemeshingGetAvailableShaderCodeTypes(AvailableShaderCodeTypes* codeTypes)
{
    for(uint32_t i = 0; i < gpu::eShaderCodeMaxTypes; i++)
    {
        codeTypes->isAvailable[i] = (i == gpu::eShaderCodeSPIRV);
    }
    return Result::eSuccess;
}


gpu::UserPipelineInfo createUserPipelineInfo(GpuRemeshing_s& remeshing, uint32_t index, const std::string& name)
{
    gpu::UserPipelineInfo info;
    remeshing.pipelineNames[index] = name;
    info.debugName                 = remeshing.pipelineNames[index].c_str();
    info.type                      = PipelineType::eCompute;

    return info;
}

gpu::PipelineInfo createPipelineInfo(GpuRemeshing_s& remeshing, uint32_t index, const uint32_t* data, size_t size, const std::string& name)
{
    gpu::PipelineInfo info;
    remeshing.pipelineNames[index]         = name;
    info.debugName                         = remeshing.pipelineNames[index].c_str();
    info.pipelineLayoutIndex               = 0;
    info.sourceCount                       = 1;
    info.sources[eShaderCompute].codeType  = eShaderCodeSPIRV;
    info.sources[eShaderCompute].data      = data;
    info.sources[eShaderCompute].entryName = info.debugName;
    info.sources[eShaderCompute].fileName  = info.debugName;
    info.sources[eShaderCompute].size      = size;
    info.sources[eShaderCompute].type      = eShaderCompute;
    info.type                              = PipelineType::eCompute;
    return info;
}

#define CREATE_PIPELINE_INFO(index_, data_)                                                                            \
    remeshing.pipelineInfo[index_] = createPipelineInfo(remeshing, index_, data_, sizeof(data_), #data_)

#define CREATE_USER_PIPELINE_INFO(index_)                                                                              \
    remeshing.userPipelineInfo[index_] = createUserPipelineInfo(remeshing, index_, #index_)

// init internals
MICROMESH_API Result MICROMESH_CALL micromeshGpuRemeshingCreate(const GpuRemeshing_config* config,
                                                                GpuRemeshing*              pRemeshing,
                                                                const MessageCallbackInfo* messageCallback)
{
    // Sanity checks
    if(config->versionMajor != 0 || config->versionMinor != 1)
        return Result::eUnsupportedVersion;


    if(config->codeType != ShaderCodeType::eShaderCodeSPIRV)
        return Result::eUnsupportedShaderCodeType;

    if(config->supportedModeCount == 0 || config->supportedModes == nullptr)
        return Result::eInvalidValue;

    (*pRemeshing)             = new GpuRemeshing_s;
    GpuRemeshing_s& remeshing = *(*pRemeshing);

    remeshing.config = *config;


    for(uint32_t i = 0; i < gpu::GpuRemeshingResource::eGpuRemeshingScratchStart; i++)
    {
        remeshing.descriptorRanges[i].baseRegisterIndex = i;
        remeshing.descriptorRanges[i].descriptorCount   = 1;
        remeshing.descriptorRanges[i].descriptorType    = DescriptorType::eBufferReadWrite;
    }

    for(size_t i = 0; i < remesher::ScratchBuffers::eScratchBufferCount; i++)
    {
        remeshing.descriptorRanges[gpu::GpuRemeshingResource::eGpuRemeshingScratchStart + i].baseRegisterIndex =
            gpu::GpuRemeshingResource::eGpuRemeshingScratchStart + uint32_t(i);
        remeshing.descriptorRanges[gpu::GpuRemeshingResource::eGpuRemeshingScratchStart + i].descriptorCount = 1;
        remeshing.descriptorRanges[gpu::GpuRemeshingResource::eGpuRemeshingScratchStart + i].descriptorType =
            DescriptorType::eBufferReadWrite;
    }


    for(uint32_t i = 0; i < remesher::ScratchBuffers::eScratchBufferCount; i++)
    {
        remeshing.descriptorRanges[i].baseRegisterIndex = i;
        remeshing.descriptorRanges[i].descriptorCount   = 1;
        remeshing.descriptorRanges[i].descriptorType    = DescriptorType::eBufferReadWrite;
    }
    CREATE_PIPELINE_INFO(remesher::eInitializeQuadrics, remesh_init_quadrics_comp_glsl);
    CREATE_PIPELINE_INFO(remesher::eComputePerVertexQuadrics, remesh_quadrics_comp_glsl);
    CREATE_PIPELINE_INFO(remesher::eBuildEdgeList, remesh_edge_list_comp_glsl);
    CREATE_PIPELINE_INFO(remesher::eFinalizeEdgeList, remesh_edge_list_finalize_comp_glsl);
    CREATE_PIPELINE_INFO(remesher::eCollapseFlag, remesh_collapse_flag_comp_glsl);
    CREATE_PIPELINE_INFO(remesher::eCollapsePropagate, remesh_collapse_propagate_comp_glsl);
    CREATE_PIPELINE_INFO(remesher::eCollapseResolve, remesh_collapse_resolve_comp_glsl);
    CREATE_PIPELINE_INFO(remesher::eDeduplicate, remesh_deduplicate_comp_glsl);
    CREATE_PIPELINE_INFO(remesher::eDeduplicateFinalize, remesh_deduplicate_finalize_comp_glsl);
    CREATE_PIPELINE_INFO(remesher::eEdgeCostDistribute, remesh_edge_cost_distribute_comp_glsl);
    CREATE_PIPELINE_INFO(remesher::eDeduplicateBase, remesh_deduplicate_base_comp_glsl);
    CREATE_PIPELINE_INFO(remesher::eDeduplicateBaseFinalize, remesh_deduplicate_finalize_base_comp_glsl);
    CREATE_PIPELINE_INFO(remesher::eCompactIndices, remesh_compact_indices_comp_glsl);
    CREATE_PIPELINE_INFO(remesher::eCompactVertices, remesh_compact_vertices_comp_glsl);
    CREATE_PIPELINE_INFO(remesher::eClearHashMap, remesh_clear_hash_map_comp_glsl);
    CREATE_PIPELINE_INFO(remesher::eClearEdgeList, remesh_clear_edge_list_comp_glsl);
    CREATE_PIPELINE_INFO(remesher::eLinkHighLowVertices, remesh_link_high_low_vertices_comp_glsl);
    CREATE_PIPELINE_INFO(remesher::eDeduplicateBaseSaveAliases, remesh_deduplicate_save_aliases_base_comp_glsl);
    CREATE_PIPELINE_INFO(remesher::eApplyMinDisplacement, remesh_apply_min_displacement_comp_glsl);
    CREATE_PIPELINE_INFO(remesher::eGenerateSubdivisionInfo, remesh_generate_subdivision_info_comp_glsl);
    CREATE_PIPELINE_INFO(remesher::eClearMicromeshData, remesh_clear_micromesh_data_comp_glsl);

    remeshing.userDescriptorRanges[0].baseRegisterIndex = 0;
    remeshing.userDescriptorRanges[0].descriptorCount   = 1;
    remeshing.userDescriptorRanges[0].descriptorType    = DescriptorType::eBufferRead;

    CREATE_USER_PIPELINE_INFO(gpu::GpuRemeshingUserPipeline::eGpuRemeshingUserMergeVertices);

    return Result::eSuccess;
}
#undef CREATE_PIPELINE_INFO


MICROMESH_API Result MICROMESH_CALL micromeshGpuRemeshingDestroy(GpuRemeshing remeshing)
{
    delete remeshing;
    return Result::eSuccess;
}

MICROMESH_API Result MICROMESH_CALL micromeshGpuRemeshingGetSetup(GpuRemeshing remeshing, gpu::SetupInfo* setup)
{
    setup->pipelineCount       = remesher::ePipelineCount;
    setup->userPipelineCount   = gpu::GpuRemeshingUserPipeline::eGpuRemeshingUserPipelineCount;
    setup->pipelineLayoutCount = 1;
    setup->pipelineTypesUsed   = (1u << uint32_t(PipelineType::eCompute));


    setup->descriptorSetAllocationInfo.bufferMaxCount         = RemesherBindings::eBindingCount;
    setup->descriptorSetAllocationInfo.constantBufferMaxCount = 1;
    setup->descriptorSetAllocationInfo.bufferMaxCount         = 0;
    setup->descriptorSetAllocationInfo.storageBufferMaxCount  = RemesherBindings::eBindingCount;

    setup->globalConstantBuffer.type        = DescriptorType::eConstantBuffer;
    setup->globalConstantBuffer.buffer.size = 0;

    setup->readResourcesMaxCount = remesher::ReadbackBuffers::eReadbackCount;  // Metadata buffer, FIXME for others

    // maximum over all pipeline layouts' descriptorRangeCount
    setup->descriptorRangeMaxCountPerLayout = 1;

    if(setup->spirvBindingOffsetsUsed)  // FIXME
    {
        return Result::eFailure;
    }


    setup->scratchTaskCount       = remesher::eScratchBufferCount;
    setup->scratchPersistentCount = 0;

    return Result::eSuccess;
}
MICROMESH_API Result MICROMESH_CALL micromeshGpuRemeshingGetPipelineLayout(GpuRemeshing             remeshing,
                                                                           uint32_t                 pipelineLayoutIndex,
                                                                           gpu::PipelineLayoutInfo* pipeline)
{
    if(pipelineLayoutIndex == remesher::PipelineLayouts::eInternalPipelines)
    {
        pipeline->descriptorRangeCount  = uint32_t(remeshing->descriptorRanges.size());
        pipeline->descriptorRanges      = remeshing->descriptorRanges.data();
        pipeline->hasGlobalConstants    = false;
        pipeline->localPushConstantSize = sizeof(RemesherConstants);
        pipeline->pipelineTypesUsed     = 1u << uint32_t(PipelineType::eCompute);
    }
    return Result::eSuccess;
}
MICROMESH_API Result MICROMESH_CALL micromeshGpuRemeshingGetPipeline(GpuRemeshing remeshing, uint32_t pipelineIndex, gpu::PipelineInfo* pipeline)
{
    if(pipelineIndex >= remeshing->pipelineNames.size())
        return Result::eInvalidValue;
    *pipeline = remeshing->pipelineInfo[pipelineIndex];
    return Result::eSuccess;
}
MICROMESH_API Result MICROMESH_CALL micromeshGpuRemeshingGetUserPipeline(GpuRemeshing             remeshing,
                                                                         GpuRemeshingUserPipeline userPipelineIndex,
                                                                         gpu::UserPipelineInfo*   pipeline)
{
    if(userPipelineIndex >= eGpuRemeshingUserPipelineCount)
        return Result::eInvalidValue;

    *pipeline = remeshing->userPipelineInfo[userPipelineIndex];

    return Result::eSuccess;
}
MICROMESH_API Result MICROMESH_CALL micromeshGpuRemeshingGetPersistent(GpuRemeshing remeshing, gpu::PersistentResourceInfo* persistent)
{
    persistent->scratchPersistentCount = 0;
    return Result::eSuccess;
}

static uint32_t computeHashMapSize(uint32_t triangleCount, uint32_t vertexCount)
{
    // Estimate the number of edges in the mesh using Euler's formula, times 2 to provide extra space in the hash
    // map for collision mitigation
    return (std::max(1024u * 1024u, 3 * (vertexCount + triangleCount - 2) * 2));
}

// fills in GpuRemeshing_output::scratchTaskSizes
MICROMESH_API Result MICROMESH_CALL micromeshGpuRemeshingBeginTask(GpuRemeshing                remeshing,
                                                                   const OpRemeshing_settings* settings,
                                                                   const GpuRemeshing_input*   input,
                                                                   GpuRemeshing_output*        output,
                                                                   GpuRemeshingTask*           pTask)
{
    size_t requiredScratchMem{0ull};

    (*pTask) = new GpuRemeshingTask_s;
    for(uint32_t i = 0; i < output->scratchTaskCount; i++)
    {
        output->scratchTaskAllocs[i].type = DescriptorType::eBufferReadWrite;
    }

    output->scratchTaskAllocs[remesher::ScratchBuffers::eScratchIndexBuffer].buffer.size =
        multipleOf(input->meshTriangleCount * 3 * 2 * sizeof(uint32_t), 12 * sizeof(uint32_t));
    output->scratchTaskAllocs[remesher::ScratchBuffers::eScratchMetadata].buffer.size = sizeof(RemesherMetadata);

    uint32_t hashMapSize = computeHashMapSize(input->meshTriangleCount, input->meshVertexCount);
    output->scratchTaskAllocs[remesher::ScratchBuffers::eScratchHashMap].buffer.size = hashMapSize * sizeof(RemesherHashEntry);

    output->scratchTaskAllocs[remesher::ScratchBuffers::eScratchTriangles].buffer.size =
        input->meshTriangleCount * sizeof(RemesherTriangle);
    output->scratchTaskAllocs[remesher::ScratchBuffers::eScratchTrianglesDesc].buffer.size =
        input->meshTriangleCount * sizeof(uint64_t);

    uint32_t edgeListSize = std::max(1024u * 1024u, input->meshTriangleCount * 3);
    output->scratchTaskAllocs[remesher::ScratchBuffers::eScratchEdgeList].buffer.size = edgeListSize * sizeof(RemesherEdge);
    output->scratchTaskAllocs[remesher::ScratchBuffers::eScratchVertices].buffer.size =
        input->meshVertexCount * sizeof(RemesherVertex);

    output->scratchTaskAllocs[remesher::ScratchBuffers::eScratchVertexAliases].buffer.size =
        input->meshVertexCount * sizeof(uint32_t);

    // 3 floats for position, 1 for the max edge length of the original triangles adjacent to that vertex
    // (used for the displacement bounds estimate)
    output->scratchTaskAllocs[remesher::ScratchBuffers::eScratchOriginalPos].buffer.size =
        input->meshVertexCount * 4 * sizeof(float);

    output->scratchTaskAllocs[remesher::ScratchBuffers::eScratchTriangleSubdivisionInfoBackupBuffer].buffer.size =
        input->meshTriangleCount * sizeof(uint32_t);
    output->scratchTaskAllocs[remesher::ScratchBuffers::eScratchActiveVertices].buffer.size =
        input->meshVertexCount * sizeof(uint32_t);

    for(uint32_t i = 0; i < output->scratchTaskCount; i++)
    {
        requiredScratchMem += output->scratchTaskAllocs[i].buffer.size;
    }
    requiredScratchMem /= (1024 * 1024);
    if(requiredScratchMem > size_t(input->deviceMemoryBudgetMegaBytes))
        return Result::eFailure;

    remeshing->constants             = {};
    remeshing->constants.vertexCount = input->meshVertexCount;
    remeshing->constants.indexCount  = input->meshTriangleCount * 3;

    remeshing->constants.edgeListSize = edgeListSize;

    remeshing->constants.hashMapSize = hashMapSize;

    remeshing->constants.clampedSubdLevel = settings->clampDecimationLevel;

    remeshing->constants.errorThreshold      = settings->maxTriangleCount == -1 ? settings->errorThreshold : 0.1f;
    remeshing->constants.curvatureImportance = settings->vertexImportanceWeight;
    remeshing->constants.remeshingMode       = uint32_t(settings->mode);
    remeshing->constants.dispMapResolution.x = settings->dispMapResolution.x;
    remeshing->constants.dispMapResolution.y = settings->dispMapResolution.y;

    remeshing->constants.maxValence    = settings->maxVertexValence;
    remeshing->constants.maxImportance = settings->maxVertexImportance;

    remeshing->constants.backupPositions = 1;
    remeshing->constants.activeVertices  = 0;

    remeshing->constants.texcoordCount = settings->texcoordCount;
    remeshing->constants.texcoordIndex = settings->texcoordIndex;

    remeshing->constants.directionBoundsFactor = settings->directionBoundsFactor;

    remeshing->settings = *settings;

    (*pTask)->stableSince       = 0;
    (*pTask)->decimationStarted = false;
    (*pTask)->iterationIndex    = ~0u;
    (*pTask)->lastTriangleCount = ~0u;
    (*pTask)->decimationState   = remesher::DecimationState::eDecimation;
    return Result::eSuccess;
}


void cmdPushConstants(gpu::CommandSequenceInfo<GpuRemeshingResource>* seqInfo, const RemesherConstants& c, uint32_t pipelineLayoutIndex)
{
    CmdLocalConstants pc;
    pc.pipelineLayoutIndex = pipelineLayoutIndex;
    pc.byteSize            = sizeof(RemesherConstants);
    pc.data                = &c;
    seqInfo->pfnGenerateGpuCommand(micromesh::gpu::CommandType::eGlobalConstants, &pc, seqInfo->userData);
}

void cmdFillBuffer(gpu::CommandSequenceInfo<GpuRemeshingResource>* seqInfo, uint32_t index, uint32_t clearValue = 0u)
{
    CmdClearResources cr;
    cr.resourceCount = 1;
    gpu::ResourceInfo<uint32_t> res;
    res.resourceIndex = index;
    cr.resources      = &res;
    cr.clearValue     = clearValue;
    seqInfo->pfnGenerateGpuCommand(micromesh::gpu::CommandType::eClearResources, &cr, seqInfo->userData);
}
void cmdBufferBarrier(gpu::CommandSequenceInfo<GpuRemeshingResource>* seqInfo)
{
    CmdBarrier b;
    b.readBits  = BarrierBits::eBarrierBufferBit;
    b.writeBits = BarrierBits::eBarrierBufferBit;
    seqInfo->pfnGenerateGpuCommand(micromesh::gpu::CommandType::eBarrier, &b, seqInfo->userData);
}

void cmdDispatch(gpu::CommandSequenceInfo<GpuRemeshingResource>* seqInfo,
                 uint32_t                                        pipelineIndex,
                 uint32_t                                        threadCount,
                 bool                                            isUserPipeline = false,
                 uint32_t                                        blockSize      = REMESHER_BLOCK_SIZE)
{
    CmdBindPipeline bp;
    bp.pipelineIndex = pipelineIndex;
    seqInfo->pfnGenerateGpuCommand(isUserPipeline ? micromesh::gpu::CommandType::eBindUserPipeline :
                                                    micromesh::gpu::CommandType::eBindPipeline,
                                   &bp, seqInfo->userData);

    CmdDispatch d;
    d.gridX = getBlockCount(threadCount, blockSize);
    d.gridY = 1;
    d.gridZ = 1;
    seqInfo->pfnGenerateGpuCommand(micromesh::gpu::CommandType::eDispatch, &d, seqInfo->userData);
    cmdBufferBarrier(seqInfo);
}

void cmdUpdateLocalConstants(GpuRemeshing remeshing, GpuRemeshingTask task, gpu::CommandSequenceInfo<GpuRemeshingResource>* seqInfo)
{
    CmdLocalConstants lc;
    lc.byteSize            = sizeof(RemesherConstants);
    lc.data                = &(remeshing->constants);
    lc.pipelineLayoutIndex = 0;
    seqInfo->pfnGenerateGpuCommand(micromesh::gpu::CommandType::eLocalConstants, &lc, seqInfo->userData);
}

uint32_t activeVertices(GpuRemeshing remeshing)
{
    if(remeshing->constants.activeVertices == 0)
        return remeshing->constants.vertexCount;
    return remeshing->constants.activeVertices;
}


void cmdInitializeBuffers(GpuRemeshing remeshing, GpuRemeshingTask task, gpu::CommandSequenceInfo<GpuRemeshingResource>* seqInfo)
{
    cmdPushConstants(seqInfo, remeshing->constants, remesher::PipelineLayouts::eInternalPipelines);

    cmdFillBuffer(seqInfo, gpu::eGpuRemeshingScratchStart + remesher::ScratchBuffers::eScratchMetadata);
    cmdFillBuffer(seqInfo, gpu::eGpuRemeshingCurrentStateBuffer);

    if(task->iterationIndex == ~0u)
    {
        cmdFillBuffer(seqInfo, gpu::eGpuRemeshingScratchStart + remesher::ScratchBuffers::eScratchVertices);
        cmdFillBuffer(seqInfo, gpu::eGpuRemeshingScratchStart + remesher::ScratchBuffers::eScratchTriangles, ~0u);
        cmdFillBuffer(seqInfo, gpu::eGpuRemeshingScratchStart + remesher::ScratchBuffers::eScratchVertexAliases, ~0u);
        task->iterationIndex = 0u;
    }

    cmdBufferBarrier(seqInfo);

    cmdDispatch(seqInfo, remesher::Pipelines::eClearHashMap, remeshing->constants.hashMapSize / 2);

    cmdDispatch(seqInfo, remesher::Pipelines::eClearEdgeList, remeshing->constants.edgeListSize);
    cmdDispatch(seqInfo, remesher::Pipelines::eInitializeQuadrics, remeshing->constants.vertexCount);
}


void cmdDeduplicate(GpuRemeshing remeshing, GpuRemeshingTask task, gpu::CommandSequenceInfo<GpuRemeshingResource>* seqInfo)
{
    cmdDispatch(seqInfo, remesher::Pipelines::eDeduplicateBase, activeVertices(remeshing));

    cmdDispatch(seqInfo, remesher::Pipelines::eDeduplicateBaseSaveAliases, activeVertices(remeshing));

    cmdDispatch(seqInfo, remesher::Pipelines::eDeduplicateBaseFinalize, remeshing->currentIndexCount / 3);

    if(task->decimationState != remesher::DecimationState::eFinalCompaction)
    {
        cmdDispatch(seqInfo, remesher::Pipelines::eClearHashMap, remeshing->constants.hashMapSize / 2);

        cmdDispatch(seqInfo, remesher::Pipelines::eDeduplicate, activeVertices(remeshing));
        cmdDispatch(seqInfo, remesher::Pipelines::eDeduplicateFinalize, remeshing->currentIndexCount / 3);

        cmdDispatch(seqInfo, remesher::Pipelines::eClearHashMap, remeshing->constants.hashMapSize / 2);
    }
}

void cmdBuildEdgeList(GpuRemeshing remeshing, GpuRemeshingTask task, gpu::CommandSequenceInfo<GpuRemeshingResource>* seqInfo)
{
    cmdDispatch(seqInfo, remesher::Pipelines::eBuildEdgeList, remeshing->currentIndexCount / 3);
    cmdDispatch(seqInfo, remesher::Pipelines::eFinalizeEdgeList, remeshing->currentIndexCount / 3);
}

void cmdOptimizeVertexPositions(GpuRemeshing remeshing, GpuRemeshingTask task, gpu::CommandSequenceInfo<GpuRemeshingResource>* seqInfo)
{
    //cmdDispatch(seqInfo, remesher::Pipelines::eClearMicromeshData, remeshing->currentIndexCount / 3);
    if(task->decimationState == remesher::DecimationState::eVertexOptimization
       || task->decimationState == remesher::DecimationState::eMicromeshGeneration)
    {
        cmdDispatch(seqInfo, remesher::Pipelines::eClearMicromeshData, remeshing->constants.indexCount / 3);
    }

    cmdDispatch(seqInfo, remesher::Pipelines::eLinkHighLowVertices, remeshing->constants.vertexCount);

    if(task->decimationState == remesher::DecimationState::eVertexOptimization
       || task->decimationState == remesher::DecimationState::eMicromeshGeneration

    )
    {
        cmdDispatch(seqInfo, remesher::Pipelines::eApplyMinDisplacement, remeshing->currentIndexCount / 3);
    }
}


void cmdGenerateMicromeshInfo(GpuRemeshing remeshing, GpuRemeshingTask task, gpu::CommandSequenceInfo<GpuRemeshingResource>* seqInfo)
{
    // 11 iterations is also hardcoded in the shader
    // FIXME: do something more generic
    for(uint i = 0; i < 12; i++)
    {
        remeshing->constants.iterationIndex = i;
        cmdUpdateLocalConstants(remeshing, task, seqInfo);
        cmdDispatch(seqInfo, remesher::Pipelines::eGenerateSubdivisionInfo, remeshing->currentIndexCount / 3);  //remeshing->constants.indexCount/3);
    }
    remeshing->constants.iterationIndex = 12;
    cmdUpdateLocalConstants(remeshing, task, seqInfo);
    cmdDispatch(seqInfo, remesher::Pipelines::eGenerateSubdivisionInfo, remeshing->constants.vertexCount);  //remeshing->constants.indexCount/3);
}


void cmdCompactVertices(GpuRemeshing remeshing, GpuRemeshingTask task, gpu::CommandSequenceInfo<GpuRemeshingResource>* seqInfo)
{
    remeshing->constants.compactionPass = 0;
    cmdUpdateLocalConstants(remeshing, task, seqInfo);

    const uint32_t vertexCount = remeshing->constants.vertexCount;

    cmdDispatch(seqInfo, remesher::Pipelines::eCompactVertices, vertexCount);

    remeshing->constants.compactionPass = 1;
    cmdUpdateLocalConstants(remeshing, task, seqInfo);
    cmdDispatch(seqInfo, remesher::Pipelines::eCompactVertices, vertexCount);

    remeshing->constants.compactionPass = 2;
    cmdUpdateLocalConstants(remeshing, task, seqInfo);
    cmdDispatch(seqInfo, remesher::Pipelines::eCompactVertices, vertexCount);

    cmdDispatch(seqInfo, eGpuRemeshingUserMergeVertices, vertexCount, true);
}


void cmdCompactIndices(GpuRemeshing remeshing, GpuRemeshingTask task, gpu::CommandSequenceInfo<GpuRemeshingResource>* seqInfo)
{
    remeshing->constants.compactionPass = 0;
    cmdUpdateLocalConstants(remeshing, task, seqInfo);
    cmdDispatch(seqInfo, remesher::Pipelines::eCompactIndices, remeshing->currentIndexCount / 3);

    remeshing->constants.compactionPass = 1;
    cmdUpdateLocalConstants(remeshing, task, seqInfo);
    cmdDispatch(seqInfo, remesher::Pipelines::eCompactIndices, remeshing->currentIndexCount / 3);

    remeshing->constants.compactionPass = 2;
    cmdUpdateLocalConstants(remeshing, task, seqInfo);
    cmdDispatch(seqInfo, remesher::Pipelines::eCompactIndices, remeshing->currentIndexCount / 3);
}


// command buffer generation
// calls callback inside seqInfo
// returns Result::eContinue if multiple submits with potential readback inbetween are needed
MICROMESH_API Result MICROMESH_CALL micromeshGpuRemeshingContinueTask(GpuRemeshing     remeshing,
                                                                      GpuRemeshingTask task,
                                                                      gpu::CommandSequenceInfo<GpuRemeshingResource>* seqInfo)
{
    bool hasTriangleCountTarget = (remeshing->settings.maxTriangleCount > 0);

    remeshing->currentIndexCount = remeshing->constants.indexCount;
    if(seqInfo->previousReadData != nullptr)
    {
        auto* cs = reinterpret_cast<const RemeshingCurrentState*>(
            seqInfo->previousReadData->resourceDatas[remesher::ReadbackBuffers::eReadbackCurrentState]);
        auto* md = reinterpret_cast<const RemesherMetadata*>(
            seqInfo->previousReadData->resourceDatas[remesher::ReadbackBuffers::eReadbackMetadata]);
        if(cs->errorState != RemesherErrorState::eRemesherErrorNone)
            return Result::eFailure;
        remeshing->constants.activeVertices = md->activeVertices;
        if(cs->triangleCount > 0)
        {
            remeshing->currentIndexCount = cs->triangleCount * 3;
        }
        uint32_t gain = (100 * (task->lastTriangleCount - cs->triangleCount)) / std::max(1u, task->lastTriangleCount);
        if(task->lastTriangleCount != 0 && cs->triangleCount > 0 && (task->lastTriangleCount - cs->triangleCount) > 1)
            task->decimationStarted = true;

        // If the remesher could not achieve any gain within the first 1000 iterations, the mesh is likely
        // too simple and/or too constrained, and cannot be decimated
        if(task->iterationIndex > 1000 && !task->decimationStarted)
            return Result::eFailure;

        if(task->decimationState == remesher::DecimationState::eVertexOptimization)
        {
            task->decimationState = remesher::DecimationState::eFinalCompaction;
        }

        if(task->decimationState == remesher::DecimationState::eDecimation)
        {
            bool hasHitTargetTriangleCount = (hasTriangleCountTarget && cs->triangleCount > 0
                                              && (cs->triangleCount) < uint32_t(remeshing->settings.maxTriangleCount));
            hasHitTargetTriangleCount =
                hasHitTargetTriangleCount
                || ((task->lastTriangleCount - cs->triangleCount) == 0 && task->stableSince > 50 && task->decimationStarted);
            if(hasHitTargetTriangleCount)
                remeshing->constants.errorThreshold = 0.f;
            if((gain < 1 || hasHitTargetTriangleCount) && task->iterationIndex != ~0u)
            {
                if(!hasTriangleCountTarget || hasHitTargetTriangleCount)
                {
                    task->stableSince++;
                    if((!hasTriangleCountTarget && task->stableSince > 50 && task->iterationIndex > 2)
                       || (hasHitTargetTriangleCount && task->stableSince > 50))
                    {
                        if(remeshing->settings.generateMicromeshInfo && task->decimationState == remesher::DecimationState::eDecimation)
                        {
                            task->decimationState = remesher::DecimationState::eVertexOptimization;
                        }
                        else
                        {
                            task->decimationState = remesher::DecimationState::eFinalCompaction;
                        }
                    }
                }
                else
                {
                    float ratio =
                        cs->triangleCount > 0 ? (float(cs->triangleCount) / float(remeshing->settings.maxTriangleCount)) : 2.f;

                    remeshing->constants.errorThreshold *= std::min(1.2f, ratio);
                    task->stableSince++;
                }
            }
            else
            {
                task->stableSince = 0;
            }
        }
        task->lastTriangleCount = cs->triangleCount;
    }


    CmdBindResources br;
    br.pipelineLayoutIndex = 0;
    br.resourceCount = gpu::GpuRemeshingResource::eGpuRemeshingScratchStart + remesher::ScratchBuffers::eScratchBufferCount;
    std::array<gpu::ResourceInfo<uint32_t>, gpu::GpuRemeshingResource::eGpuRemeshingScratchStart + remesher::ScratchBuffers::eScratchBufferCount> resources;
    for(uint32_t i = 0; i < gpu::GpuRemeshingResource::eGpuRemeshingScratchStart; i++)
        resources[i].resourceIndex = uint32_t(i);


    for(size_t i = 0; i < remesher::ScratchBuffers::eScratchBufferCount; i++)
        resources[gpu::eGpuRemeshingScratchStart + i].resourceIndex = gpu::eGpuRemeshingScratchStart + uint32_t(i);
    br.resources = resources.data();


    seqInfo->pfnGenerateGpuCommand(micromesh::gpu::CommandType::eBindResources, &br, seqInfo->userData);

    CmdLocalConstants lc;
    lc.byteSize            = sizeof(RemesherConstants);
    lc.data                = &(remeshing->constants);
    lc.pipelineLayoutIndex = 0;
    seqInfo->pfnGenerateGpuCommand(micromesh::gpu::CommandType::eLocalConstants, &lc, seqInfo->userData);

    if(task->decimationState != remesher::DecimationState::eFinalReadback)
    {
        cmdInitializeBuffers(remeshing, task, seqInfo);

        cmdDeduplicate(remeshing, task, seqInfo);


        if(task->decimationState != remesher::DecimationState::eFinalCompaction)
        {
            cmdDispatch(seqInfo, remesher::Pipelines::eComputePerVertexQuadrics, remeshing->currentIndexCount / 3);

            cmdBuildEdgeList(remeshing, task, seqInfo);


            cmdOptimizeVertexPositions(remeshing, task, seqInfo);

            if(task->decimationState == remesher::DecimationState::eMicromeshGeneration)
            {
                cmdGenerateMicromeshInfo(remeshing, task, seqInfo);
            }
            if(task->decimationState == remesher::DecimationState::eDecimation)
            {
                cmdDispatch(seqInfo, remesher::Pipelines::eEdgeCostDistribute, remeshing->constants.edgeListSize);
                cmdDispatch(seqInfo, remesher::Pipelines::eCollapseFlag, remeshing->constants.edgeListSize);

                cmdDispatch(seqInfo, remesher::Pipelines::eCollapsePropagate, remeshing->currentIndexCount / 3);

                cmdDispatch(seqInfo, remesher::Pipelines::eCollapseResolve, activeVertices(remeshing));


                cmdDispatch(seqInfo, eGpuRemeshingUserMergeVertices, activeVertices(remeshing), true);

                seqInfo->pfnGenerateGpuCommand(micromesh::gpu::CommandType::eBindResources, &br, seqInfo->userData);

                remeshing->constants.isFinalCompaction = 0;
                //if (task->iterationIndex % 10 == 0)
                {
                    cmdCompactIndices(remeshing, task, seqInfo);
                }
            }
        }
        if(task->decimationState == remesher::DecimationState::eFinalCompaction)
        {
            remeshing->constants.isFinalCompaction = 1;
            cmdCompactVertices(remeshing, task, seqInfo);
            seqInfo->pfnGenerateGpuCommand(micromesh::gpu::CommandType::eBindResources, &br, seqInfo->userData);
            cmdCompactIndices(remeshing, task, seqInfo);
        }
    }

    {
        CmdReadResources                      rr;
        std::array<ResourceInfo<uint32_t>, 2> ri;
        rr.resourceCount = uint32_t(ri.size());
        ri[remesher::ReadbackBuffers::eReadbackMetadata].resourceIndex =
            gpu::eGpuRemeshingScratchStart + remesher::ScratchBuffers::eScratchMetadata;
        ri[remesher::ReadbackBuffers::eReadbackCurrentState].resourceIndex = gpu::eGpuRemeshingCurrentStateBuffer;
        rr.resources                                                       = ri.data();
        seqInfo->pfnGenerateGpuCommand(micromesh::gpu::CommandType::eReadResources, &rr, seqInfo->userData);
    }
    task->iterationIndex++;
    remeshing->constants.backupPositions = 0;
    remeshing->constants.iterationIndex  = task->iterationIndex;
    //remeshing->constants.hashMapSize = computeHashMapSize(remeshing->currentIndexCount/3, activeVertices(remeshing));

    if(remeshing->settings.generateMicromeshInfo)
    {
        // Finalize decimation and micromesh generation info
        if(task->decimationState == remesher::DecimationState::eVertexOptimization)
        {
            task->decimationState = remesher::DecimationState::eMicromeshGeneration;
            return Result::eContinue;
        }

        if(task->decimationState == remesher::DecimationState::eMicromeshGeneration)
        {
            task->decimationState = remesher::DecimationState::eFinalCompaction;
            return Result::eContinue;
        }
    }
    if(task->decimationState == remesher::DecimationState::eFinalCompaction)
    {
        task->decimationState = remesher::DecimationState::eFinalReadback;
        return Result::eContinue;
    }
    if(task->decimationState == remesher::DecimationState::eFinalReadback)
    {
        return Result::eSuccess;
    }
    // Decimation continues
    return Result::eContinue;
}

// fills in GpuRemeshing_output::outputTriangleCount and GpuRemeshing_output::outputVertexCount
// task handle is invalid afterwards
MICROMESH_API Result MICROMESH_CALL micromeshGpuRemeshingEndTask(GpuRemeshing remeshing, GpuRemeshingTask task, GpuRemeshing_output* output)
{
    //delete reinterpret_cast<GpuRemeshingTask_s*>(task);
    GpuRemeshingTask_s* t = task;
    delete t;
    return Result::eSuccess;
}

}  // namespace gpu
}  // namespace micromesh