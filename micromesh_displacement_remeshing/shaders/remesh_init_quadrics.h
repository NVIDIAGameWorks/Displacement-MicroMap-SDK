/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
* 
* Initialize per-vertex metadata
* 
*/


#undef MAIN_NAME
#define MAIN_NAME remeshInitQuadrics

#include "remesh_common.h"

MAIN
{
    if(RM_DATA(currentState).errorState != eRemesherErrorNone)
        return;

    uint index = uint(gl_GlobalInvocationID.x);
    if(index >= RM_CONSTANTS.vertexCount)
        return;


    if(hasFlag(rvGetFlags(index), RM_V_ORPHAN))
        return;

    // Enqueue the vertex in the list of active vertices
    uint activeIndex                            = atomicAdd(RM_DATA(scratchMetadata).activeVertices, 1);
    RM_DATA(scratchActiveVertices)[activeIndex] = index;

    // If the vertex is not already known as being orphan mark it as unknown. Its actual status will
    // be determined by the subsequent deduplication step
    if(!hasFlag(rvGetFlags(index), RM_V_ORPHAN))
        rvAtomicSetFlags(index, RM_V_UNKNOWN);


    rvSetLastTriangle(index, ~0u);
    rvSetMergingWith(index, ~0u);
    rvSetDedupMerged(index, ~0u);
    rvSetAttribsHash(index, 0);
    rvSetHashIndex(index, RM_H_NOT_FOUND);

    // On the first iteration, save the location of the original vertices
    // for later use when estimating the min/max displacement per triangle
    if(RM_CONSTANTS.backupPositions == 1)
    {
        setOriginalVertex(index, getOutputVertex(index));
    }
}
