/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
* 
* Propagate the vertex merge orders flagged on deduplicated vertices onto
* the original vertices
* 
*/


#undef MAIN_NAME
#define MAIN_NAME remeshCollapsePropagate

#include "remesh_common.h"

// Find whether one of the triangle edges has been flagged for merge
uint propagateMerge(RM_DATA_ARG uvec3 dedupIndices, uvec3 originalIndices)
{
    // Iterate over the 3 deduplicated vertices
    for(uint base = 0; base < 3; base++)
    {
        // Find the potential merge target of the deduplicated vertex
        uint mergeTarget = rvGetDedupMerged(dedupIndices[base]);

        if(mergeTarget != ~0u)
        {
            // Check whether one of the other two dedup vertices correspond
            // to the merge target. If yes, mark the corresponding original
            // vertices for merge
            for(uint i = 1; i < 3; i++)
            {
                uint candidate = (base + i) % 3;
                if(mergeTarget == dedupIndices[candidate])
                {
                    uint srcIndex = min(originalIndices[base], originalIndices[candidate]);
                    uint dstIndex = max(originalIndices[base], originalIndices[candidate]);
                    rvSetMergingWith(srcIndex, dstIndex);
                    rvSetMergingWith(dstIndex, ~0u);
                    return candidate;
                }
            }
        }
    }
    return ~0u;
}


MAIN
{
    if(RM_DATA(currentState).errorState != eRemesherErrorNone)
        return;

    uint index = uint(gl_GlobalInvocationID.x);

    if(index >= RM_DATA(scratchMetadata).uncompactedTriangleCount)
        return;

    if(!rtGetIsValid(index))
        return;


    uvec3 originalIndices =
        uvec3(RM_DATA(triangles)[3 * index + 0], RM_DATA(triangles)[3 * index + 1], RM_DATA(triangles)[3 * index + 2]);

    uvec3 dedupIndices = siGetDedupTriangle(index);

    if(originalIndices != dedupIndices)
    {
        // Propagate the flags from deduplicated vertices onto the original ones
        rvAtomicSetFlags(originalIndices[0], rvGetFlags(dedupIndices[0]));
        rvAtomicSetFlags(originalIndices[1], rvGetFlags(dedupIndices[1]));
        rvAtomicSetFlags(originalIndices[2], rvGetFlags(dedupIndices[2]));
    }
    // Propagate merge information if relevant
    propagateMerge(dedupIndices, originalIndices);
}