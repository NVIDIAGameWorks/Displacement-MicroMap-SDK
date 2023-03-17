/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
* 
* 
* Second pass of full vertex deduplication: for each vertex fetch its
* representative unique vertex in the hash map. 
* 
*/


#undef MAIN_NAME
#define MAIN_NAME remeshDeduplicateFinalizeBase

#include "remesh_common.h"


MAIN
{
    if(RM_DATA(currentState).errorState != eRemesherErrorNone)
        return;

    uint index = uint(gl_GlobalInvocationID.x);
    if(index >= RM_CONSTANTS.indexCount / 3)
        return;

    uvec3 indices = uvec3(0);
    bvec3 replaced;
    // Rebuild the indices of the triangle from the hash map contents
    for(uint i = 0; i < 3; i++)
    {
        uint vertexIndex = RM_DATA(triangles)[3 * index + i];
        uint h           = rvGetHashIndex(vertexIndex);

        if(h == RM_H_NOT_FOUND)
        {
            rtSetIsValid(index, false);
            RM_DATA(currentState).errorState = eRemesherErrorVertexHashNotFound;
            return;
        }
        indices[i]  = rhGetStoredIndex(h);
        replaced[i] = (indices[i] != vertexIndex);
        if(h != ~0u)
            atomicMax(RM_DATA(scratchMetadata).uncompactedVertexCount, indices[i] + 1);
        else
        {
            RM_DATA(currentState).errorState = eRemesherErrorDebug;
            return;
        }
    }

    // If the triangle is degenerate, explicitly set
    // all indices to 0
    if(!isValid(indices))
    {
        rtSetIsValid(index, false);
        for(uint i = 0; i < 3; i++)
        {
            RM_DATA(triangles)[3 * index + i] = 0;
        }
        return;
    }


    for(uint i = 0; i < 3; i++)
    {
        if(indices[i] == ~0u)
        {
            RM_DATA(currentState).errorState = eRemesherErrorDebug;
            return;
        }
        if(replaced[i])
            RM_DATA(triangles)[3 * index + i] = indices[i];
        rvAtomicRemoveFlag(indices[i], RM_V_ORPHAN | RM_V_UNKNOWN);
    }

    rtSetIsValid(index, true);

    atomicAdd(RM_DATA(currentState).triangleCount, 1);
    atomicMax(RM_DATA(scratchMetadata).uncompactedTriangleCount, index + 1);

    for(uint i = 0; i < 3; i++)
    {
        rtSetPreviousTriangle(index, i, 0);
        rtSetEdgeIndex(index, i, 0);
    }
}
