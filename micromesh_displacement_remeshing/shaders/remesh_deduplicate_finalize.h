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
* Second pass of position-based deduplication: for each vertex fetch its
* representative unique vertex in the hash map. We also flag the vertices
* located on attribute discontinuities
* 
*/


#undef MAIN_NAME
#define MAIN_NAME remeshDeduplicateFinalize

#include "remesh_common.h"


MAIN
{
    if(RM_DATA(currentState).errorState != eRemesherErrorNone)
        return;

    uint index = uint(gl_GlobalInvocationID.x);
    if(index >= RM_DATA(scratchMetadata).uncompactedTriangleCount)
        return;

    if(!rtGetIsValid(index))
        return;

    uvec3 indices;
    uint  markedVertices = 0;
    for(uint i = 0; i < 3; i++)
    {
        uint h = rvGetHashIndex(RM_DATA(triangles)[3 * index + i]);
        if(h == RM_H_NOT_FOUND)
        {
            rtSetIsValid(index, false);
            RM_DATA(currentState).errorState = eRemesherErrorVertexHashNotFound;
            return;
        }
        uint storedIndex = rhGetStoredIndex(h);
        siSetDedupIndex(3 * index + i, storedIndex);

        indices[i] = storedIndex;

        if(hasFlag(rvGetFlags(storedIndex), RM_V_ORPHAN))
        {
            RM_DATA(currentState).errorState = eRemesherErrorDebug;
            RM_DATA(currentState).debug.x    = 3;
        }

        // If the same position has been added more than twice to the hash map it means
        // the vertex is located at the crossing of multiple discontinuities.
        // This vertex must be kept as-if, and is therefore marked as fixed.
        uint hRefCounter = rhGetRefCounter(h);
        if(hRefCounter > 2)
        {
            rvAtomicAddFlag(storedIndex, RM_V_MARKED | RM_V_FIXED);
            markedVertices++;
        }

        // If the same position has been added twice to the hash map
        // this location is on an attribute discontinuity. For decimation
        // we mark the vertex as being on a discontinuity. For relaxation
        // we mark it as fixed to be sure to preserve the exact UV boundaries
        if(hRefCounter == 2)
        {
            //uint attribHash = rvGetAttribsHash(RM_DATA(triangles)[3 * index + i]);
            {
                rvAtomicAddFlag(storedIndex, RM_CONSTANTS.remeshingMode == eDecimate ? RM_V_MARKED : (RM_V_MARKED | RM_V_FIXED));
                markedVertices++;
            }
        }
    }
    // If all vertices have been marked as being on a discontinuity the triangle defines
    // an intersection of discontinuity lines. In this case we may have two adjacent vertices
    // marked at being on a discontinuity, but in effect belong to two separate discontinuities.
    // We mark all 3 vertices as DOUBLE_MARKED to pay special attention to that case in later steps.
    if(markedVertices == 3)
    {
        for(uint i = 0; i < 3; i++)
        {
            rvAtomicAddFlag(indices[i], RM_V_DOUBLE_MARKED);
        }
    }
}
