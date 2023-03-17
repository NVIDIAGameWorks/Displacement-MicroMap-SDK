/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
* 
* First pass of the edge list builder: for each edge we add it to the hash map
* if it has a chance of becoming a collapse candidate
* 
*/


#undef MAIN_NAME
#define MAIN_NAME remeshEdgeList

#include "remesh_common.h"


uvec2 getEdge(uint i, uvec3 indices)
{
    switch(i)
    {
    case 0:
        return indices.xy;
    case 1:
        return indices.xz;
    case 2:
        return indices.zy;
    }
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


    uvec3 indices = siGetDedupTriangle(index);

    bvec3 hasNewEdge  = bvec3(false);
    uvec3 hashIndices = uvec3(RM_H_NOT_SET);
    uint  addedEdges  = 0;
    for(uint i = 0; i < 3; i++)
    {
        uvec2      edge     = getEdge(i, indices);
        uint       h        = hashEdgeIndex(edge);
        const uint checksum = hashEdgeChecksum(edge);
        bool       found    = false;
        for(uint j = 0; j < 1000; j++)
        {
            uint r = rhAtomicCompSwapChecksum(h, 0, checksum);

            if(r == 0)  // If empty, this edge is new, add it to the list
            {
                hasNewEdge[i] = true;

                addedEdges++;
                found = true;
                break;
            }
            if(r != checksum)  // Collision, rehash and restart
            {
                h = rehashIndex(h);
            }
            else  // Edge already exists, stop searching
            {
                found = true;
                break;
            }
        }

        if(found)
        {
            rhAtomicIncRefCounter(h);
            hashIndices[i] = h;
        }
        else
        {
            hashIndices[i]                   = RM_H_NOT_FOUND;
            RM_DATA(currentState).errorState = eRemesherErrorEdgeHashNotFound;
            return;
        }
    }

    for(uint i = 0; i < 3; i++)
        siSetHashIndex(3 * index + i, hashIndices[i]);


    if(addedEdges > 0)
    {
        uint edgeIndex = atomicAdd(RM_DATA(scratchMetadata).edgeCount, addedEdges);

        if(edgeIndex >= RM_CONSTANTS.edgeListSize)
        {
            RM_DATA(currentState).errorState = eRemesherErrorOutOfEdgeStorage;
            return;
        }
        uint currentEdge = 0;
        for(uint i = 0; i < 3; i++)
        {
            if(hasNewEdge[i])
            {
                uvec2 edge = getEdge(i, indices);
                reSetVertices(edgeIndex + currentEdge, edge);
                rhSetStoredIndex(hashIndices[i], edgeIndex + currentEdge);
                currentEdge++;
            }
        }
    }
}
