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
* Deduplicate the input vertices based on their full attribute set, identified
*.by their user-defined hash key
* This kernel adds each vertex into the hash map and stores the resulting hash index
* 
*/


#undef MAIN_NAME
#define MAIN_NAME remeshDeduplicateBase

#include "remesh_common.h"


MAIN
{
    if(RM_DATA(currentState).errorState != eRemesherErrorNone)
        return;

    if(gl_GlobalInvocationID.x == 0)
        RM_DATA(currentState).vertexCount = RM_DATA(scratchMetadata).activeVertices;

    uint index = getActiveVertex(gl_GlobalInvocationID.x);
    if(index == ~0u)
        return;


    if(hasFlag(rvGetFlags(index), RM_V_ORPHAN))
        return;


    uint h = hashFullVertexIndex(index);

    const uint checksum = hashFullVertexChecksum(index);
    bool       found    = false;
    for(uint j = 0; j < 100; j++)
    {
        uint r = rhAtomicCompSwapChecksum(h, 0, checksum);

        if(r == 0)  // If empty, this vertex is new, add it to the list
        {
            found = true;
            break;
        }
        if(r != checksum)  // Collision, rehash and restart
        {
            h = rehashIndex(h);
        }
        else  // Vertex already exists, stop searching
        {
            found = true;
            break;
        }
    }
    if(!found)
        h = RM_H_NOT_FOUND;

    // We need a vertex of the mesh that will represent
    // all its duplicates.
    // We store the index of the vertex in the hash map
    // only if it is effectively used in the current state
    // of the mesh
    if(RM_DATA(scratchVertexAliases)[index] == ~0u)
        rhSetStoredIndex(h, index);

    rvSetHashIndex(index, h);
}
