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
* Deduplicate the input vertices based on their location only. This first pass
* adds each vertex into the hash map and stores the resulting hash index
* 
*/


#undef MAIN_NAME
#define MAIN_NAME remeshDeduplicate

#include "remesh_common.h"


MAIN
{
    if(RM_DATA(currentState).errorState != eRemesherErrorNone)
        return;

    uint index = getActiveVertex(gl_GlobalInvocationID.x);
    if(index == ~0u)
        return;

    // Unused vertices are not considered to for better hash map performance
    if(hasFlag(rvGetFlags(index), RM_V_ORPHAN))
        return;
    if(hasFlag(rvGetFlags(index), RM_V_UNKNOWN))
    {
        rvAtomicAddFlag(index, RM_V_ORPHAN);
        return;
    }


    vec3 v = getOutputVertex(index);

    uint       h        = hashVertexIndex(v);
    const uint checksum = hashVertexChecksum(v);
    bool       found    = false;
    for(uint j = 0; j < 100; j++)
    {
        uint r = rhAtomicCompSwapChecksum(h, 0, checksum);

        if(r == 0)  // If empty, this vertex is new, add it to the list
        {
            rhSetStoredIndex(h, index);
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
    if(found)
    {
        rhAtomicIncRefCounter(h);
    }
    else
        h = RM_H_NOT_FOUND;
    rvSetHashIndex(index, h);
}
