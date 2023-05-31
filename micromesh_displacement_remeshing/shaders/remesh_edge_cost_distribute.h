/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
* 
* Propagate the edge cost descriptor of each edge into the triangles
* adjacent to its vertices
* 
*/


#undef MAIN_NAME
#define MAIN_NAME remeshEdgeCostDistribute


#include "remesh_common.h"

#define LOCAL_TRIANGLE_INDICES 32
uint localTriangleIndices[LOCAL_TRIANGLE_INDICES];
uint localTriangleCount = ~0u;


// An edge cost descriptor is a 64-bit uint containing the cost in the higher 32 bits and
// the edge index in the lower 32 bits
uint64_t packEdge(uint index, float cost)
{
    return pack64(u32vec2(computeEdgeId(reGetVertices(index)), encodeMinMaxFp32(cost)));
}


void propagateList(RM_DATA_ARG uint64_t edgeDesc)
{
    for(uint i = 0; i < localTriangleCount; i++)
    {
        atomicMin(RM_DATA(scratchTriangleDescs)[localTriangleIndices[i]], edgeDesc);
    }
}

// Iterate over the triangles neighboring the vertex at vertexIndex,
// and update the edge cost descriptor of those triangles if their
// descriptor has a value higher than the descriptor edgeDesc
void propagate(RM_DATA_ARG uint vertexIndex, uint64_t edgeDesc)
{
    uint encodedLastTriangleIndex = rvGetLastTriangle(vertexIndex);
    while(encodedLastTriangleIndex != ~0u)
    {
        uvec2 lastTri           = decodePreviousTriangle(encodedLastTriangleIndex);
        uint  lastTriangleIndex = lastTri.x;
        atomicMin(RM_DATA(scratchTriangleDescs)[lastTriangleIndex], edgeDesc);
        uint localVertexIndex    = lastTri.y;
        encodedLastTriangleIndex = rtGetPreviousTriangle(lastTriangleIndex, localVertexIndex);
    }
}

// Count the triangles adjacent to the vertex at vertexIndex
uint vertexValence(RM_DATA_ARG uint vertexIndex)
{
    uint counter = 0;

    uint encodedLastTriangleIndex = rvGetLastTriangle(vertexIndex);
    while(encodedLastTriangleIndex != ~0u)
    {
        uvec2 lastTri           = decodePreviousTriangle(encodedLastTriangleIndex);
        uint  lastTriangleIndex = lastTri.x;
        if(localTriangleCount < LOCAL_TRIANGLE_INDICES)
        {
            localTriangleIndices[localTriangleCount++] = lastTriangleIndex;
        }
        counter++;
        uint localVertexIndex    = lastTri.y;
        encodedLastTriangleIndex = rtGetPreviousTriangle(lastTriangleIndex, localVertexIndex);
    }
    return counter;
}


bool connectsWith(RM_DATA_ARG uint vertex, uvec2 searchEdge)
{
    uint encodedLastTriangleIndex = rvGetLastTriangle(vertex);

    bvec2 found = bvec2(false);

    while(encodedLastTriangleIndex != ~0u)
    {
        uvec2 lastTri           = decodePreviousTriangle(encodedLastTriangleIndex);
        uint  lastTriangleIndex = lastTri.x;


        uvec3 dedupVertices = siGetDedupTriangle(RM_DATA_VAL lastTriangleIndex);
        for(uint i = 0; i < 3; i++)
        {
            if(dedupVertices[i] == searchEdge.x)
                found.x = true;
            if(dedupVertices[i] == searchEdge.y)
                found.y = true;
            if(found.x && found.y)
                return true;
        }
        uint localVertexIndex    = lastTri.y;
        encodedLastTriangleIndex = rtGetPreviousTriangle(lastTriangleIndex, localVertexIndex);
    }
    return found.x && found.y;
}

bool isConnectedToValence4(RM_DATA_ARG uint edgeIndex)
{
    uvec2 edge                     = reGetVertices(RM_DATA_VAL edgeIndex);
    uint  encodedLastTriangleIndex = rvGetLastTriangle(edge.x);

    uint connectCounter = 0;
    uint connectingVertices[3];

    while(encodedLastTriangleIndex != ~0u)
    {
        uvec2 lastTri           = decodePreviousTriangle(encodedLastTriangleIndex);
        uint  lastTriangleIndex = lastTri.x;

        uvec3 dedupVertices = siGetDedupTriangle(RM_DATA_VAL lastTriangleIndex);

        for(uint i = 0; i < 3; i++)
        {
            if(dedupVertices[i] == edge.x || dedupVertices[i] == edge.y)
                continue;

            if(connectsWith(RM_DATA_VAL dedupVertices[i], edge))
            {
                bool alreadyCounted = false;
                for(uint j = 0; j < connectCounter; j++)
                {
                    if(connectingVertices[j] == dedupVertices[i])
                        alreadyCounted = true;
                }
                if(!alreadyCounted)
                {
                    if(vertexValence(RM_DATA_VAL dedupVertices[i]) <= 4)
                        return true;
                    connectingVertices[connectCounter++] = dedupVertices[i];
                }
            }
        }
        uint localVertexIndex    = lastTri.y;
        encodedLastTriangleIndex = rtGetPreviousTriangle(lastTriangleIndex, localVertexIndex);
    }
    return false;
}


bool isInSet(uint v, uint testSet[64], in uint testSetSize)
{
    for(uint i = 0; i < testSetSize; i++)
    {
        if(v == testSet[i])
            return true;
    }
    return false;
}

uint getNeighborVertices(RM_DATA_ARG uint vertexIndex, inout uint vertices[64])
{
    uint encodedLastTriangleIndex = rvGetLastTriangle(vertexIndex);

    uint counter = 0;

    while(encodedLastTriangleIndex != ~0u)
    {
        uvec2 lastTri           = decodePreviousTriangle(encodedLastTriangleIndex);
        uint  lastTriangleIndex = lastTri.x;

        uvec3 indices = siGetDedupTriangle(lastTriangleIndex);

        for(uint i = 0; i < 3; i++)
        {
            if(!isInSet(indices[i], vertices, counter))
            {
                vertices[counter++] = indices[i];
            }
        }
        uint localVertexIndex    = lastTri.y;
        encodedLastTriangleIndex = rtGetPreviousTriangle(lastTriangleIndex, localVertexIndex);
    }
    return counter;
}

uint countCommonVertices(RM_DATA_ARG uint edgeIndex)
{
    uvec2 edgeVertices = reGetVertices(edgeIndex);

    uint v0Neighbors[64], v1Neighbors[64];
    uint v0Counter = 0, v1Counter = 0;

    v0Counter = getNeighborVertices(RM_DATA_VAL edgeVertices[0], v0Neighbors);
    v1Counter = getNeighborVertices(RM_DATA_VAL edgeVertices[1], v1Neighbors);

    uint counter = 0;

    for(uint i = 0; i < v0Counter; i++)
    {
        if(isInSet(v0Neighbors[i], v1Neighbors, v1Counter))
        {
            counter++;
        }
    }
    return counter - 2;
}

// Iterate over the triangles adjacent to the vertex to determine
// whether the vertex is at the intersection of several discontinuity
// lines
bool isIntersection(RM_DATA_ARG uint vertexIndex)
{
    if(!hasFlag(rvGetFlags(vertexIndex), RM_V_MARKED))
        return false;

    uvec2 markedNeighbors = uvec2(~0u);

    uint encodedLastTriangleIndex = rvGetLastTriangle(vertexIndex);

    while(encodedLastTriangleIndex != ~0u)
    {
        uvec2 lastTri           = decodePreviousTriangle(encodedLastTriangleIndex);
        uint  lastTriangleIndex = lastTri.x;
        uint  counter           = 0;
        uvec3 dedupIndices      = siGetDedupTriangle(lastTriangleIndex);
        for(uint i = 0; i < 3; i++)
        {
            uint index = dedupIndices[i];
            if(index != vertexIndex && hasFlag(rvGetFlags(index), RM_V_MARKED))
            {
                // Count the unique marked neighbors
                if(markedNeighbors.x == ~0u)
                    markedNeighbors.x = index;
                else
                {
                    if(markedNeighbors.x != index)
                    {
                        if(markedNeighbors.y == ~0u)
                        {
                            markedNeighbors.y = index;
                        }
                        else
                        {
                            if(markedNeighbors.y != index)
                            {
                                // If 3 or more neighbors are marked, the vertex is at a
                                // discontinuity intersection
                                return true;
                            }
                        }
                    }
                }
            }
        }
        uint localVertexIndex    = lastTri.y;
        encodedLastTriangleIndex = rtGetPreviousTriangle(lastTriangleIndex, localVertexIndex);
    }
    return false;
}


bool isBlockedEdge(RM_DATA_ARG uvec2 edge)
{
    uvec2 flags = uvec2(rvGetFlags(edge.x), rvGetFlags(edge.y));

    bvec2 isMarked         = bvec2(hasFlag(flags.x, RM_V_MARKED), hasFlag(flags.y, RM_V_MARKED));
    bvec2 isDoubleMarked   = bvec2(hasFlag(flags.x, RM_V_DOUBLE_MARKED), hasFlag(flags.y, RM_V_DOUBLE_MARKED));
    bvec2 isAtIntersection = bvec2(hasFlag(flags.x, RM_V_FIXED) || isIntersection(RM_DATA_VAL edge.x),
                                   hasFlag(flags.y, RM_V_FIXED) || isIntersection(RM_DATA_VAL edge.y));

    bool isBlocked = (isMarked.x != isMarked.y) || (isAtIntersection.x || isAtIntersection.y)
                     || (isDoubleMarked.x && isDoubleMarked.y);

    return isBlocked;
}

// Compute the cost of collapsing edge, based on the edge length,
// local curvature, resulting vertex valence, and maximum number of
// original vertices represented by the decimated triangles
float edgeCost(RM_DATA_ARG uvec2 edge, uint edgeIndex)
{
    localTriangleCount   = 0;
    uvec2 vertexValences = uvec2(vertexValence(RM_DATA_VAL edge.x), vertexValence(RM_DATA_VAL edge.y));


    if(isBlockedEdge(RM_DATA_VAL edge))
        return 1e34f;


    bool isOnMeshEdge = hasFlag(rvGetFlags(edge.x), RM_V_EDGE) && hasFlag(rvGetFlags(edge.y), RM_V_EDGE);

    if(!isOnMeshEdge && isConnectedToValence4(RM_DATA_VAL edgeIndex))
        return 1e34f;


    if(!isOnMeshEdge && countCommonVertices(RM_DATA_VAL edgeIndex) > 2)
        return 1e34f;

    // If the max subd is reached return an absolute max cost
    if(RM_CONSTANTS.clampedSubdLevel != ~0u)
    {
        float storedCost = reGetCost(edgeIndex);
        if(RM_CONSTANTS.clampedSubdLevel == 0 || storedCost > subdivLevelGetVertexCount(RM_CONSTANTS.clampedSubdLevel - 1))
            return 1e34f;
    }


    float vertexDistance = length(getOutputVertex(edge.x) - getOutputVertex(edge.y));
    if(vertexDistance == 0.f)
        return -1.f;
    float cost = vertexDistance;

    if(cost > RM_CONSTANTS.errorThreshold)
        return cost;

    float ao = mixCurvatures(rvGetCurvature(edge.x), rvGetCurvature(edge.y));

    if(ao > RM_CONSTANTS.maxImportance)
        return 1e34f;

    cost *= 1.f + RM_CONSTANTS.curvatureImportance * ao;
    if(cost > RM_CONSTANTS.errorThreshold)
        return cost;


    uint valenceIfCollapsed = vertexValences.x + vertexValences.y - 4;
    if(valenceIfCollapsed > RM_CONSTANTS.maxValence || valenceIfCollapsed == 3)
        return 1e34f;


    float valenceCost = 1.f;
    return cost * valenceCost;
}


MAIN
{
    if(RM_DATA(currentState).errorState != eRemesherErrorNone)
        return;

    uint index = uint(gl_GlobalInvocationID.x);
    if(index >= RM_DATA(scratchMetadata).edgeCount)
        return;

    uvec2 verts = reGetVertices(index);
    if(verts.x == ~0u || verts.y == ~0u)
        return;

    float cost = edgeCost(RM_DATA_VAL verts, index);

    if(cost < 0.f)
        return;

    if(RM_CONSTANTS.remeshingMode == eDecimate)
        reSetCost(index, cost);

    uint64_t edgeDesc = packEdge(index, cost);

    // If the edge cannot be collapsed, do not propagate its cost
    uvec2 flags = uvec2(rvGetFlags(verts.x), rvGetFlags(verts.y));
    if((hasFlag(flags.x, RM_V_MARKED) != hasFlag(flags.y, RM_V_MARKED))
       || (hasFlag(flags.x, RM_V_FIXED) || hasFlag(flags.y, RM_V_FIXED)))
        return;

    if(cost > RM_CONSTANTS.errorThreshold)
        return;

    // Propagate the edge cost descriptor to the triangles adjacent to either vertex
    // Use the fast path if the vertex valence is not too high
    if(localTriangleCount < LOCAL_TRIANGLE_INDICES)
    {
        propagateList(RM_DATA_VAL edgeDesc);
    }
    else
    {
        propagate(RM_DATA_VAL verts.x, edgeDesc);
        propagate(RM_DATA_VAL verts.y, edgeDesc);
    }
}
