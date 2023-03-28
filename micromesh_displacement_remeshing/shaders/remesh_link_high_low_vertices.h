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
* Associate the original vertices to the resulting simplified vertices 
* and triangles
* 
*/


#undef MAIN_NAME
#define MAIN_NAME remeshLinkHighLowVertices

#include "remesh_common.h"


vec3 getBaryCoordDirect(vec3 verts[3], float roots, vec3 d[3], vec3 pos)
{
    vec3 v0 = verts[0] + roots * d[0];
    vec3 u  = verts[1] + roots * d[1] - v0;
    vec3 v  = verts[2] + roots * d[2] - v0;
    vec3 n  = cross(u, v);
    vec3 w  = pos - v0;

    float n2Inv = 1.f / dot(n, n);

    vec3 baryCoord;
    baryCoord.z = (dot(cross(u, w), n) * n2Inv);
    baryCoord.y = (dot(cross(w, v), n) * n2Inv);
    baryCoord.x = 1.f - baryCoord.y - baryCoord.z;
    return baryCoord;
}


// Cubic equation solver, providing only real roots
// Based on Tschirnhaus-Vieta approach
// https://math.stackexchange.com/questions/1908861/using-trig-identity-to-solve-a-cubic-equation
vec3 cubicSolveReal(float a, float b, float c, float d)
{
    const float notFound = FLT_MAX;

    if(d == 0.f)
        return vec3(0.f, notFound, notFound);
    // Handle non-cubic equations
    if(a == 0.f)
    {
        if(b == 0.f)
        {
            if(c == 0.f)
            {
                return vec3(notFound);
            }
            // Linear
            return vec3(-d / c, notFound, notFound);
        }
        else
        {
            // Quadratic
            float delta = c * c - 4.f * b * d;
            if(delta < 0.f)
                return vec3(notFound);
            delta = sqrt(delta);
            return vec3((-c - delta) / (2.f * a), (-c + delta) / (2.f * a), notFound);
        }
    }

    b /= a;
    c /= a;
    d /= a;

    float disc, q, r, dum1, s, t, term1, r13;
    q = (3.f * c - (b * b)) / 9.f;
    r = -(27.f * d) + b * (9.f * c - 2.f * (b * b));
    r /= 54.f;
    disc = q * q * q + r * r;

    term1 = (b / 3.f);
    if(disc > 0.f)
    {  // one root real, two are complex
        float sqrtDisc = sqrt(disc);
        s              = r + sqrtDisc;
        s              = ((s < 0.f) ? -pow(-s, (1.f / 3.f)) : pow(s, (1.f / 3.f)));
        t              = r - sqrtDisc;
        t              = ((t < 0.f) ? -pow(-t, (1.f / 3.f)) : pow(t, (1.f / 3.f)));
        return vec3(-term1 + s + t, notFound, notFound);
    }

    // The remaining options are all real
    if(disc == 0.f)
    {  // All roots real, at least two are equal.
        r13 = ((r < 0.f) ? -pow(-r, (1.f / 3.f)) : pow(r, (1.f / 3.f)));
        return vec3(-term1 + 2.f * r13, -(r13 + term1), notFound);
    }
    // Only option left is that all roots are real and unequal (to get here, q < 0)
    q    = -q;
    dum1 = q * q * q;
    dum1 = acos(r / sqrt(dum1));
    r13  = 2.f * sqrt(q);
    vec3 res;
    res.x = -term1 + r13 * cos(dum1 / 3.f);
    res.y = -term1 + r13 * cos((dum1 + 2.f * M_PI) / 3.f);
    res.z = -term1 + r13 * cos((dum1 + 4.f * M_PI) / 3.f);
    return res;
}

float det(vec3 v0, vec3 v1, vec3 v2)
{
    return v0[0] * (v1[1] * v2[2] - v1[2] * v2[1]) - v0[1] * (v1[0] * v2[2] - v1[2] * v2[0])
           + v0[2] * (v1[0] * v2[1] - v1[1] * v2[0]);
}

// Find a (real-valued) parameter t defining a plane within the prismoid {p, d} that contains point q
// Algorithm by Andrea Maggiordomo, U. Milan, 2022
// FIXME: link to publication when available
vec3 findT(vec3 v[3], vec3 dir[3], vec3 q)
{
    vec3 f1 = dir[1] - dir[0];
    vec3 f2 = dir[2] - dir[0];

    vec3 p  = v[0] - q;
    vec3 e1 = v[1] - v[0];
    vec3 e2 = v[2] - v[0];

    float a = det(dir[0], f1, f2);
    float b = det(dir[0], f1, e2) + det(dir[0], e1, f2) + det(p, f1, f2);
    float c = det(p, e1, f2) + det(p, f1, e2) + det(dir[0], e1, e2);
    float d = det(p, e1, e2);

    return cubicSolveReal(a, b, c, d);
}

// Return the element of v with the smallest absolute value
float minAbs(vec3 v)
{
    if(abs(v[0]) < abs(v[1]) && abs(v[0]) < abs(v[2]))
        return v[0];
    if(abs(v[1]) < abs(v[2]))
        return v[1];
    return v[2];
}


float getDisplacement(vec3 v[3], vec3 d[3], vec3 q, out vec3 bary)
{
    vec3 roots = findT(v, d, q);

    vec3 validRoots     = vec3(FLT_MAX);
    uint validRootCount = 0;

    for(uint i = 0; i < 3; i++)
    {
        if(roots[i] == FLT_MAX)
            continue;

        bary = getBaryCoordDirect(v, roots[i], d, q);
        if(bary == clamp(bary, vec3(0.f), vec3(1.f)))
        {
            // Identify the roots yielding a point within the prismoid
            validRoots[validRootCount++] = roots[i];
        }
    }
    // Return the valid root that results in the minimum absolute displacement
    if(validRootCount > 0)
        return minAbs(validRoots);

    return uintBitsToFloat(~0u);
}

vec2 computeDisplacement(RM_DATA_ARG uint triIndex, vec3 position, out vec3 bary)
{
    vec3  triVerts[3];
    vec3  triNormals[3];
    uvec3 dedupTriangle = siGetDedupTriangle(triIndex);
    for(uint i = 0; i < 3; i++)
    {
        getOutputPosNormal(dedupTriangle[i], triVerts[i], triNormals[i]);
        // If the decimated vertex has the same position as the original vertex,
        // the displacement is null
        if(triVerts[i] == position)
            return vec2(0.f, 0.f);
    }

    float analyticalDisp = getDisplacement(triVerts, triNormals, position, bary);

    return vec2(analyticalDisp);
}

bool isSignificantDisplacement(uint triangle, float disp)
{
    return true;
    vec3  verts[3];
    uvec3 dedupTriangle = siGetDedupTriangle(triangle);
    for(uint i = 0; i < 3; i++)
    {
        verts[i] = getOutputVertex(dedupTriangle[i]);
    }
    vec3 l;
    for(uint i = 0; i < 3; i++)
    {
        l[i] = length(verts[i] - verts[(i + 1) % 3]);
    }
    float minL = min(l[0], min(l[1], l[2]));
    return (abs(disp) > 2.f * minL / 100.f);
}

void propagateToAdjacentTriangles(RM_DATA_ARG uint vertexIndex, vec3 position, uint originalIndex)
{
    uint encodedLastTriangleIndex = rvGetLastTriangle(vertexIndex);

    uint  bestTriangle = ~0u;
    float bestScore    = 1e34f;
    vec2  bestDisp;
    vec3  bestBary;

    float originalMaxEdgeLength = getOriginalMaxEdgeLength(originalIndex);

    while(encodedLastTriangleIndex != ~0u)
    {
        uvec2 lastTri           = decodePreviousTriangle(encodedLastTriangleIndex);
        uint  lastTriangleIndex = lastTri.x;
        vec3  bary;
        vec2  disp = computeDisplacement(RM_DATA_VAL lastTriangleIndex, position, bary);

        if(floatBitsToUint(disp.x) != ~0u)
        {
            // Extend the computed displacement by the maximum length of the edges
            // adjacent to the original vertex. This allows the bounds estimate to
            // encompass not only the original vertices, but also the surface of the
            // original triangles that may lie across the boundaries of the decimated
            // triangles
            disp += vec2(-originalMaxEdgeLength, originalMaxEdgeLength);
            float score = abs(disp.x);
            if(score < bestScore)
            {
                bestScore    = score;
                bestDisp     = disp;
                bestTriangle = lastTriangleIndex;
                bestBary     = bary;
            }
        }
        uint localVertexIndex    = lastTri.y;
        encodedLastTriangleIndex = rtGetPreviousTriangle(lastTriangleIndex, localVertexIndex);
    }

    if(bestTriangle != ~0u)
    {
        if(isSignificantDisplacement(bestTriangle, bestDisp.x))
            rtAtomicAddAggregatedCounter(bestTriangle, 1);
        rtAtomicMinMaxDisplacement(bestTriangle, vec2(bestDisp));
    }
}

float sqLength(vec3 v)
{
    return dot(v, v);
}

float computeSqMaxEdgeLengthTriangle(uint triangleIndex)
{
    uvec3 indices = siGetDedupTriangle(triangleIndex);
    vec3  v[3];
    for(uint i = 0; i < 3; i++)
    {
        v[i] = getOriginalVertex(indices[i]);
    }

    float d0 = sqLength(v[1] - v[0]);
    float d1 = sqLength(v[2] - v[0]);
    float d2 = sqLength(v[1] - v[2]);

    return max(d0, max(d1, d2));
}

float computeMaxOriginalEdgeLength(uint index)
{
    uint encodedLastTriangleIndex = rvGetLastTriangle(index);

    float maxEdgeLength = 0.f;

    while(encodedLastTriangleIndex != ~0u)
    {
        uvec2 lastTri           = decodePreviousTriangle(encodedLastTriangleIndex);
        uint  lastTriangleIndex = lastTri.x;

        maxEdgeLength = max(maxEdgeLength, computeSqMaxEdgeLengthTriangle(lastTriangleIndex));


        uint localVertexIndex    = lastTri.y;
        encodedLastTriangleIndex = rtGetPreviousTriangle(lastTriangleIndex, localVertexIndex);
    }
    return sqrt(maxEdgeLength);
}

MAIN
{
    if(RM_DATA(currentState).errorState != eRemesherErrorNone)
        return;

    uint index = uint(gl_GlobalInvocationID.x);
    if(index >= RM_CONSTANTS.vertexCount)
        return;

    vec3 originalPosition = getOriginalVertex(index);

    if(RM_CONSTANTS.backupPositions == 1)
    {
        setOriginalMaxEdgeLength(index, computeMaxOriginalEdgeLength(index));
    }

    uint currentIndex = index;
    bool isActive     = (RM_DATA(scratchVertexAliases)[currentIndex] == ~0u);

    while(RM_DATA(scratchVertexAliases)[currentIndex] != ~0u)
    {
        currentIndex = RM_DATA(scratchVertexAliases)[currentIndex];
    }
    if(!isActive)
    {
        // Flatten the collapse history
        RM_DATA(scratchVertexAliases)[index] = currentIndex;
    }
    else
    {
        return;
    }
    if(currentIndex == ~0u)
    {
        RM_DATA(currentState).errorState = eRemesherErrorNoTriangleFound;
        return;
    }
    // Uncomment for external display of the original vertices linked to the decimated vertex
    //vertexDebug[index] = currentIndex;
    propagateToAdjacentTriangles(RM_DATA_VAL currentIndex, originalPosition, index);
}
