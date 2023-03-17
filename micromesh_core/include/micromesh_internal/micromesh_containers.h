//
// Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#pragma once

// let's route all heap-based allocation data-structures through here
// and for now just use STL

#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace micromesh
{
namespace container
{
template <class T>
using vector = std::vector<T>;

template <class Tkey, class Tval>
using unordered_map = std::unordered_map<Tkey, Tval>;

template <class Tkey>
using unordered_set = std::unordered_set<Tkey>;

template <typename T>
void fill(vector<T>& vec, T value)
{
    for(size_t i = 0; i < vec.size(); i++)
    {
        vec[i] = value;
    }
}

}  // namespace container
}  // namespace micromesh
