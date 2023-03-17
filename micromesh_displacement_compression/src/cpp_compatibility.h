/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Several C++20-like standard library functions and types backported for use
// in displacement_encoder and other projects.
// Define NVDISP_COMPAT_USE_CPP20 to make micromesh::compat be an alias for std::
// instead.

#pragma once

#ifdef NVDISP_COMPAT_USE_CPP20

#include <bit>
#include <span>
namespace nv::displacement
{
namespace compat = std;
}

#else

#include <cassert>
#include <cstddef>

namespace micromesh::compat
{
// C++17-compatible version of std::span
template <class T>
class span
{
    T*     _data;
    size_t _size_in_elements;

  public:
    using iterator       = T*;
    using const_iterator = const T*;

    span(T* start, size_t size_in_elements)
        : _data(start)
        , _size_in_elements(size_in_elements)
    {
    }

    template <class Container>
    span(Container& c)
    {
        _data             = c.data();
        _size_in_elements = c.size();
    }

    T* data() { return _data; }

    size_t size() { return _size_in_elements; }

    T& back()
    {
        assert(_size_in_elements > 0);
        return _data[_size_in_elements - 1];
    }

    iterator begin() { return _data; }

    const_iterator cbegin() { return _data; }

    iterator end() { return _data + _size_in_elements; }

    const_iterator cend() { return _data + _size_in_elements; }

    T& operator[](size_t idx) const
    {
        assert(idx < _size_in_elements);
        return _data[idx];
    }

    operator span<T const>() const { return span<T const>(_data, _size_in_elements); }
};

// C++17-compatible version of std::countl_zero, which returns the number of
// consecutive 0 bits in x, starting from the MSB.
int countl_zero(unsigned int x) noexcept;

// C++17-compatible version of std::popcount, which returns the number of
// set 1 bits in x.
int popcount(unsigned int x) noexcept;

}  // namespace micromesh::compat

#endif
