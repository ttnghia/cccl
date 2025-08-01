//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// insert_iterator

// Test nested types and data members:

// template <InsertionContainer Cont>
// class insert_iterator {
// protected:
//   Cont* container;
//   Cont::iterator iter;
// public:
//   typedef Cont                   container_type;
//   typedef void                   value_type;
//   typedef void                   difference_type;
//   typedef void                   reference;
//   typedef void                   pointer;
// };

#include <cuda/std/inplace_vector>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class C>
struct find_members : private cuda::std::insert_iterator<C>
{
  __host__ __device__ explicit find_members(C& c)
      : cuda::std::insert_iterator<C>(c, c.begin())
  {}
  __host__ __device__ void test()
  {
    this->container = 0;
    TEST_IGNORE_NODISCARD(this->iter == this->iter);
  }
};

template <class C>
__host__ __device__ void test()
{
  typedef cuda::std::insert_iterator<C> R;
  C c;
  find_members<C> q(c);
  q.test();
  static_assert((cuda::std::is_same<typename R::container_type, C>::value), "");
  static_assert((cuda::std::is_same<typename R::value_type, void>::value), "");
#if _CCCL_STD_VER < 2020
  static_assert((cuda::std::is_same<typename R::difference_type, void>::value), "");
#else // ^^^ _CCCL_STD_VER < 2020 ^^^ / vvv _CCCL_STD_VER >= 2020 vvv
  static_assert((cuda::std::is_same<typename R::difference_type, cuda::std::ptrdiff_t>::value), "");
#endif // _CCCL_STD_VER < 2020
  static_assert((cuda::std::is_same<typename R::reference, void>::value), "");
  static_assert((cuda::std::is_same<typename R::pointer, void>::value), "");
  static_assert((cuda::std::is_same<typename R::iterator_category, cuda::std::output_iterator_tag>::value), "");
}

int main(int, char**)
{
  test<cuda::std::inplace_vector<int, 5>>();

  return 0;
}
