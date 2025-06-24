//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#define CCCL_DISABLE_EXCEPTIONS

#include <cuda/std/__cccl/exceptions.h>
#include <cuda/std/cassert>

#include <nv/target>

struct ExceptionBase
{
  int value;
};

struct Exception : ExceptionBase
{};

__host__ __device__ void test()
{
  _CCCL_TRY
  {
    assert(true);
  }
  _CCCL_CATCH (const Exception & e)
  {
    assert(e.value == 0);
    assert(true);
  }
  _CCCL_CATCH (const ExceptionBase & e)
  {
    assert(e.value == 0);
    assert(false);
  }
  _CCCL_CATCH_ALL
  {
    assert(false);
  }
}

int main(int, char**)
{
  test();
  return 0;
}
