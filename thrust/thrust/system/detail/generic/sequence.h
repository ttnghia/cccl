/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/iterator/iterator_traits.h>
#include <thrust/tabulate.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
namespace detail
{
template <typename T>
struct compute_sequence_value
{
  T init;
  T step;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE T operator()(std::size_t i) const
  {
    if constexpr (::cuda::std::is_arithmetic_v<T>)
    {
      return init + step * static_cast<T>(i);
    }
    else
    {
      return init + step * i;
    }
  }
};
} // namespace detail

template <typename DerivedPolicy, typename ForwardIterator, typename T = thrust::detail::it_value_t<ForwardIterator>>
_CCCL_HOST_DEVICE void sequence(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  T init = T{},
  T step = T{1})
{
  thrust::tabulate(
    exec, first, last, detail::compute_sequence_value<T>{::cuda::std::move(init), ::cuda::std::move(step)});
}
} // namespace system::detail::generic
THRUST_NAMESPACE_END
