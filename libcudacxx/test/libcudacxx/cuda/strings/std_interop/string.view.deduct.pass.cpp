// <cuda/std/string_view>

//  constexpr basic_string_view(std::basic_string_view sv);

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include <string_view>

#include "literal.h"

template <class CharT>
constexpr void test_with_default_type_traits()
{
  const CharT* str = TEST_STRLIT(CharT, "some text");

  std::basic_string_view host_sv{str};
  cuda::std::basic_string_view cuda_sv{host_sv};

  static_assert(cuda::std::is_same_v<typename decltype(cuda_sv)::value_type, CharT>);
  static_assert(cuda::std::is_same_v<typename decltype(cuda_sv)::traits_type, cuda::std::char_traits<CharT>>);

  assert(cuda_sv.data() == host_sv.data());
  assert(cuda_sv.size() == host_sv.size());
}

template <class CharT>
struct custom_type_traits
    : private std::char_traits<CharT>
    , private cuda::std::char_traits<CharT>
{
  using cuda::std::char_traits<CharT>::char_type;
  using cuda::std::char_traits<CharT>::int_type;
  using std::char_traits<CharT>::pos_type;
  using std::char_traits<CharT>::off_type;
  using std::char_traits<CharT>::state_type;

  using cuda::std::char_traits<CharT>::assign;
  using cuda::std::char_traits<CharT>::eq;
  using cuda::std::char_traits<CharT>::lt;
  using cuda::std::char_traits<CharT>::compare;
  using cuda::std::char_traits<CharT>::length;
  using cuda::std::char_traits<CharT>::find;
  using cuda::std::char_traits<CharT>::move;
  using cuda::std::char_traits<CharT>::copy;
  using cuda::std::char_traits<CharT>::to_char_type;
  using cuda::std::char_traits<CharT>::to_int_type;
  using cuda::std::char_traits<CharT>::eq_int_type;
  using std::char_traits<CharT>::eof;
  using std::char_traits<CharT>::not_eof;
};

template <class CharT>
constexpr void test_with_custom_type_traits()
{
  const CharT* str = TEST_STRLIT(CharT, "some text");

  std::basic_string_view<CharT, custom_type_traits<CharT>> host_sv{str};
  cuda::std::basic_string_view cuda_sv{host_sv};

  static_assert(cuda::std::is_same_v<typename decltype(cuda_sv)::value_type, typename decltype(host_sv)::value_type>);
  static_assert(cuda::std::is_same_v<typename decltype(cuda_sv)::traits_type, custom_type_traits<CharT>>);

  assert(cuda_sv.data() == host_sv.data());
  assert(cuda_sv.size() == host_sv.size());
}

template <class CharT>
constexpr void test_type()
{
  test_with_default_type_traits<CharT>();
  test_with_custom_type_traits<CharT>();
}

constexpr bool test()
{
  test_type<char>();
#if _CCCL_HAS_CHAR8_T()
  test_type<char8_t>();
#endif // _CCCL_HAS_CHAR8_T()
  test_type<char16_t>();
  test_type<char32_t>();
#if _CCCL_HAS_WCHAR_T()
  test_type<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

static_assert(test());

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
