# Flattener.hpp

[API docs](https://lrmoorejr.github.io/flattener/)

A small header for mapping a multi-dimensional coordinate space to a single flat index, and
back -- for backing a dynamically-sized multi-dimensional array with one flat buffer, or for
enumerating a parameter space with a single integer instead of nested loops.

I originally wrote this to convert between flat job indices and multi-dimensional coordinates
for a thread-pool work dispatcher, where the shape of the space isn't known until runtime. It
turned out to be useful on its own wherever you need dynamically-sized multi-dimensional indexing
without nested loops or a compile-time-fixed shape.

```cpp
#include "Flattener.hpp"

Flattener<> f({2, 3, 4});     // a 2x3x4 coordinate space

f.size();                     // 24
f.dimensions();                // 3

f.flatten(1, 2, 3);           // -> 23  (variadic, allocation-free)
f.flatten({1, 2, 3});         // -> 23  (equivalent, via std::vector)

f.indices(23);                // -> {1, 2, 3}
f.index(0, 23);               // -> 1   (just the first dimension)
```

## Requirements

- C++17 or later (uses fold expressions and `if constexpr`)
- Header-only -- copy `Flattener.hpp` into your project and `#include` it
- Optional: [`Ensure.hpp`](https://github.com/lrmoorejr/ensure) for its `throw_if()` helper; if
  it's not present, `Flattener` falls back to an equivalent local implementation (see below)

## API

| Call | Behavior |
|---|---|
| `Flattener<T=size_t>(shape)` | Constructs for the given shape -- a `std::vector<unsigned int>` or brace-init list of dimension sizes, outermost first. |
| `size()` | Total element count: the product of all dimension sizes (1 for an empty/zero-dimensional shape). |
| `dimensions()` | The number of dimensions. |
| `flatten(coord0, coord1, ...)` | Coordinates to flat index, as separate arguments. Allocation-free. |
| `flatten({coord0, coord1, ...})` | Same, taking a `std::vector<unsigned int>`. |
| `flatten(pointer)` | Same, reading `dimensions()` coordinates from a raw buffer. |
| `index(dimension, flatIndex)` | Flat index back to the coordinate of just one dimension. |
| `indices(flatIndex)` | Flat index back to the full coordinate tuple, as a `std::vector<unsigned int>`. |

`T` (default `size_t`) is the type used for flat indices and the total element count; use a
narrower or wider type to match how large a flattened space you actually need.

### Validation

Every constructor and lookup validates its input by throwing rather than aborting, since shapes
and coordinates are ordinary caller-supplied data, not internal invariants:

- `std::invalid_argument` -- a dimension size of 0, or a coordinate count that doesn't match
  `dimensions()`
- `std::out_of_range` -- a coordinate outside its dimension's range (including negative, for
  signed coordinate types), or a flat index / dimension index outside its valid range
- `std::overflow_error` -- the product of the dimension sizes doesn't fit in `T`

These checks are always active, independent of `NDEBUG`.

### constexpr

Construction and lookups are usable from a constexpr context, but only *transiently* -- e.g. a
local `Flattener` fully consumed inside another `constexpr` function to compute a compile-time
constant. A *persisted* constexpr `Flattener` (a namespace-scope variable, for instance) isn't
possible, since its internal `std::vector`s can't outlive the constant evaluation that allocated
them. That's a fundamental restriction of using `std::vector` at all, not something specific to
this class.

```cpp
constexpr size_t total = [] {
    Flattener<> f({2, 3, 4});
    return f.size();
}();
static_assert(total == 24);
```

### Ensure.hpp fallback

`Flattener` uses one helper from [`Ensure.hpp`](https://github.com/lrmoorejr/ensure):
`throw_if<ExceptionType>(condition, args...)`, which throws `ExceptionType(args...)` when
`condition` is true. If `Ensure.hpp` is available -- either checked out alongside
`Flattener.hpp`, or reachable as `commons/Ensure.hpp` -- that's what gets used. Otherwise
`Flattener.hpp` defines an equivalent `throw_if` itself, so it works standalone either way.

## License

Apache License 2.0 -- see [LICENSE](https://github.com/lrmoorejr/flattener/blob/main/LICENSE).
