#pragma once

/*
 * Copyright 2026 L. Richard Moore Jr.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

// Ensure.hpp is an optional dependency: if it's available (either as part of this
// checkout or vendored alongside this header), use its throw_if() for input
// validation; otherwise fall back to an equivalent local implementation so this
// header still works standalone.
#if __has_include("commons/Ensure.hpp")
	#include "commons/Ensure.hpp"
#elif __has_include("Ensure.hpp")
	#include "Ensure.hpp"
#else
	template<class T, class... Args>
	constexpr inline void throw_if(bool condition, Args&&... args) {
		if (condition)
			throw T(std::forward<Args>(args)...);
	}
#endif

/**
 * @brief Maps a multi-dimensional coordinate space to a single flat index, and back.
 *
 * Construct a Flattener with the size of each dimension (outermost first); it then converts
 * between a full coordinate tuple and the equivalent flat index, as if iterating a row-major
 * array of that shape. This is useful for backing a dynamically-sized multi-dimensional array
 * with a single flat buffer, or for enumerating a parameter space with a single integer
 * instead of nested loops.
 *
 * All input validation (bad shape, wrong coordinate count, out-of-range coordinate, dimension
 * count overflow) throws rather than aborting, since shapes and coordinates are treated as
 * caller-supplied input rather than internal invariants; see the individual methods below for
 * which exception type each check throws. This also keeps construction and lookups usable
 * from a constexpr context -- though only transiently (e.g. a local variable fully consumed
 * within another constexpr function): a *persisted* constexpr Flattener, such as a
 * namespace-scope variable, isn't possible, since its std::vector members can't outlive the
 * constant evaluation that allocated them.
 *
 * @tparam T Type used for flat indices and the total element count. Defaults to size_t; use a
 * wider or narrower type to match how large a flattened space you need to represent.
 */
template<typename T=size_t>
class Flattener {
public:
	/**
	 * @brief Constructs a Flattener for the given shape.
	 *
	 * @param shape The size of each dimension, outermost first. An empty shape is allowed
	 * and describes a zero-dimensional space of size 1.
	 * @throws std::invalid_argument If any dimension size is 0.
	 * @throws std::overflow_error If the product of the dimension sizes does not fit in T.
	 */
	constexpr Flattener(const std::vector<unsigned int>& shape) {
		initialize(std::rbegin(shape), std::rend(shape));
	}

	/**
	 * @brief Constructs a Flattener for the given shape.
	 *
	 * Equivalent to the std::vector constructor; provided so a shape can be written inline,
	 * e.g. `Flattener<> f({2, 3, 4});`.
	 *
	 * @param shape The size of each dimension, outermost first.
	 * @throws std::invalid_argument If any dimension size is 0.
	 * @throws std::overflow_error If the product of the dimension sizes does not fit in T.
	 */
	constexpr Flattener(std::initializer_list<unsigned int> shape) {
		initialize(std::rbegin(shape), std::rend(shape));
	}

	/**
	 * @brief Returns the coordinate of one dimension for a given flat index.
	 *
	 * @param dimension Which dimension to extract, indexed the same way as the shape passed
	 * to the constructor (0 is the outermost/first dimension).
	 * @param flatIndex A flat index previously produced by flatten(), in `[0, size())`.
	 * @return The coordinate of @p dimension corresponding to @p flatIndex.
	 * @throws std::out_of_range If @p dimension >= dimensions(), or @p flatIndex >= size().
	 */
	constexpr inline unsigned int index(unsigned int dimension, T flatIndex) const {
		throw_if<std::out_of_range>(dimension >= modulators.size(), "Flattener dimension out of range");
		throw_if<std::out_of_range>(flatIndex >= count, "Flattener flat index out of range");
		return indexUnchecked(dimension, flatIndex);
	}

	/**
	 * @brief Returns the full coordinate tuple for a given flat index.
	 *
	 * Equivalent to calling index() for every dimension and collecting the results; see
	 * index() for the per-dimension behavior of the underlying lookup.
	 *
	 * @param flatIndex A flat index previously produced by flatten(), in `[0, size())`.
	 * @return One coordinate per dimension, in the same order as the shape passed to the
	 * constructor.
	 * @throws std::out_of_range If @p flatIndex >= size().
	 */
	constexpr inline std::vector<unsigned int> indices(T flatIndex) const {
		throw_if<std::out_of_range>(flatIndex >= count, "Flattener flat index out of range");
		std::vector<unsigned int> indices;
		indices.reserve(dimensions());
		// Validated once above rather than per dimension via index(): dimension is always
		// in range here by construction of the loop bound, and flatIndex doesn't change.
		for(unsigned int dimension = 0; dimension < dimensions(); ++ dimension)
			indices.push_back(indexUnchecked(dimension, flatIndex));
		return indices;
	}

	/**
	 * @brief Returns the flat index for the given coordinates.
	 *
	 * @param coordinates One coordinate per dimension, in the same order as the shape passed
	 * to the constructor.
	 * @return The flat index corresponding to @p coordinates.
	 * @throws std::invalid_argument If `coordinates.size() != dimensions()`.
	 * @throws std::out_of_range If any coordinate is out of range for its dimension.
	 */
	constexpr inline T flatten(const std::vector<unsigned int>& coordinates) const {
		throw_if<std::invalid_argument>(coordinates.size() != divisors.size(),
			"Flattener coordinate count does not match its dimensionality");
		return flatten<unsigned int>(coordinates.data());
	}

	/**
	 * @brief Returns the flat index for the given coordinates.
	 *
	 * Like the std::vector overload, but reads dimensions() coordinates directly from a
	 * buffer instead of a std::vector; callers are responsible for ensuring @p coordinates
	 * has at least dimensions() elements -- there is no way to validate a coordinate count
	 * from a bare pointer. Prefer the std::vector overload, or the variadic overload below,
	 * unless you already have coordinates in a compatible buffer.
	 *
	 * @tparam CoordT Coordinate element type; may be signed or unsigned.
	 * @param coordinates Pointer to dimensions() coordinates, in the same order as the shape
	 * passed to the constructor.
	 * @return The flat index corresponding to @p coordinates.
	 * @throws std::out_of_range If any coordinate is negative or is out of range for its
	 * dimension.
	 */
	template<class CoordT>
	constexpr inline T flatten(const CoordT* coordinates) const {
		T index = 0;
		for(size_t i = 0; i < divisors.size(); ++i) {
			if constexpr (std::is_signed_v<CoordT>)
				throw_if<std::out_of_range>(coordinates[i] < 0, "Flattener coordinate is negative");
			size_t dimension = divisors.size() - i - 1;
			throw_if<std::out_of_range>(static_cast<unsigned int>(coordinates[i]) >= modulators[dimension],
				"Flattener coordinate out of range for its dimension");
			index += coordinates[i] * divisors[dimension];
		}
		// No final "index >= count" check needed here: once every coordinate has passed the
		// per-dimension bound check above, index is provably < count. (Mixed-radix identity:
		// summing (modulators[i]-1) * divisors[i] over every dimension telescopes to exactly
		// count-1, the maximum representable index.)
		return index;
	}

	/**
	 * @brief Returns the flat index for the given coordinates.
	 *
	 * Variadic sugar for flatten(), e.g. `flattener.flatten(1, 3, 9)` in place of
	 * `flattener.flatten({1, 3, 9})`. Unlike the std::vector overload, this never allocates:
	 * the coordinates live in a plain local array and are handed straight to the
	 * pointer-based flatten() overload above.
	 *
	 * @tparam Args Coordinate argument types; each must be an integral type (signed or
	 * unsigned), and need not all match -- they're combined via std::common_type.
	 * @param coordinates One coordinate per dimension, in the same order as the shape passed
	 * to the constructor.
	 * @return The flat index corresponding to @p coordinates.
	 * @throws std::invalid_argument If `sizeof...(Args) != dimensions()`.
	 * @throws std::out_of_range If any coordinate is negative or is out of range for its
	 * dimension.
	 */
	template<class... Args, class = std::enable_if_t<(std::is_integral_v<Args> && ...)>>
	constexpr inline T flatten(Args... coordinates) const {
		throw_if<std::invalid_argument>(sizeof...(Args) != divisors.size(),
			"Flattener coordinate count does not match its dimensionality");
		if constexpr (sizeof...(Args) == 0) {
			return 0;
		} else {
			using CoordT = std::common_type_t<Args...>;
			CoordT coordinateArray[] = {static_cast<CoordT>(coordinates)...};
			return flatten<CoordT>(coordinateArray);
		}
	}

	/**
	 * @brief Returns the total number of elements in the flattened space, i.e. the product
	 * of all dimension sizes (1 for a zero-dimensional shape).
	 */
	constexpr inline T size() const {
		return count;
	}

	/**
	 * @brief Returns the number of dimensions, i.e. the size of the shape passed to the
	 * constructor.
	 */
	constexpr inline size_t dimensions() const {
		return modulators.size();
	}

private:
	// Unchecked core of index(): callers must guarantee dimension < dimensions() and
	// flatIndex < size() themselves. Used directly by indices(), which validates flatIndex
	// once up front rather than paying for the same check again on every dimension.
	constexpr inline unsigned int indexUnchecked(unsigned int dimension, T flatIndex) const {
		dimension = static_cast<std::uint32_t>(modulators.size()) - dimension - 1;
		return flatIndex / divisors[dimension] % modulators[dimension];
	}

	template<class It>
	constexpr void initialize(It rbegin, It rend) {
		// Compute modulators, divisors, and the overall count of the space
		const size_t dimensionCount = static_cast<size_t>(std::distance(rbegin, rend));
		divisors.reserve(dimensionCount);
		modulators.reserve(dimensionCount);
		count = 1;
		for(auto iterator = rbegin; iterator != rend; ++iterator) {
			divisors.push_back(static_cast<T>(count));
			std::uint32_t dimensionSize = *iterator;
			throw_if<std::invalid_argument>(dimensionSize == 0, "Flattener dimension size cannot be 0");
			modulators.push_back(dimensionSize);
			T countBefore = count;
			count *= dimensionSize;
			// Detect multiplication overflow: since dimensionSize != 0, count / dimensionSize
			// must recover countBefore exactly unless count wrapped around T's range. (A plain
			// "count >= countBefore" comparison is not a reliable overflow check for unsigned
			// multiplication -- the wrapped result can still land above countBefore.)
			throw_if<std::overflow_error>(count / dimensionSize != countBefore,
				"Flattener dimension count overflowed");
		}
	}

	// The total number of iterations required.  Obviously there is a hard limit in T, but even an empty loop
	// take a while to count to that number.
	T count = 0;

	// The divisor and modulus
	std::vector<unsigned int> modulators;
	std::vector<T> divisors;
};
