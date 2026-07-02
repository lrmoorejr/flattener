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

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "Flattener.hpp"

// Regression test for the class being marked constexpr throughout despite being unusable
// in a constant expression: initialize() unconditionally calls the non-constexpr ensure()
// macro, so even a transient, fully-consumed Flattener local inside a constexpr function
// fails to compile. (A *persisted* constexpr Flattener object, e.g. a namespace-scope
// variable, can never work regardless, since its std::vector members would need to escape
// the constant evaluation that allocated them -- that's a fundamental library restriction,
// not a bug in this class. Transient, function-local use is the achievable bar.)
constexpr size_t flattenerConstexprSize() {
	Flattener<> f({2, 3, 4});
	return f.size();
}
static_assert(flattenerConstexprSize() == 24, "Flattener must be usable in a constexpr context");

constexpr unsigned int flattenerConstexprFlatten() {
	Flattener<> f({2, 3});
	return f.flatten({1, 2});
}
static_assert(flattenerConstexprFlatten() == 5, "Flattener::flatten must be usable in a constexpr context");

TEST_CASE( "Flattener size" ) {
	Flattener flattener({1, 2, 3});
	REQUIRE(flattener.size() == 6);
}

TEST_CASE( "Flattener indexes" ) {
	Flattener flattener({2, 3});
	REQUIRE(flattener.index(0, 0) == 0);
	REQUIRE(flattener.index(0, 1) == 0);
	REQUIRE(flattener.index(0, 2) == 0);
	REQUIRE(flattener.index(0, 3) == 1);
	REQUIRE(flattener.index(0, 4) == 1);
	REQUIRE(flattener.index(0, 5) == 1);

	REQUIRE(flattener.index(1, 0) == 0);
	REQUIRE(flattener.index(1, 1) == 1);
	REQUIRE(flattener.index(1, 2) == 2);
	REQUIRE(flattener.index(1, 3) == 0);
	REQUIRE(flattener.index(1, 4) == 1);
	REQUIRE(flattener.index(1, 5) == 2);
}

TEST_CASE( "Flattener flatten()" ) {
	Flattener flattener({2, 3});
	REQUIRE(flattener.flatten({0, 0}) == 0);
	REQUIRE(flattener.flatten({0, 1}) == 1);
	REQUIRE(flattener.flatten({0, 2}) == 2);
	REQUIRE(flattener.flatten({1, 0}) == 3);
	REQUIRE(flattener.flatten({1, 1}) == 4);
	REQUIRE(flattener.flatten({1, 2}) == 5);
}

TEST_CASE( "Flattener::flatten() variadic overload matches the vector overload" ) {
	Flattener flattener({2, 3});
	REQUIRE(flattener.flatten(0, 0) == 0);
	REQUIRE(flattener.flatten(0, 1) == 1);
	REQUIRE(flattener.flatten(0, 2) == 2);
	REQUIRE(flattener.flatten(1, 0) == 3);
	REQUIRE(flattener.flatten(1, 1) == 4);
	REQUIRE(flattener.flatten(1, 2) == 5);

	// Argument types need not match; they're combined via std::common_type.
	int a = 1; unsigned int b = 2;
	REQUIRE(flattener.flatten(a, b) == 5);
}

TEST_CASE( "Flattener::flatten() variadic overload validates like the others" ) {
	Flattener flattener({2, 3});
	REQUIRE_THROWS_AS(flattener.flatten(0, 3), std::out_of_range);          // out of range
	REQUIRE_THROWS_AS(flattener.flatten(0, -1), std::out_of_range);        // negative
	REQUIRE_THROWS_AS(flattener.flatten(0), std::invalid_argument);        // too few
	REQUIRE_THROWS_AS(flattener.flatten(0, 0, 0), std::invalid_argument);  // too many

	Flattener<> zero(std::vector<unsigned int>{});
	REQUIRE(zero.flatten() == 0);
}

// Regression test for a divisor/modulator storage bug: internally the stride for each
// dimension was truncated to std::uint32_t regardless of T, so total index spaces bigger
// than 2^32 silently corrupted round-tripped coordinates even though T=uint64_t was chosen
// specifically to support them, and no assertion ever fired.
TEST_CASE( "Flattener supports index spaces larger than 2^32 when T is wide enough" ) {
	Flattener<uint64_t> flattener({3, 2000000000u, 5u});
	REQUIRE(flattener.size() == 30000000000ull);

	auto flat = flattener.flatten({1u, 1500000000u, 2u});
	auto back = flattener.indices(flat);
	REQUIRE(back[0] == 1u);
	REQUIRE(back[1] == 1500000000u);
	REQUIRE(back[2] == 2u);
}

// Regression test for an unsound overflow check: comparing the post-multiplication count
// against its pre-multiplication value does not reliably detect unsigned multiplication
// wraparound. This shape's true product (4294967301) does not fit in uint32_t, but the old
// check let it through silently instead of reporting the overflow.
TEST_CASE( "Flattener detects dimension-count overflow" ) {
	REQUIRE_THROWS_AS((Flattener<uint32_t>({1431655767u, 3u})), std::overflow_error);
}

// Regression test for missing per-coordinate bounds checking: flatten() only validated the
// combined flat index against the total count, so an out-of-range coordinate for one
// dimension could silently alias to a different, valid-looking coordinate instead of being
// rejected.
TEST_CASE( "Flattener rejects out-of-range coordinates in flatten()" ) {
	Flattener flattener({2, 3});
	REQUIRE_THROWS_AS(flattener.flatten({2, 0}), std::out_of_range);  // first dimension out of range
	REQUIRE_THROWS_AS(flattener.flatten({0, 3}), std::out_of_range);  // second dimension out of range
}

TEST_CASE( "Flattener::flatten() rejects a negative coordinate for signed coordinate types" ) {
	Flattener flattener({2, 3});
	int coordinates[] = {0, -1};
	REQUIRE_THROWS_AS(flattener.flatten<int>(coordinates), std::out_of_range);
}

TEST_CASE( "Flattener::flatten() rejects a coordinate count that doesn't match its dimensionality" ) {
	Flattener flattener({2, 3});
	REQUIRE_THROWS_AS(flattener.flatten({0}), std::invalid_argument);
	REQUIRE_THROWS_AS(flattener.flatten({0, 0, 0}), std::invalid_argument);
}

TEST_CASE( "Flattener::index() rejects an out-of-range dimension" ) {
	Flattener flattener({2, 3});
	REQUIRE_THROWS_AS(flattener.index(2, 0), std::out_of_range);
}

TEST_CASE( "Flattener::index() rejects an out-of-range flat index" ) {
	Flattener flattener({2, 3});
	REQUIRE_THROWS_AS(flattener.index(0, flattener.size()), std::out_of_range);
}

TEST_CASE( "Flattener rejects a zero-sized dimension" ) {
	REQUIRE_THROWS_AS((Flattener({2, 0, 3})), std::invalid_argument);
}

// Companion to the overflow-detection regression test above: makes sure the fix didn't turn
// the check into a false positive for a shape that legitimately fits within T.
TEST_CASE( "Flattener does not reject a large shape that fits exactly" ) {
	// True product is 4000000000, which fits within uint32_t (max 4294967295).
	Flattener<uint32_t> flattener({2u, 2000000000u});
	REQUIRE(flattener.size() == 4000000000u);
}

TEST_CASE( "Flattener::dimensions() reports the number of dimensions" ) {
	Flattener flattener({2, 3, 4});
	REQUIRE(flattener.dimensions() == 3);
}

TEST_CASE( "Flattener supports a zero-dimensional (empty) shape" ) {
	Flattener<> flattener(std::vector<unsigned int>{});
	REQUIRE(flattener.dimensions() == 0);
	REQUIRE(flattener.size() == 1);
	REQUIRE(flattener.flatten({}) == 0);
}

// General correctness net: every flat index in the space should decode via indices() to a
// coordinate set that re-encodes, via flatten(), back to the same flat index.
TEST_CASE( "Flattener flatten()/indices() round-trip over the entire space" ) {
	Flattener flattener({2, 1, 3, 4});
	for(size_t flat = 0; flat < flattener.size(); ++flat) {
		auto coordinates = flattener.indices(flat);
		REQUIRE(flattener.flatten(coordinates) == flat);
	}
}

