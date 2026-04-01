// === AUDIT STATUS ===
// internal:    { status: Complete, auditors: [Nishat], commit: 94f596f8b3bbbc216f9ad7dc33253256141156b2 }
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

#include "polynomial.hpp"
#include "barretenberg/common/assert.hpp"
#include "barretenberg/common/bb_bench.hpp"
#include "barretenberg/common/thread.hpp"
#include "barretenberg/numeric/bitop/get_msb.hpp"
#include "barretenberg/numeric/bitop/pow.hpp"
#include "barretenberg/polynomials/backing_memory.hpp"
#include "barretenberg/polynomials/shared_shifted_virtual_zeroes_array.hpp"
#include "polynomial_arithmetic.hpp"
#include <cstddef>
#include <fcntl.h>
#include <list>
#include <memory>
#include <mutex>
#include <span>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unordered_map>
#include <utility>

namespace bb {

// Note: This function is pretty gnarly, but we try to make it the only function that deals
// with copying polynomials. It should be scrutinized thusly.
template <typename Fr>
SharedShiftedVirtualZeroesArray<Fr> _clone(const SharedShiftedVirtualZeroesArray<Fr>& array,
                                           size_t right_expansion = 0,
                                           size_t left_expansion = 0)
{
    size_t expanded_size = array.size() + right_expansion + left_expansion;
    BackingMemory<Fr> backing_clone = BackingMemory<Fr>::allocate(expanded_size);
    // zero any left extensions to the array
    memset(static_cast<void*>(backing_clone.raw_data), 0, sizeof(Fr) * left_expansion);
    // copy our cloned array over
    memcpy(static_cast<void*>(backing_clone.raw_data + left_expansion),
           static_cast<const void*>(array.data()),
           sizeof(Fr) * array.size());
    // zero any right extensions to the array
    memset(static_cast<void*>(backing_clone.raw_data + left_expansion + array.size()), 0, sizeof(Fr) * right_expansion);
    return {
        array.start_ - left_expansion, array.end_ + right_expansion, array.virtual_size_, std::move(backing_clone)
    };
}

template <typename Fr>
void Polynomial<Fr>::allocate_backing_memory(size_t size, size_t virtual_size, size_t start_index)
{
    BB_BENCH_NAME("Polynomial::allocate_backing_memory");
    BB_ASSERT_LTE(start_index + size, virtual_size);
    coefficients_ = SharedShiftedVirtualZeroesArray<Fr>{
        start_index,        /* start index, used for shifted polynomials and offset 'islands' of non-zeroes */
        size + start_index, /* end index, actual memory used is (end - start) */
        virtual_size,       /* virtual size, i.e. until what size do we conceptually have zeroes */
        BackingMemory<Fr>::allocate(size)
    };
}

/**
 * Constructors / Destructors
 **/

/**
 * @brief Initialize a Polynomial to size 'size', zeroing memory.
 *
 * @param size The size of the polynomial.
 */
template <typename Fr> Polynomial<Fr>::Polynomial(size_t size, size_t virtual_size, size_t start_index)
{
    BB_BENCH_NAME("Polynomial::Polynomial(size_t, size_t, size_t)");
    allocate_backing_memory(size, virtual_size, start_index);

    parallel_for([&](const ThreadChunk& chunk) {
        auto range = chunk.range(size);
        if (!range.empty()) {
            size_t start = *range.begin();
            size_t range_size = range.size();
            BB_ASSERT(start < size || size == 0);
            BB_ASSERT_LTE((start + range_size), size);
            memset(static_cast<void*>(coefficients_.data() + start), 0, sizeof(Fr) * range_size);
        }
    });
}

/**
 * @brief Initialize a Polynomial to size 'size'.
 * Important: This does NOT zero memory.
 *
 * @param size The initial size of the polynomial.
 * @param flag Signals that we do not zero memory.
 */
template <typename Fr>
Polynomial<Fr>::Polynomial(size_t size, size_t virtual_size, size_t start_index, [[maybe_unused]] DontZeroMemory flag)
{
    allocate_backing_memory(size, virtual_size, start_index);
}

template <typename Fr>
Polynomial<Fr>::Polynomial(const Polynomial<Fr>& other)
    : Polynomial<Fr>(other, other.size())
{}

// fully copying "expensive" constructor
template <typename Fr> Polynomial<Fr>::Polynomial(const Polynomial<Fr>& other, const size_t target_size)
{
    BB_ASSERT_LTE(other.size(), target_size);
    coefficients_ = _clone(other.coefficients_, target_size - other.size());
}

// interpolation constructor
template <typename Fr>
Polynomial<Fr>::Polynomial(std::span<const Fr> interpolation_points,
                           std::span<const Fr> evaluations,
                           size_t virtual_size)
    : Polynomial(interpolation_points.size(), virtual_size)
{
    BB_ASSERT_GT(coefficients_.size(), static_cast<size_t>(0));

    polynomial_arithmetic::compute_efficient_interpolation(
        evaluations.data(), coefficients_.data(), interpolation_points.data(), coefficients_.size());
}

template <typename Fr> Polynomial<Fr>::Polynomial(std::span<const Fr> coefficients, size_t virtual_size)
{
    allocate_backing_memory(coefficients.size(), virtual_size, 0);

    memcpy(static_cast<void*>(data()), static_cast<const void*>(coefficients.data()), sizeof(Fr) * coefficients.size());
}

// Assignments

// full copy "expensive" assignment
template <typename Fr> Polynomial<Fr>& Polynomial<Fr>::operator=(const Polynomial<Fr>& other)
{
    if (this == &other) {
        return *this;
    }
    coefficients_ = _clone(other.coefficients_);
    return *this;
}

template <typename Fr> Polynomial<Fr> Polynomial<Fr>::share() const
{
    Polynomial p;
    p.coefficients_ = coefficients_;
    return p;
}

template <typename Fr> bool Polynomial<Fr>::operator==(Polynomial const& rhs) const
{
    // If either is empty, both must be
    if (is_empty() || rhs.is_empty()) {
        return is_empty() && rhs.is_empty();
    }
    // Size must agree
    if (virtual_size() != rhs.virtual_size()) {
        return false;
    }
    // Each coefficient must agree
    for (size_t i = std::min(coefficients_.start_, rhs.coefficients_.start_);
         i < std::max(coefficients_.end_, rhs.coefficients_.end_);
         i++) {
        if (coefficients_.get(i) != rhs.coefficients_.get(i)) {
            return false;
        }
    }
    return true;
}

template <typename Fr> Polynomial<Fr>& Polynomial<Fr>::operator+=(PolynomialSpan<const Fr> other)
{
    BB_ASSERT_LTE(start_index(), other.start_index);
    BB_ASSERT_GTE(end_index(), other.end_index());
    parallel_for([&](const ThreadChunk& chunk) {
        for (size_t offset : chunk.range(other.size())) {
            size_t i = offset + other.start_index;
            at(i) += other[i];
        }
    });
    return *this;
}

template <typename Fr> Fr Polynomial<Fr>::evaluate(const Fr& z) const
{
    // Evaluate only the backing data; virtual zeroes beyond backing contribute nothing.
    // When start_index > 0, multiply by z^start_index to account for the offset.
    Fr result = polynomial_arithmetic::evaluate(data(), z, size());
    if (start_index() > 0) {
        result *= z.pow(start_index());
    }
    return result;
}

template <typename Fr> Fr Polynomial<Fr>::evaluate_mle(std::span<const Fr> evaluation_points, bool shift) const
{
    return _evaluate_mle(evaluation_points, coefficients_, shift);
}

template <typename Fr> Polynomial<Fr>& Polynomial<Fr>::operator-=(PolynomialSpan<const Fr> other)
{
    BB_ASSERT_LTE(start_index(), other.start_index);
    BB_ASSERT_GTE(end_index(), other.end_index());
    parallel_for([&](const ThreadChunk& chunk) {
        for (size_t offset : chunk.range(other.size())) {
            size_t i = offset + other.start_index;
            at(i) -= other[i];
        }
    });
    return *this;
}

template <typename Fr> Polynomial<Fr>& Polynomial<Fr>::operator*=(const Fr& scaling_factor)
{
    parallel_for([scaling_factor, this](const ThreadChunk& chunk) { multiply_chunk(chunk, scaling_factor); });
    return *this;
}

template <typename Fr> void Polynomial<Fr>::multiply_chunk(const ThreadChunk& chunk, const Fr& scaling_factor)
{
    for (size_t i : chunk.range(size())) {
        data()[i] *= scaling_factor;
    }
}

template <typename Fr> Polynomial<Fr> Polynomial<Fr>::create_non_parallel_zero_init(size_t size, size_t virtual_size)
{
    Polynomial p(size, virtual_size, Polynomial<Fr>::DontZeroMemory::FLAG);
    memset(static_cast<void*>(p.coefficients_.data()), 0, sizeof(Fr) * size);
    return p;
}

template <typename Fr> void Polynomial<Fr>::shrink_end_index(const size_t new_end_index)
{
    BB_ASSERT_LTE(new_end_index, end_index());
    coefficients_.end_ = new_end_index;
}

template <typename Fr> Polynomial<Fr> Polynomial<Fr>::full() const
{
    Polynomial result = *this;
    // Make 0..virtual_size usable
    result.coefficients_ = _clone(coefficients_, virtual_size() - end_index(), start_index());
    return result;
}

template <typename Fr> void Polynomial<Fr>::add_scaled(PolynomialSpan<const Fr> other, const Fr& scaling_factor)
{
    BB_ASSERT_LTE(start_index(), other.start_index);
    BB_ASSERT_GTE(end_index(), other.end_index());
    parallel_for(
        [&other, scaling_factor, this](const ThreadChunk& chunk) { add_scaled_chunk(chunk, other, scaling_factor); });
}

template <typename Fr>
void Polynomial<Fr>::add_scaled_chunk(const ThreadChunk& chunk,
                                      PolynomialSpan<const Fr> other,
                                      const Fr& scaling_factor)
{
    // Iterate over the chunk of the other polynomial's range
    for (size_t offset : chunk.range(other.size())) {
        size_t index = other.start_index + offset;
        at(index) += scaling_factor * other[index];
    }
}

template <typename Fr> Polynomial<Fr> Polynomial<Fr>::shifted() const
{
    BB_ASSERT_GTE(coefficients_.start_, static_cast<size_t>(1));
    Polynomial result;
    result.coefficients_ = coefficients_;
    result.coefficients_.start_ -= 1;
    result.coefficients_.end_ -= 1;
    return result;
}

template <typename Fr> Polynomial<Fr> Polynomial<Fr>::reverse() const
{
    const size_t end_index = this->end_index();
    const size_t start_index = this->start_index();
    const size_t poly_size = this->size();
    Polynomial reversed(/*size=*/poly_size, /*virtual_size=*/end_index);
    for (size_t idx = end_index; idx > start_index; --idx) {
        reversed.at(end_index - idx) = this->at(idx - 1);
    }
    return reversed;
}

template <typename Fr> void Polynomial<Fr>::serialize_to_file(const std::string& path) const
{
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw_or_abort("Failed to open file for polynomial serialization: " + path);
    }
    uint64_t si = start_index();
    uint64_t ei = end_index();
    uint64_t vs = virtual_size();
    file.write(reinterpret_cast<const char*>(&si), sizeof(si));
    file.write(reinterpret_cast<const char*>(&ei), sizeof(ei));
    file.write(reinterpret_cast<const char*>(&vs), sizeof(vs));
    // Write the raw coefficient data (size() elements starting from data())
    file.write(reinterpret_cast<const char*>(data()), static_cast<std::streamsize>(size() * sizeof(Fr)));
}

template <typename Fr> Polynomial<Fr> Polynomial<Fr>::deserialize_from_file(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw_or_abort("Failed to open file for polynomial deserialization: " + path);
    }
    uint64_t si = 0;
    uint64_t ei = 0;
    uint64_t vs = 0;
    file.read(reinterpret_cast<char*>(&si), sizeof(si));
    file.read(reinterpret_cast<char*>(&ei), sizeof(ei));
    file.read(reinterpret_cast<char*>(&vs), sizeof(vs));
    size_t actual_size = ei - si;
    Polynomial result(actual_size, vs, si, DontZeroMemory::FLAG);
    file.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(actual_size * sizeof(Fr)));
    return result;
}

template <typename Fr>
Polynomial<Fr> Polynomial<Fr>::deserialize_range_from_file(const std::string& path, size_t range_start, size_t range_end)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw_or_abort("Failed to open file for polynomial range deserialization: " + path);
    }
    uint64_t si = 0;
    uint64_t ei = 0;
    uint64_t vs = 0;
    file.read(reinterpret_cast<char*>(&si), sizeof(si));
    file.read(reinterpret_cast<char*>(&ei), sizeof(ei));
    file.read(reinterpret_cast<char*>(&vs), sizeof(vs));

    // Intersect requested range with actual data range
    size_t load_start = std::max(range_start, static_cast<size_t>(si));
    size_t load_end = std::min(range_end, static_cast<size_t>(ei));
    if (load_start >= load_end) {
        return Polynomial(); // no data in range
    }

    size_t count = load_end - load_start;
    Polynomial result(count, vs, load_start, DontZeroMemory::FLAG);

    // Seek past header + skip elements before load_start
    auto file_offset = static_cast<std::streamoff>(3 * sizeof(uint64_t) + (load_start - si) * sizeof(Fr));
    file.seekg(file_offset);
    file.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(count * sizeof(Fr)));
    return result;
}

template <typename Fr> Polynomial<Fr> Polynomial<Fr>::mmap_from_file(const std::string& path)
{
#ifdef __wasm__
    // WASM has no mmap — fall back to regular deserialization
    return deserialize_from_file(path);
#else
    // Read header via ifstream (avoids POSIX read() name collision)
    std::ifstream hdr(path, std::ios::binary);
    if (!hdr) {
        throw_or_abort("Failed to open file for mmap: " + path);
    }
    uint64_t si = 0, ei = 0, vs = 0;
    hdr.read(reinterpret_cast<char*>(&si), sizeof(si));
    hdr.read(reinterpret_cast<char*>(&ei), sizeof(ei));
    hdr.read(reinterpret_cast<char*>(&vs), sizeof(vs));
    hdr.close();

    size_t actual_size = ei - si;
    if (actual_size == 0) {
        return Polynomial();
    }

    // mmap the entire file (header + data)
    constexpr size_t header_bytes = 3 * sizeof(uint64_t);
    size_t data_bytes = actual_size * sizeof(Fr);
    size_t total_bytes = header_bytes + data_bytes;

    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw_or_abort("Failed to open file for mmap: " + path);
    }

    void* addr = mmap(nullptr, total_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        close(fd);
        throw_or_abort("mmap failed for: " + path);
    }

    // Advise OS for sequential access (enables aggressive readahead)
    madvise(addr, total_bytes, MADV_SEQUENTIAL);

    // Data starts after the header
    Fr* data_ptr = reinterpret_cast<Fr*>(static_cast<char*>(addr) + header_bytes);

    // Build BackingMemory with FileBackedData for automatic munmap cleanup
    // Set filename="" so destructor doesn't delete the external file
    auto file_data = std::make_shared<typename BackingMemory<Fr>::FileBackedData>();
    file_data->file_size = total_bytes;
    file_data->filename = ""; // Don't delete external file
    file_data->fd = fd;
    file_data->raw_data_ptr = static_cast<Fr*>(addr); // munmap needs original addr

    // Construct the Polynomial by directly setting up the internal array
    Polynomial result;
    result.coefficients_.start_ = si;
    result.coefficients_.end_ = ei;
    result.coefficients_.virtual_size_ = vs;
    result.coefficients_.backing_memory_.raw_data = data_ptr;
    result.coefficients_.backing_memory_.file_backed = std::move(file_data);

    return result;
#endif
}

template class Polynomial<bb::fr>;
template class Polynomial<grumpkin::fr>;
} // namespace bb
