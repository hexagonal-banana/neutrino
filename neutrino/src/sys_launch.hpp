// Native launch-marshalling primitives for the opt-in system-scope tracer.
// These helpers neither intercept APIs nor execute or retry CUDA work.
#pragma once

#include <cuda.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace neutrino::sys {

class ArgumentError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct Parameter {
    std::size_t offset;
    std::size_t size;
    std::size_t alignment;
};

class ParameterLayout {
public:
    ParameterLayout(std::vector<Parameter> parameters, std::size_t original_size,
                    std::size_t device_limit, std::size_t packed_min = std::numeric_limits<std::size_t>::max())
        : parameters_(std::move(parameters)), original_size_(original_size),
          packed_min_(packed_min == std::numeric_limits<std::size_t>::max() ? original_size : packed_min) {
        if (packed_min_ > original_size || (packed_min_ != original_size &&
            (!packed_min_ || parameters_.size() != 1 || parameters_[0].offset != 0)))
            throw ArgumentError("Invalid packed-prefix contract");
        if (device_limit == 0 || original_size > device_limit ||
            original_size > std::numeric_limits<std::size_t>::max() - 15)
            throw ArgumentError("Invalid device parameter-space limit/size");
        std::size_t end = 0;
        for (const auto& parameter : parameters_) {
            const auto alignment = parameter.alignment;
            if (!alignment || alignment > 128 || (alignment & (alignment - 1)) || !parameter.size ||
                end > std::numeric_limits<std::size_t>::max() - (alignment - 1))
                throw ArgumentError("Invalid parameter size/alignment");
            const auto offset = (end + alignment - 1) & ~(alignment - 1);
            if (parameter.offset != offset || offset > original_size || parameter.size > original_size - offset)
                throw ArgumentError("Parameter offsets disagree with original PTX layout");
            end = offset + parameter.size;
        }
        if (end != original_size)
            throw ArgumentError("Original parameter extent disagrees with declarations");
        context_offset_ = (original_size + 7) & ~std::size_t(7);
        if (context_offset_ > device_limit || device_limit - context_offset_ < sizeof(std::uint64_t))
            throw ArgumentError("Hidden context exceeds device parameter space");
    }

    const std::vector<Parameter>& parameters() const { return parameters_; }
    std::size_t original_size() const { return original_size_; }
    std::size_t packed_min() const { return packed_min_; }
    std::size_t context_offset() const { return context_offset_; }
    std::size_t extended_size() const { return context_offset_ + sizeof(std::uint64_t); }

private:
    std::vector<Parameter> parameters_;
    std::size_t original_size_, packed_min_, context_offset_;
};

class PreparedArguments {
    struct AlignedFree {
        void operator()(std::byte* value) const { ::operator delete(value, std::align_val_t(128)); }
    };
public:
    // Layout must belong to the exact original kernel in the replacement image.
    // Ownership is stable through submission; CUDA copies these arguments during
    // the launch API call, not when the asynchronously scheduled kernel starts.
    PreparedArguments(const ParameterLayout& layout, void** parameters, void** extra,
                      std::uint64_t device_context)
        : data_(static_cast<std::byte*>(::operator new(layout.extended_size(), std::align_val_t(128)))),
          size_(layout.extended_size()), packed_(extra != nullptr) {
        if (parameters && extra)
            throw ArgumentError("Both kernelParams and extra were supplied");
        std::memset(data_.get(), 0, size_);
        if (extra) {
            const void* original = nullptr;
            std::size_t original_size = 0;
            bool have_pointer = false, have_size = false, terminated = false;
            // CUDA defines exactly two key/value pairs and an END token. Bound
            // parsing so malformed lists never become an unbounded scan.
            for (std::size_t index = 0; index <= 4; index += 2) {
                const auto token = extra[index];
                if (token == CU_LAUNCH_PARAM_END) {
                    terminated = true;
                    break;
                }
                if (index == 4)
                    throw ArgumentError("Unterminated/oversized CUDA extra list");
                if (token == CU_LAUNCH_PARAM_BUFFER_POINTER && !have_pointer) {
                    original = extra[index + 1];
                    have_pointer = true;
                } else if (token == CU_LAUNCH_PARAM_BUFFER_SIZE && !have_size && extra[index + 1]) {
                    std::memcpy(&original_size, extra[index + 1], sizeof(original_size));
                    have_size = true;
                } else {
                    throw ArgumentError("Unknown, duplicate, or null CUDA extra token");
                }
            }
            if (!terminated || !have_pointer || !have_size ||
                original_size < layout.packed_min() || original_size > layout.context_offset() ||
                (original_size && !original))
                throw ArgumentError("Packed buffer disagrees with original parameter layout");
            if (original_size)
                std::memcpy(data_.get(), original, original_size);
        } else {
            if (!parameters && !layout.parameters().empty())
                throw ArgumentError("Missing original kernel parameters");
            for (std::size_t index = 0; index < layout.parameters().size(); ++index) {
                const auto& parameter = layout.parameters()[index];
                if (!parameters[index])
                    throw ArgumentError("Null original parameter pointer");
                std::memcpy(data_.get() + parameter.offset, parameters[index], parameter.size);
                pointers_.push_back(data_.get() + parameter.offset);
            }
            pointers_.push_back(data_.get() + layout.context_offset());
        }
        std::memcpy(data_.get() + layout.context_offset(), &device_context, sizeof(device_context));
        extra_[0] = CU_LAUNCH_PARAM_BUFFER_POINTER;
        extra_[1] = data_.get();
        extra_[2] = CU_LAUNCH_PARAM_BUFFER_SIZE;
        extra_[3] = &size_;
        extra_[4] = CU_LAUNCH_PARAM_END;
    }

    PreparedArguments(const PreparedArguments&) = delete;
    PreparedArguments& operator=(const PreparedArguments&) = delete;
    PreparedArguments(PreparedArguments&&) = delete;
    PreparedArguments& operator=(PreparedArguments&&) = delete;
    void** parameters() { return packed_ ? nullptr : pointers_.data(); }
    void** extra() { return packed_ ? extra_ : nullptr; }
    const std::byte* data() const { return data_.get(); }
    std::size_t size() const { return size_; }

private:
    std::unique_ptr<std::byte, AlignedFree> data_;
    std::size_t size_;
    bool packed_;
    std::vector<void*> pointers_;
    void* extra_[5]{};
};

class OwnedLaunchConfig {
public:
    explicit OwnedLaunchConfig(const CUlaunchConfig& original) : config_(original) {
        if (!original.gridDimX || !original.gridDimY || !original.gridDimZ ||
            !original.blockDimX || !original.blockDimY || !original.blockDimZ ||
            (original.numAttrs && !original.attrs))
            throw ArgumentError("Invalid extended launch configuration");
        if (original.numAttrs)
            attributes_.assign(original.attrs, original.attrs + original.numAttrs);
        config_.attrs = attributes_.empty() ? nullptr : attributes_.data();
    }
    OwnedLaunchConfig(const OwnedLaunchConfig&) = delete;
    OwnedLaunchConfig& operator=(const OwnedLaunchConfig&) = delete;
    OwnedLaunchConfig(OwnedLaunchConfig&&) = delete;
    OwnedLaunchConfig& operator=(OwnedLaunchConfig&&) = delete;
    const CUlaunchConfig* get() const { return &config_; }

private:
    CUlaunchConfig config_;
    std::vector<CUlaunchAttribute> attributes_;
};

// The outer runtime wrapper owns correlation and preparation; a nested driver
// wrapper must submit its already-prepared call once without reinstrumenting it.
class SubmissionScope {
public:
    SubmissionScope() : outermost_(depth_ == 0) {
        if (depth_ == std::numeric_limits<std::size_t>::max())
            throw ArgumentError("Submission recursion depth overflow");
        ++depth_;
    }
    ~SubmissionScope() { --depth_; }
    SubmissionScope(const SubmissionScope&) = delete;
    SubmissionScope& operator=(const SubmissionScope&) = delete;
    bool outermost() const { return outermost_; }
private:
    bool outermost_;
    static inline thread_local std::size_t depth_ = 0;
};

// Exposes only the currently submitted selected launch to read-only CUPTI callbacks.
class ActivityCorrelationScope {
public:
    explicit ActivityCorrelationScope(std::uint64_t ordinal) : previous_(current_) { current_ = ordinal; }
    ~ActivityCorrelationScope() { current_ = previous_; }
    ActivityCorrelationScope(const ActivityCorrelationScope&) = delete;
    ActivityCorrelationScope& operator=(const ActivityCorrelationScope&) = delete;
    static std::uint64_t current() { return current_; }
private:
    std::uint64_t previous_;
    static inline thread_local std::uint64_t current_ = 0;
};

}  // namespace neutrino::sys
