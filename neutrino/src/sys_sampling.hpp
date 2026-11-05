// Kernel policies and block-local compact thread coordinates; no CUDA calls.
#pragma once
#include "sys_launch.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <utility>

namespace neutrino::sys {
constexpr unsigned SAMPLING_CONTEXT_ABI = 2;
inline unsigned sampling_mode(const std::string& name) {
    if (name == "all") return 0;
    if (name == "warp") return 1;
    if (name == "block") return 2;
    throw ArgumentError("Unknown capture sampling");
}
inline bool kernel_glob(const std::string& pattern, const std::string& text) {
    std::size_t p = 0, t = 0, star = std::string::npos, retry = 0;
    while (t < text.size()) {
        if (p < pattern.size() && pattern[p] == '*') { star = p++; retry = t; }
        else if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == text[t])) { ++p; ++t; }
        else if (star != std::string::npos) { t = ++retry; p = star + 1; }
        else return false;
    }
    while (p < pattern.size() && pattern[p] == '*') ++p;
    return p == pattern.size();
}
inline nlohmann::json validate_policy(nlohmann::json value) {
    using J = nlohmann::json;
    if (!value.is_object()) throw ArgumentError("Invalid capture policy");
    for (auto it = value.begin(); it != value.end(); ++it)
        if (it.key() != "schema" && it.key() != "default" && it.key() != "rules")
            throw ArgumentError("Unknown capture policy field");
    if (!value.contains("schema") || !value.at("schema").is_number_integer() || value.at("schema") != 1)
        throw ArgumentError("Unsupported capture policy schema");
    if (!value.contains("default")) value["default"] = "warp";
    if (!value.contains("rules")) value["rules"] = J::array();
    sampling_mode(value.at("default").get<std::string>());
    if (!value.at("rules").is_array()) throw ArgumentError("Invalid capture rules");
    for (const auto& rule : value.at("rules")) {
        if (!rule.is_object() || !rule.contains("kernel_glob") || !rule.contains("sampling"))
            throw ArgumentError("Invalid capture rule");
        for (auto it = rule.begin(); it != rule.end(); ++it)
            if (it.key() != "kernel_glob" && it.key() != "library" && it.key() != "sampling")
                throw ArgumentError("Unknown capture rule field");
        sampling_mode(rule.at("sampling").get<std::string>());
        for (const auto* key : {"kernel_glob", "library"}) if (rule.contains(key)) {
            const auto text = rule.at(key).get<std::string>();
            if (text.empty() || text.size() > 4096) throw ArgumentError("Invalid capture pattern length");
            for (unsigned char c : text) if (c < 32 || c >= 127) throw ArgumentError("Capture pattern must be ASCII");
        }
    }
    return value;
}
inline std::pair<std::string, int> resolve_policy(const nlohmann::json& policy,
                                                 const std::string& kernel, const std::string& library) {
    int index = 0;
    for (const auto& rule : policy.at("rules")) {
        if ((!rule.contains("library") || rule.at("library") == library) &&
            kernel_glob(rule.at("kernel_glob"), kernel)) return {rule.at("sampling"), index};
        ++index;
    }
    return {policy.at("default"), -1};
}
inline std::uint64_t sampled_width(std::uint64_t width, unsigned mode) {
    if (!width || mode > 2 || width > UINT32_MAX) throw ArgumentError("Invalid sampling geometry/mode");
    return mode == 0 ? width : mode == 1 ? (width + 31) / 32 : 1;
}
inline std::uint64_t sampled_slot(std::uint64_t physical, std::uint64_t width, unsigned mode) {
    const auto slots = sampled_width(width, mode);
    const auto block = physical / width, local = physical % width;
    if ((mode == 1 && local % 32) || (mode == 2 && local)) throw ArgumentError("Unsampled physical thread");
    return block * slots + (mode == 1 ? local / 32 : mode == 2 ? 0 : local);
}
} // namespace neutrino::sys
