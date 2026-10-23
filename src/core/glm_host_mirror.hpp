// src/core/glm_host_mirror.hpp - the routed experts' permanent host mirror (STRATA_GLM_HOST_MIRROR=1, MAYA_GLM_HOST_MIRROR
// too; off by default): its sizing, its refusals and its RAM-slot decisions as plain functions, which
// tests/glm_host_mirror_test.cpp checks on the CPU alone.  No CUDA or HIP headers.
//
// The exclusive tiers (the default) keep an expert in VRAM or in the pinned RAM tier: a promotion frees its RAM copy
// at the next boundary, and an eviction from VRAM copies the blob back (D2H) because no other copy is in memory.  The
// mirror keeps every active expert's blob in ONE pinned RAM slot for the model's lifetime - the warm-up fills it, VRAM
// is seeded from it, nothing frees, reuses or overwrites it - so an eviction only retires the VRAM copy.  The device's
// rule is unchanged: a VRAM copy wins, and only an expert with no VRAM copy can go to the CPU lane.
//
// With the mirror off every decision below is the one the exclusive tiers always made (the tests compare each against
// the frozen source's own expression).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace strata::core::glmfast {

// the RAM tier's slot states by value (Glm5Model::FastState's kRFree .. kRHelper; glm_fast_state.hpp asserts they agree)
namespace ram_st {
enum : char { kFree = 0, kHold = 1, kRelease = 2, kDemote = 3, kNew = 4, kLoad = 5, kPin = 6, kHelper = 7 };
}

// STRATA_GLM_HOST_MIRROR's (and STRATA_GLM_HOST_MIRROR_VERIFY's) value: on for a non-zero number; unset, empty or 0 off
inline bool host_mirror_on(const char* v) { return v != nullptr && std::atoi(v) != 0; }

// ---- capacity.  Each RAM class (one blob stride) holds every expert of every MoE layer of that stride - nothing more:
//      with the mirror no path needs a free RAM slot (a disk miss cannot happen, an eviction or a prompt's park keeps
//      the copy that is there, the lookahead has nothing to read).  layer_class: each layer's class, -1 not a MoE layer
//      on this part.
inline std::vector<int64_t> host_mirror_slots(const std::vector<int>& layer_class, int n_classes, int n_expert) {
    std::vector<int64_t> n((size_t) std::max(0, n_classes), 0);
    for (int c : layer_class)
        if (c >= 0 && c < n_classes) n[(size_t) c] += std::max(0, n_expert);
    return n;
}
inline uint64_t host_mirror_bytes(const std::vector<int64_t>& slots, const std::vector<size_t>& stride) {
    uint64_t b = 0;
    for (size_t c = 0; c < slots.size() && c < stride.size(); ++c) b += (uint64_t) std::max<int64_t>(0, slots[c]) * stride[c];
    return b;
}
// whether `need` bytes can be pinned and still leave `headroom` for the OS and the server and `pin_room` pinnable above
// the tier (the driver's own buffers) out of `avail`, the free RAM now (MemAvailable and reclaimable ARC; <= 0
// unknown: the pinning itself decides)
inline bool host_mirror_fits(uint64_t need, int64_t avail, int64_t headroom, int64_t pin_room) {
    if (avail <= 0) return true;
    return (int64_t) need + std::max<int64_t>(0, headroom) + std::max<int64_t>(0, pin_room) <= avail;
}
// ... and a fixed budget (STRATA_GLM_RAM_GB, or the engine's own; < 0: none) never below the mirror
inline bool host_mirror_budget_ok(uint64_t need, int64_t budget) { return budget < 0 || (uint64_t) budget >= need; }
// the start's refusal: per class what it needs and what it got, and why - all or nothing, never a smaller mirror
inline std::string host_mirror_shortfall(const std::vector<int64_t>& need, const std::vector<int64_t>& got,
                                         const std::vector<size_t>& stride, const std::string& why) {
    std::string s = "glm fast: STRATA_GLM_HOST_MIRROR=1 needs every active expert in pinned RAM (" + why + "):";
    for (size_t c = 0; c < need.size(); ++c) {
        char b[160];
        std::snprintf(b, sizeof b, "%s class %zu (%.2f MB a slot): %lld slots needed, %lld obtained", c ? ";" : "", c,
                      c < stride.size() ? (double) stride[c] / 1e6 : 0.0, (long long) need[c],
                      (long long) (c < got.size() ? got[c] : 0));
        s += b;
    }
    s += "; " + std::to_string(host_mirror_bytes(need, stride)) +
         " bytes in all. There is no partial mirror: free RAM (MAYA_GLM_RAM_HEADROOM_GB, other programs), lower "
         "--max-context, or leave MAYA_GLM_HOST_MIRROR off";
    return s;
}

// ---- the settings the mirror refuses: those that discard parked experts, fill the tiers lazily, leave experts out or
//      bias the routes by residency (approximations whose result would follow the changed residency), and the
//      ownership exchanges it has not been checked with.  Supported: one GPU, one sequence, ordinary decode, the prompt
//      path's lending and return.
struct HostMirrorSettings {
    bool lend_drop = false;        // STRATA_GLM_LEND_DROP (any value: the engine tests its presence)
    bool warm_off = false;         // STRATA_GLM_WARM=0 (lazy fill)
    double miss_skip_w = 0.0;      // STRATA_GLM_MISS_SKIP_W
    int disk_skip_from = 8;        // STRATA_GLM_DISK_SKIP_FROM
    double disk_skip_w = 0.0;      // STRATA_GLM_DISK_SKIP_W
    int trunk_keep = 8;            // STRATA_GLM_TRUNK_KEEP
    bool trunk_keep_map = false;   // STRATA_GLM_TRUNK_KEEP_MAP leaves out an expert on some layer
    double route_affinity = 0.0;   // STRATA_GLM_ROUTE_AFFINITY
    int parts = 1;                 // GPUs the layers are split across
    bool helper = false;           // STRATA_GLM_EXPERT_HELPER
    bool unified = false;          // an integrated GPU's unified memory
    bool ram_resident = false;     // STRATA_GLM_RAM_RESIDENT (the other tier policy)
    bool drafting = false;         // a DFlash drafter or the NextN block on this part (speculative windows)
    bool grouped = false;          // STRATA_GLM_GROUP_EXPERTS with batched rows
    int seqs = 1;                  // sequences (STRATA_GLM_SEQS)
};
// the environment's part, parsed as the engine parses each setting
inline void host_mirror_read_env(HostMirrorSettings& s) {
    s.lend_drop = std::getenv("STRATA_GLM_LEND_DROP") != nullptr;
    if (const char* v = std::getenv("STRATA_GLM_WARM")) s.warm_off = std::atoi(v) == 0;
    if (const char* v = std::getenv("STRATA_GLM_MISS_SKIP_W")) s.miss_skip_w = std::max(0.0, std::min(1.0, std::atof(v)));
    if (const char* v = std::getenv("STRATA_GLM_DISK_SKIP_FROM")) s.disk_skip_from = std::max(0, std::min(8, std::atoi(v)));
    if (const char* v = std::getenv("STRATA_GLM_DISK_SKIP_W")) s.disk_skip_w = std::max(0.0, std::atof(v));
    if (const char* v = std::getenv("STRATA_GLM_TRUNK_KEEP")) s.trunk_keep = std::max(1, std::min(8, std::atoi(v)));
    if (const char* v = std::getenv("STRATA_GLM_TRUNK_KEEP_MAP"))
        for (const char* p = v; *p != '\0'; ++p) s.trunk_keep_map |= *p >= '1' && *p <= '7';
    if (const char* v = std::getenv("STRATA_GLM_ROUTE_AFFINITY")) s.route_affinity = std::max(0.0, std::atof(v));
    if (const char* v = std::getenv("STRATA_GLM_EXPERT_HELPER")) s.helper = v[0] != '\0';
    if (const char* v = std::getenv("STRATA_GLM_RAM_RESIDENT")) s.ram_resident = std::atoi(v) != 0;
}
// "" when the mirror runs with these settings, else every conflict ("; "-separated)
inline std::string host_mirror_conflicts(const HostMirrorSettings& s) {
    std::string out;
    const auto add = [&](const std::string& w) { out += (out.empty() ? "" : "; ") + w; };
    if (s.lend_drop) add("STRATA_GLM_LEND_DROP (drops lent experts; unset it)");
    if (s.warm_off) add("STRATA_GLM_WARM=0 (the mirror is filled eagerly at the start)");
    if (s.miss_skip_w > 0.0) add("STRATA_GLM_MISS_SKIP_W (leaves out experts not in VRAM)");
    if (s.disk_skip_from < 8) add("STRATA_GLM_DISK_SKIP_FROM (leaves out disk-only experts)");
    if (s.disk_skip_w > 0.0) add("STRATA_GLM_DISK_SKIP_W (leaves out disk-only experts)");
    if (s.trunk_keep < 8) add("STRATA_GLM_TRUNK_KEEP (leaves out experts not in VRAM)");
    if (s.trunk_keep_map) add("STRATA_GLM_TRUNK_KEEP_MAP (leaves out experts not in VRAM)");
    if (s.route_affinity > 0.0) add("STRATA_GLM_ROUTE_AFFINITY (biases routes towards VRAM residents)");
    if (s.parts > 1) add(std::to_string(s.parts) + " GPUs (one GPU only)");
    if (s.helper) add("STRATA_GLM_EXPERT_HELPER (a helper GPU's ownership exchange)");
    if (s.unified) add("unified memory (an integrated GPU)");
    if (s.ram_resident) add("STRATA_GLM_RAM_RESIDENT (the exclusive RAM-resident policy)");
    if (s.drafting) add("a draft model or NextN drafting (STRATA_GLM_NO_MTP=1, no DFlash)");
    if (s.grouped) add("STRATA_GLM_GROUP_EXPERTS (batched rows)");
    if (s.seqs > 1) add(std::to_string(s.seqs) + " sequences (one sequence only)");
    return out;
}

// ---- the RAM slots' decisions (mirror false: the exclusive tiers' own)
// a demand or predicted promotion: its held RAM copy is released at the next boundary
inline bool ram_release_on_promotion(bool mirror, char st) { return !mirror && st == ram_st::kHold; }
// the RAM tier's eviction may take this slot (a spare refill's demotion target, the free-slot reserve, a lending park)
inline bool ram_victim_eligible(bool mirror, char st) { return !mirror && st == ram_st::kHold; }
// the boundary's clean-up of held copies (duplicates freed, lost ones adopted; STRATA_GLM_TIER_GC) looks at this slot
inline bool ram_gc_applies(bool mirror, bool tier_gc, char st) { return !mirror && tier_gc && st == ram_st::kHold; }
// the free RAM slots each class keeps for the next token's disk reads (STRATA_GLM_RAM_FREE)
inline int ram_keep_free(bool mirror, bool ram_resident, const char* ram_free, bool ahead_read) {
    if (mirror) return 0;
    return ram_resident ? 0 : ram_free ? std::max(1, std::atoi(ram_free)) : ahead_read ? 12 : 4;
}
// a background promotion landed: its source slot's state after (the exclusive tiers free the pinned source, the
// mirror makes it its readable permanent copy again) ...
inline char bg_up_source_after(bool mirror, char st) {
    return st != ram_st::kPin ? st : mirror ? (char) ram_st::kHold : (char) ram_st::kFree;
}
// ... and whether its key's RAM entry (ram_of, the device's RAM table) is cleared
inline bool bg_up_clears_ram_entry(bool mirror) { return !mirror; }
// an expert's permanent host copy: its slot holds this key and is readable - held, or the source of a background
// promotion in flight (kPin: read, never written).  Not a demotion still landing, an AHEAD read or a released slot.
inline bool host_copy_valid(int key, int rs, int slot_key, char st) {
    return key >= 0 && rs >= 0 && slot_key == key && (st == ram_st::kHold || st == ram_st::kPin);
}
// an eviction from VRAM (a spare refill, a background swap): the exclusive tiers copy the blob to RAM (or drop it);
// the mirror only retires the VRAM copy - and a missing host copy is a broken invariant, never a copy elsewhere
enum class Evict { kCopy, kRetire, kBroken };
inline Evict evict_mode(bool mirror, bool host_valid) {
    return !mirror ? Evict::kCopy : host_valid ? Evict::kRetire : Evict::kBroken;
}
// a prompt's lending moves a resident out of VRAM: the exclusive tiers park it in RAM by a D2H (or drop it with
// STRATA_GLM_LEND_DROP, or without a RAM class); the mirror keeps the copy it has - decided BEFORE the discard choice
enum class Park { kCopy, kDrop, kKeep };
inline Park park_mode(bool mirror, bool lend_drop, bool has_class) {
    if (mirror) return Park::kKeep;
    return lend_drop || !has_class ? Park::kDrop : Park::kCopy;
}
// the free RAM slots the warm-up leaves per class (the exclusive tiers 4 for disk reads; RAM-resident and the mirror none)
inline int warm_ram_reserve(bool mirror, bool ram_resident) { return mirror || ram_resident ? 0 : 4; }
// the LOOKAHEAD's reads into free RAM slots (STRATA_GLM_AHEAD_READ): off with the mirror, where no key lacks a copy
inline bool ahead_read_on(bool mirror, bool requested) { return requested && !mirror; }

}  // namespace strata::core::glmfast
