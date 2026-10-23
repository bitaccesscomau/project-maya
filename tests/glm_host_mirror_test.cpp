// CPU-only checks of the routed experts' permanent host mirror (STRATA_GLM_HOST_MIRROR, src/core/glm_host_mirror.hpp):
// its sizing per RAM class, its refusals of conflicting settings, and which RAM slots a promotion, the boundary's
// clean-up, the RAM tier's victims, the free-slot reserve and a landed background promotion may release.  With the
// mirror off every decision must be the exclusive tiers' own (the legacy expressions are written out below).
// No CUDA or HIP headers, device queries or allocations: fixtures only, never model or service evidence.
#include "../src/core/glm_host_mirror.hpp"
#include "../include/strata/platform/env_alias.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
int g_fail = 0, g_pass = 0;
#define CHECK(expr)                                                                \
    do {                                                                           \
        if (expr) {                                                                \
            ++g_pass;                                                              \
        } else {                                                                   \
            ++g_fail;                                                              \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr);   \
        }                                                                          \
    } while (0)

using namespace strata::core::glmfast;
namespace rs = strata::core::glmfast::ram_st;

constexpr size_t kLarge = 11046912;   // 3.5-bit layer 3 (EXL3-sc_2.60bpw manifest), already 4 KiB aligned
constexpr size_t kSmall = 7901184;    // 2.5-bit layers 4..44
constexpr size_t kNextN = 12619776;   // the inactive NextN layer 45, were it loaded

// the active one-GPU trunk of this pack: layers 0..2 dense, 3 large, 4..44 small, slot 45 (NextN) not loaded
std::vector<int> trunk_classes(bool with_nextn) {
    std::vector<int> c(46, -1);
    c[3] = 0;
    for (int il = 4; il <= 44; ++il) c[(size_t) il] = 1;
    if (with_nextn) c[45] = 2;
    return c;
}

void flag() {
    CHECK(!host_mirror_on(nullptr));
    CHECK(!host_mirror_on(""));
    CHECK(!host_mirror_on("0"));
    CHECK(!host_mirror_on("00"));
    CHECK(!host_mirror_on("no"));   // (std::atoi: 0, as every STRATA_GLM_* switch reads)
    CHECK(host_mirror_on("1"));
    CHECK(host_mirror_on("2"));
    // STRATA_GLM_EVICT_STATS reads as the other new switches: unset, empty or 0 off (=0 must not enable it)
    CHECK(!evict_stats_on(nullptr));
    CHECK(!evict_stats_on(""));
    CHECK(!evict_stats_on("0"));
    CHECK(evict_stats_on("1"));
}

// the MAYA_ spellings: main() calls env_aliases() first, which copies every MAYA_X to STRATA_X, so the new switches
// work under either name and =0 stays off
void aliases() {
    const char* names[] = {"STRATA_GLM_HOST_MIRROR", "STRATA_GLM_EVICT_STATS", "MAYA_GLM_HOST_MIRROR",
                           "MAYA_GLM_EVICT_STATS"};
    for (const char* n : names) unsetenv(n);
    setenv("MAYA_GLM_HOST_MIRROR", "1", 1);
    setenv("MAYA_GLM_EVICT_STATS", "0", 1);
    strata::platform::env_aliases();
    CHECK(host_mirror_on(std::getenv("STRATA_GLM_HOST_MIRROR")));
    CHECK(std::getenv("STRATA_GLM_EVICT_STATS") != nullptr && !evict_stats_on(std::getenv("STRATA_GLM_EVICT_STATS")));
    for (const char* n : names) unsetenv(n);
}

void capacity() {
    const auto slots = host_mirror_slots(trunk_classes(false), 2, 288);
    CHECK(slots.size() == 2);
    CHECK(slots.size() == 2 && slots[0] == 288);
    CHECK(slots.size() == 2 && slots[1] == 11808);
    CHECK(host_mirror_bytes(slots, {kLarge, kSmall}) == 96478691328ull);
    int64_t total = 0;
    for (int64_t n : slots) total += n;
    CHECK(total == 12096);
    // a NextN layer loaded on this part is its own class, covered too (never silently left out)
    const auto with_mtp = host_mirror_slots(trunk_classes(true), 3, 288);
    CHECK(with_mtp.size() == 3 && with_mtp[2] == 288);
    CHECK(host_mirror_bytes(with_mtp, {kLarge, kSmall, kNextN}) == 96478691328ull + 3634495488ull);
    // nothing for an empty part; a class index past n_classes is ignored, not written out of bounds
    CHECK(host_mirror_slots(std::vector<int>(46, -1), 2, 288) == std::vector<int64_t>({0, 0}));
    CHECK(host_mirror_slots({0, 5, 1}, 2, 288) == std::vector<int64_t>({288, 288}));
    // the mismatched-length guard: a stride missing for a class counts nothing rather than reading past the end
    CHECK(host_mirror_bytes({288, 11808}, {kLarge}) == 288ull * kLarge);
}

void fit() {
    constexpr int64_t G = int64_t{1} << 30;
    const uint64_t need = 96478691328ull;
    CHECK(host_mirror_fits(need, 140 * G, 6 * G, 512 << 20));
    CHECK(!host_mirror_fits(need, 95 * G, 6 * G, 512 << 20));
    // exactly at the edge: need + headroom + pin room == available fits; one byte less does not
    CHECK(host_mirror_fits(need, (int64_t) need + 6 * G + (512 << 20), 6 * G, 512 << 20));
    CHECK(!host_mirror_fits(need, (int64_t) need + 6 * G + (512 << 20) - 1, 6 * G, 512 << 20));
    CHECK(host_mirror_fits(need, 0, 6 * G, 512 << 20));   // MemAvailable unknown: the pinning itself decides
    // a fixed budget (STRATA_GLM_RAM_GB) below the mirror refuses
    CHECK(host_mirror_budget_ok(need, -1));
    CHECK(host_mirror_budget_ok(need, (int64_t) need));
    CHECK(!host_mirror_budget_ok(need, (int64_t) need - 1));
}

void shortfall() {
    const std::string m = host_mirror_shortfall({288, 11808}, {288, 0}, {kLarge, kSmall}, "pinning refused");
    CHECK(m.find("STRATA_GLM_HOST_MIRROR") != std::string::npos);
    CHECK(m.find("pinning refused") != std::string::npos);
    CHECK(m.find("class 0") != std::string::npos && m.find("class 1") != std::string::npos);
    CHECK(m.find("288 slots needed, 288 obtained") != std::string::npos);
    CHECK(m.find("11808 slots needed, 0 obtained") != std::string::npos);
    CHECK(m.find("96478691328") != std::string::npos);
    CHECK(m.find("no partial mirror") != std::string::npos);
    // the RAM headroom is the OS's and the server's, never expert capacity: the refusal must not suggest lowering it
    CHECK(m.find("HEADROOM") == std::string::npos);
    CHECK(m.find("free RAM") != std::string::npos && m.find("--max-context") != std::string::npos);
}

void conflicts() {
    HostMirrorSettings s;
    CHECK(host_mirror_conflicts(s).empty());   // the supported configuration: one GPU, one sequence, warm, exact
    const auto has = [](const HostMirrorSettings& x, const char* what) {
        return host_mirror_conflicts(x).find(what) != std::string::npos;
    };
    { auto x = s; x.lend_drop = true;      CHECK(has(x, "STRATA_GLM_LEND_DROP")); }
    { auto x = s; x.warm_off = true;       CHECK(has(x, "STRATA_GLM_WARM=0")); }
    { auto x = s; x.miss_skip_w = 0.1;     CHECK(has(x, "STRATA_GLM_MISS_SKIP_W")); }
    { auto x = s; x.disk_skip_from = 6;    CHECK(has(x, "STRATA_GLM_DISK_SKIP_FROM")); }
    { auto x = s; x.disk_skip_w = 0.05;    CHECK(has(x, "STRATA_GLM_DISK_SKIP_W")); }
    { auto x = s; x.trunk_keep = 6;        CHECK(has(x, "STRATA_GLM_TRUNK_KEEP")); }
    { auto x = s; x.trunk_keep_map = true; CHECK(has(x, "STRATA_GLM_TRUNK_KEEP_MAP")); }
    { auto x = s; x.route_affinity = 0.5;  CHECK(has(x, "STRATA_GLM_ROUTE_AFFINITY")); }
    { auto x = s; x.parts = 2;             CHECK(has(x, "GPUs")); }
    { auto x = s; x.helper = true;         CHECK(has(x, "STRATA_GLM_EXPERT_HELPER")); }
    { auto x = s; x.unified = true;        CHECK(has(x, "unified memory")); }
    { auto x = s; x.ram_resident = true;   CHECK(has(x, "STRATA_GLM_RAM_RESIDENT")); }
    { auto x = s; x.drafting = true;       CHECK(has(x, "draft")); }
    { auto x = s; x.grouped = true;        CHECK(has(x, "STRATA_GLM_GROUP_EXPERTS")); }
    { auto x = s; x.seqs = 2;              CHECK(has(x, "sequence")); }
    {   // every conflict is named, not only the first
        auto x = s;
        x.lend_drop = x.warm_off = true;
        x.parts = 3;
        CHECK(has(x, "STRATA_GLM_LEND_DROP") && has(x, "STRATA_GLM_WARM=0") && has(x, "GPUs"));
    }
    // the environment as the engine reads it (presence for LEND_DROP; the parsed values elsewhere)
    const char* names[] = {"STRATA_GLM_LEND_DROP", "STRATA_GLM_WARM", "STRATA_GLM_MISS_SKIP_W", "STRATA_GLM_DISK_SKIP_FROM",
                           "STRATA_GLM_DISK_SKIP_W", "STRATA_GLM_TRUNK_KEEP", "STRATA_GLM_TRUNK_KEEP_MAP",
                           "STRATA_GLM_ROUTE_AFFINITY", "STRATA_GLM_EXPERT_HELPER", "STRATA_GLM_RAM_RESIDENT"};
    for (const char* n : names) unsetenv(n);
    const auto env_conflicts = [] {
        HostMirrorSettings e;
        host_mirror_read_env(e);
        return host_mirror_conflicts(e);
    };
    CHECK(env_conflicts().empty());
    setenv("STRATA_GLM_LEND_DROP", "0", 1);   // (the engine tests presence: "0" still drops)
    CHECK(env_conflicts().find("STRATA_GLM_LEND_DROP") != std::string::npos);
    unsetenv("STRATA_GLM_LEND_DROP");
    setenv("STRATA_GLM_WARM", "1", 1);
    CHECK(env_conflicts().empty());
    setenv("STRATA_GLM_WARM", "0", 1);
    CHECK(env_conflicts().find("STRATA_GLM_WARM=0") != std::string::npos);
    unsetenv("STRATA_GLM_WARM");
    setenv("STRATA_GLM_DISK_SKIP_FROM", "8", 1);
    CHECK(env_conflicts().empty());
    setenv("STRATA_GLM_DISK_SKIP_FROM", "6", 1);
    CHECK(env_conflicts().find("STRATA_GLM_DISK_SKIP_FROM") != std::string::npos);
    unsetenv("STRATA_GLM_DISK_SKIP_FROM");
    setenv("STRATA_GLM_TRUNK_KEEP_MAP", "88888888", 1);
    CHECK(env_conflicts().empty());
    setenv("STRATA_GLM_TRUNK_KEEP_MAP", "88848888", 1);
    CHECK(env_conflicts().find("STRATA_GLM_TRUNK_KEEP_MAP") != std::string::npos);
    unsetenv("STRATA_GLM_TRUNK_KEEP_MAP");
    setenv("STRATA_GLM_TRUNK_KEEP", "8", 1);
    CHECK(env_conflicts().empty());
    setenv("STRATA_GLM_TRUNK_KEEP", "5", 1);
    CHECK(env_conflicts().find("STRATA_GLM_TRUNK_KEEP") != std::string::npos);
    unsetenv("STRATA_GLM_TRUNK_KEEP");
    setenv("STRATA_GLM_MISS_SKIP_W", "0", 1);
    CHECK(env_conflicts().empty());
    setenv("STRATA_GLM_MISS_SKIP_W", "0.12", 1);
    CHECK(env_conflicts().find("STRATA_GLM_MISS_SKIP_W") != std::string::npos);
    unsetenv("STRATA_GLM_MISS_SKIP_W");
    setenv("STRATA_GLM_ROUTE_AFFINITY", "0", 1);
    CHECK(env_conflicts().empty());
    setenv("STRATA_GLM_ROUTE_AFFINITY", "0.5", 1);
    CHECK(env_conflicts().find("STRATA_GLM_ROUTE_AFFINITY") != std::string::npos);
    unsetenv("STRATA_GLM_ROUTE_AFFINITY");
    setenv("STRATA_GLM_EXPERT_HELPER", "", 1);   // (empty: no helper, as the engine reads it)
    CHECK(env_conflicts().empty());
    setenv("STRATA_GLM_EXPERT_HELPER", "1", 1);
    CHECK(env_conflicts().find("STRATA_GLM_EXPERT_HELPER") != std::string::npos);
    unsetenv("STRATA_GLM_EXPERT_HELPER");
    setenv("STRATA_GLM_RAM_RESIDENT", "0", 1);
    CHECK(env_conflicts().empty());
    setenv("STRATA_GLM_RAM_RESIDENT", "1", 1);
    CHECK(env_conflicts().find("STRATA_GLM_RAM_RESIDENT") != std::string::npos);
    unsetenv("STRATA_GLM_RAM_RESIDENT");
}

// Off: the exclusive tiers' decisions, exactly as the frozen source wrote them inline
void legacy_decisions() {
    for (int st = 0; st <= 7; ++st) {
        const char c = (char) st;
        CHECK(ram_release_on_promotion(false, c) == (c == rs::kHold));
        CHECK(ram_victim_eligible(false, c) == (c == rs::kHold));
        CHECK(ram_gc_applies(false, true, c) == (c == rs::kHold));
        CHECK(!ram_gc_applies(false, false, c));
        CHECK(bg_up_source_after(false, c) == (c == rs::kPin ? (char) rs::kFree : c));
    }
    CHECK(bg_up_clears_ram_entry(false));
    const char* kfs[] = {nullptr, "1", "0", "7", "-3"};
    for (const char* kf : kfs)
        for (int rr = 0; rr < 2; ++rr)
            for (int ah = 0; ah < 2; ++ah) {
                const int legacy = rr ? 0 : kf ? std::max(1, std::atoi(kf)) : ah ? 12 : 4;
                CHECK(ram_keep_free(false, rr != 0, kf, ah != 0) == legacy);
            }
    CHECK(warm_ram_reserve(false, false) == 4);
    CHECK(warm_ram_reserve(false, true) == 0);
    CHECK(ahead_read_on(false, true));
    CHECK(!ahead_read_on(false, false));
    CHECK(evict_mode(false, true) == Evict::kCopy);
    CHECK(evict_mode(false, false) == Evict::kCopy);
    CHECK(park_mode(false, false, true) == Park::kCopy);
    CHECK(park_mode(false, true, true) == Park::kDrop);    // STRATA_GLM_LEND_DROP
    CHECK(park_mode(false, false, false) == Park::kDrop);  // (a layer without a RAM class)
}

// On: a mirrored slot is never released, reused or overwritten
void mirror_decisions() {
    for (int st = 0; st <= 7; ++st) {
        const char c = (char) st;
        CHECK(!ram_release_on_promotion(true, c));
        CHECK(!ram_victim_eligible(true, c));
        CHECK(!ram_gc_applies(true, true, c));
        CHECK(bg_up_source_after(true, c) == (c == rs::kPin ? (char) rs::kHold : c));
    }
    CHECK(!bg_up_clears_ram_entry(true));
    CHECK(ram_keep_free(true, false, nullptr, true) == 0);
    CHECK(ram_keep_free(true, false, "9", false) == 0);
    CHECK(warm_ram_reserve(true, false) == 0);
    CHECK(!ahead_read_on(true, true));
    CHECK(evict_mode(true, true) == Evict::kRetire);
    CHECK(evict_mode(true, false) == Evict::kBroken);
    // the mirror's park is checked BEFORE the discard choice (and the start refuses LEND_DROP with it)
    CHECK(park_mode(true, false, true) == Park::kKeep);
    CHECK(park_mode(true, true, true) == Park::kKeep);
    // a valid host copy: this key's slot, readable (held, or a background promotion's pinned source)
    CHECK(host_copy_valid(7, 3, 7, rs::kHold));
    CHECK(host_copy_valid(7, 3, 7, rs::kPin));
    CHECK(!host_copy_valid(7, -1, 7, rs::kHold));    // no slot
    CHECK(!host_copy_valid(7, 3, 8, rs::kHold));     // another key's slot (stale ram_of)
    CHECK(!host_copy_valid(7, 3, 7, rs::kFree));
    CHECK(!host_copy_valid(7, 3, 7, rs::kDemote));   // a landing D2H: not a complete copy
    CHECK(!host_copy_valid(7, 3, 7, rs::kLoad));     // an AHEAD read maybe still in flight
    CHECK(!host_copy_valid(7, 3, 7, rs::kRelease));
    CHECK(!host_copy_valid(-1, 3, -1, rs::kHold));
    // the verification's orphan count: a resident VRAM slot whose key's booked slot (slot_of) is another slot or none
    // (a duplicate the tables cannot reach; its bytes may still be right, so (a) and (b) alone miss it)
    CHECK(!vram_orphan(true, 5, 5));     // the key's own booked slot
    CHECK(vram_orphan(true, 7, 5));      // the key is booked at slot 7: slot 5 is an orphan duplicate
    CHECK(vram_orphan(true, -1, 5));     // the key is booked nowhere
    CHECK(!vram_orphan(false, 7, 5));    // not resident (free, landing, draining, lent): not counted
}

// A small tier model driven by the decisions in the order the engine applies them: the warm-up, a demand promotion,
// the boundary's release and clean-up, the free-slot reserve, a background promotion landing and spare-refill
// evictions.  Under the mirror every key keeps exactly one valid host slot throughout; off, the exclusive tiers drop
// the promoted expert's RAM copy at the boundary, as before.
struct Model {
    bool mirror;
    int ne;                       // experts (one layer)
    std::vector<int> key;         // RAM slot -> key, -1
    std::vector<char> st;         // RAM slot state
    std::vector<int> ram_of;      // key -> RAM slot, -1
    std::vector<int> vram;        // VRAM residents (keys)
    int d2h = 0, retired = 0, broken = 0;
};

Model warm(bool mirror) {
    Model m{mirror, 8, {}, {}, std::vector<int>(8, -1), {0, 1}};
    // the exclusive tiers: RAM holds the 6 experts VRAM does not, plus free slots; the mirror: all 8, nothing free
    const int slots = mirror ? (int) host_mirror_slots({0}, 1, 8)[0] : 6 + warm_ram_reserve(false, false);
    // (fewer slots than experts to place: a broken sizing, which the caller's checks report rather than write past)
    m.key.assign((size_t) std::max(0, slots), -1);
    m.st.assign((size_t) std::max(0, slots), rs::kFree);
    if (slots < (mirror ? 8 : 6)) return m;
    int s = 0;
    for (int k = 2; k < 8; ++k, ++s) m.key[(size_t) s] = k, m.st[(size_t) s] = rs::kHold, m.ram_of[(size_t) k] = s;
    if (mirror)
        for (int k : m.vram) m.key[(size_t) s] = k, m.st[(size_t) s] = rs::kHold, m.ram_of[(size_t) k] = s, ++s;
    return m;
}
void promote(Model& m, int k) {   // the service: a demand promotion of RAM expert k into a spare
    m.vram.push_back(k);
    const int r = m.ram_of[(size_t) k];
    if (r >= 0 && ram_release_on_promotion(m.mirror, m.st[(size_t) r])) m.st[(size_t) r] = rs::kRelease;
}
bool in_vram(const Model& m, int k) {
    for (int v : m.vram)
        if (v == k) return true;
    return false;
}
void boundary(Model& m, bool tier_gc) {   // step (1): releases, then the clean-up of held copies
    for (size_t s = 0; s < m.st.size(); ++s) {
        if (m.st[s] == rs::kRelease) {
            if (m.key[s] >= 0 && m.ram_of[(size_t) m.key[s]] == (int) s) m.ram_of[(size_t) m.key[s]] = -1;
            m.key[s] = -1;
            m.st[s] = rs::kFree;
        } else if (ram_gc_applies(m.mirror, tier_gc, m.st[s])) {
            const int k = m.key[s];
            if (k >= 0 && m.ram_of[(size_t) k] == (int) s && !in_vram(m, k)) continue;   // the normal case
            if (k >= 0 && m.ram_of[(size_t) k] == (int) s) m.ram_of[(size_t) k] = -1;
            m.key[s] = -1;
            m.st[s] = rs::kFree;
        }
    }
    // step (4): the free-slot reserve evicts RAM victims until enough slots are free
    int nfree = 0;
    for (char c : m.st) nfree += c == rs::kFree;
    const int keep = ram_keep_free(m.mirror, false, nullptr, false);
    for (size_t s = 0; s < m.st.size() && nfree < keep; ++s)
        if (ram_victim_eligible(m.mirror, m.st[s])) {
            m.ram_of[(size_t) m.key[s]] = -1;
            m.key[s] = -1;
            m.st[s] = rs::kFree;
            ++nfree;
        }
}
void evict(Model& m, int k) {   // step (3): a spare refill evicts VRAM resident k
    const int r = m.ram_of[(size_t) k];
    const bool valid = host_copy_valid(k, r, r >= 0 ? m.key[(size_t) r] : -1, r >= 0 ? m.st[(size_t) r] : (char) rs::kFree);
    switch (evict_mode(m.mirror, valid)) {
        case Evict::kCopy: ++m.d2h; break;   // (a D2H into a free or victim slot: not modelled further)
        case Evict::kRetire: ++m.retired; break;
        case Evict::kBroken: ++m.broken; break;
    }
    for (size_t i = 0; i < m.vram.size(); ++i)
        if (m.vram[i] == k) m.vram.erase(m.vram.begin() + (long) i), i = m.vram.size();
}
bool covered(const Model& m) {   // every key exactly one valid host slot
    std::vector<int> n((size_t) m.ne, 0);
    for (size_t s = 0; s < m.key.size(); ++s)
        if (m.key[s] >= 0) ++n[(size_t) m.key[s]];
    for (int k = 0; k < m.ne; ++k) {
        const int r = m.ram_of[(size_t) k];
        if (n[(size_t) k] != 1 || !host_copy_valid(k, r, r >= 0 ? m.key[(size_t) r] : -1, r >= 0 ? m.st[(size_t) r] : (char) 0))
            return false;
    }
    return true;
}

// A conversation set aside and restored: the engine writes the RAM tier's slot states by value and slot_load puts
// them back as the live state.  Closed PR #15 (STRATA_GLM_RAM_SHADOW) did exactly this and crashed with an illegal
// memory access in the next request's prefill, because its shadows were reclaimable duplicates and the clean-up
// could free a slot another table still pointed at.  Under the mirror there are no duplicates to free and no slot
// is ever reused, so a restored state is still one valid host slot per key.
void set_aside() {
    Model m = warm(true);
    CHECK(covered(m));
    promote(m, 2);
    boundary(m, true);
    CHECK(covered(m));
    // set aside: a copy of the live slot state is what the engine writes out
    const std::vector<int> key0 = m.key;
    const std::vector<char> st0 = m.st;
    const std::vector<int> ram0 = m.ram_of;
    // the next request runs and moves residency around
    evict(m, 2);
    promote(m, 7);
    boundary(m, true);
    CHECK(covered(m));
    // ... then the saved conversation comes back, with its residents
    m.key = key0;
    m.st = st0;
    m.ram_of = ram0;
    m.vram = {0, 1, 2};
    CHECK(covered(m));                                    // one valid host slot per expert, nothing dangling
    for (char c : m.st) CHECK(c == rs::kHold);            // every restored slot is the permanent copy
    m.d2h = m.retired = m.broken = 0;                     // counters from the restore onward
    evict(m, 2);                                          // and an eviction after the restore copies nothing back
    evict(m, 0);
    CHECK(m.d2h == 0 && m.broken == 0 && m.retired == 2);
    CHECK(covered(m));                                    // retiring VRAM copies leaves the host table intact
    boundary(m, true);
    CHECK(covered(m));
}

void tier_model() {
    {
        Model m = warm(true);
        CHECK(m.key.size() == 8);   // exactly every expert, no landing slot
        CHECK(covered(m));
        if (m.key.size() == 8 && covered(m)) {
            promote(m, 5);
            CHECK(covered(m));
            boundary(m, true);
            CHECK(covered(m));   // the promoted expert keeps its host copy; the clean-up frees no duplicate
            // a background promotion of expert 6 lands: its pinned source stays the permanent copy
            m.st[(size_t) m.ram_of[6]] = rs::kPin;
            CHECK(covered(m));
            m.st[(size_t) m.ram_of[6]] = bg_up_source_after(true, m.st[(size_t) m.ram_of[6]]);
            m.vram.push_back(6);
            CHECK(covered(m) && m.st[(size_t) m.ram_of[6]] == rs::kHold);
            evict(m, 5);
            evict(m, 0);
            CHECK(m.d2h == 0 && m.retired == 2 && m.broken == 0);
            boundary(m, true);
            CHECK(covered(m));
            // a key whose host copy was lost is a broken invariant, never a copy into another slot
            m.st[(size_t) m.ram_of[6]] = rs::kFree;
            evict(m, 6);
            CHECK(m.broken == 1 && m.d2h == 0);
        }
    }
    {
        Model m = warm(false);
        CHECK(m.key.size() == 10);   // 6 RAM experts + the 4 free slots the warm-up leaves
        CHECK(!covered(m));          // exclusive: VRAM residents 0 and 1 have no host copy
        promote(m, 5);
        boundary(m, true);
        CHECK(m.ram_of[5] < 0);      // legacy: the promoted expert's RAM copy is gone after the boundary
        evict(m, 5);
        CHECK(m.d2h == 1 && m.retired == 0);   // ... so its eviction copies it back
    }
}
}  // namespace

int main() {
    flag();
    aliases();
    capacity();
    fit();
    shortfall();
    conflicts();
    legacy_decisions();
    mirror_decisions();
    tier_model();
    set_aside();
    std::printf("glm_host_mirror_test: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
