// CPU-only checks of MAYA_GLM_CKPT_TAIL's decision (include/strata/program/ckpt_tail.hpp): which periodic marks a
// prompt read in pieces keeps, so a short tail piece is read with the piece before it instead of paying a whole call's
// expert streaming for a few tokens.  No CUDA, no engine, no model: fixtures only.
#include "../../include/strata/program/ckpt_tail.hpp"

#include <cstdio>
#include <vector>

namespace {

int g_fail = 0, g_pass = 0;
#define CHECK(expr)                                                              \
    do {                                                                         \
        if (expr) {                                                              \
            ++g_pass;                                                            \
        } else {                                                                 \
            ++g_fail;                                                            \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        }                                                                        \
    } while (0)

using strata::program::ckpt_tail_merge;
using marks = std::vector<std::size_t>;

// a 40,000-token read cut every 16,384: marks at 16384 and 32768, the tail piece 7,232 tokens
void short_tail_is_merged() {
    marks kept{16384, 32768};
    ckpt_tail_merge(kept, 32768, {}, 0, 40000);
    CHECK(kept == marks({16384}));            // the 7,232-token tail reads with the piece before it
    CHECK(40000 - 16384 <= 32768);            // (and that pair still fits one chunk)
}

void tail_that_does_not_fit_is_kept() {
    marks kept{16384, 32768};
    ckpt_tail_merge(kept, 32768, {}, 0, 40000);   // tail 7,232 fits
    CHECK(kept.size() == 1);
    marks big{16384, 32768};
    ckpt_tail_merge(big, 8192, {}, 0, 40000);     // the tail alone is bigger than the chunk
    CHECK(big.size() == 2);
}

void turns_are_never_dropped() {
    // a periodic mark at 36,000 after the last turn's mark (32,768): 36,000 - 32,768 fits one chunk, so 36,000 goes
    // and the loop then stops at the turn's mark
    marks kept{16384, 32768, 36000};
    ckpt_tail_merge(kept, 32768, marks({32768}), 0, 40000);
    CHECK(kept == marks({16384, 32768}));
    // when the last mark IS a turn's, nothing is dropped: merging would mean dropping it
    marks k2{16384, 39000};
    ckpt_tail_merge(k2, 32768, marks({39000}), 0, 40000);
    CHECK(k2 == marks({16384, 39000}));
}

void one_mark_measures_from_the_read_start() {
    // a single periodic mark, the read starting at 1,000: the piece is 1,000..40000, far more than a chunk
    marks kept{16384};
    ckpt_tail_merge(kept, 32768, {}, 1000, 40000);
    CHECK(kept.size() == 1);
    // the same mark when the whole read fits one chunk: the mark buys nothing
    marks k2{16384};
    ckpt_tail_merge(k2, 32768, {}, 1000, 20000);
    CHECK(k2.empty());
}

void off_means_untouched() {
    marks kept{16384, 32768};
    ckpt_tail_merge(kept, 0, {}, 0, 40000);   // setting off, or the engine not up
    CHECK(kept == marks({16384, 32768}));
}

void repeated_marks_merge_while_each_pair_fits() {
    // 90,000 tokens, marks every 16,384: 16384 32768 49152 65536 81920, tail 8,080
    marks kept{16384, 32768, 49152, 65536, 81920};
    ckpt_tail_merge(kept, 32768, {}, 0, 90000);
    // 65,536..90,000 is 24,464 and fits, so 81,920 goes; then 49,152..90,000 is 40,848 and does not, so 65,536 stays
    CHECK(kept == marks({16384, 32768, 49152, 65536}));
}

void already_short_read_is_untouched() {
    marks kept;                                   // a prompt that fits one piece has no marks at all
    ckpt_tail_merge(kept, 32768, {}, 0, 20000);
    CHECK(kept.empty());
}

}  // namespace

int main() {
    short_tail_is_merged();
    tail_that_does_not_fit_is_kept();
    turns_are_never_dropped();
    one_mark_measures_from_the_read_start();
    off_means_untouched();
    repeated_marks_merge_while_each_pair_fits();
    already_short_read_is_untouched();
    std::printf("ckpt_tail_test: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
