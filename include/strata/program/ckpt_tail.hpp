// include/strata/program/ckpt_tail.hpp - MAYA_GLM_CKPT_TAIL: which periodic checkpoints a prompt read in pieces keeps.
//
// A read cut at the periodic checkpoints (MAYA_GLM_CKPT_EVERY) ends with a piece that may be very short.  Every call
// streams the layers' RAM-tier experts to the GPU again (27-29 GB a call on PCIe 3.0 from 1,440 tokens to 16,384), so
// a 229-token tail piece costs nearly as much wall time as a 16,384-token one (6.2 s against 20 s on an RTX 3090,
// EXL3-sc_2.60bpw, 64K).  While the tail piece and the piece before it fit one prompt chunk together, the periodic
// checkpoint between them buys nothing and the two are read in one call.
//
// The turns' own checkpoints (the end of the system prompt and the start of the last turn) are never dropped: a later
// prompt that leaves the conversation inside the merged piece restores the checkpoint before it and reads more, but
// those two stay.
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace strata::program {

/// Drop the trailing periodic marks while the piece each one starts and the piece before it fit `chunk` tokens
/// together.  `kept` is in ascending order; `turns` holds the positions that must stay; `from` is where the read
/// starts, used as the lower edge when one mark is all that is left.  `chunk == 0` (the setting off, or the engine
/// not up) leaves `kept` alone.
inline void ckpt_tail_merge(std::vector<std::size_t>& kept, std::size_t chunk,
                            const std::vector<std::size_t>& turns, std::size_t from, std::size_t end) {
    if (chunk == 0) return;
    while (!kept.empty()) {
        const std::size_t last = kept.back();
        if (std::find(turns.begin(), turns.end(), last) != turns.end()) break;   // a turn's own checkpoint
        const std::size_t prev = kept.size() >= 2 ? kept[kept.size() - 2] : from;
        if (end - prev > chunk) break;                                          // the pair does not fit one chunk
        kept.pop_back();
    }
}

}  // namespace strata::program
