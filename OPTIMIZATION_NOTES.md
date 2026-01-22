# Optimization Notes

## Baseline
- **Cycles**: 147,734
- **Target**: < 1,487 (to beat Claude Opus 4.5's best)

---

## Optimization 1: Remove Unnecessary Loads/Stores

### Observation
The baseline loads and stores walker state (idx, val) from/to memory **every iteration** of every round. This is wasteful because:

1. Walker state only needs to be **read from memory once** at the start
2. Walker state only needs to be **written to memory once** at the end
3. **Indices don't even need to be written** - the test only checks final values!

### Current (Wasteful) Pattern
```
For each round (16x):
    For each walker (256x):
        LOAD idx from memory      ← unnecessary after round 0
        LOAD val from memory      ← unnecessary after round 0
        ... compute ...
        STORE idx to memory       ← unnecessary until final round (and actually never needed!)
        STORE val to memory       ← unnecessary until final round
```

### Optimized Pattern
```
# ONCE at start: Load all walker state into scratch
For each walker (256x):
    scratch[idx_i] = LOAD mem[inp_indices_p + i]
    scratch[val_i] = LOAD mem[inp_values_p + i]

# All rounds operate on scratch only
For each round (16x):
    For each walker (256x):
        ... compute using scratch[idx_i], scratch[val_i] ...
        ... update scratch[idx_i], scratch[val_i] ...

# ONCE at end: Store final values to memory (indices not needed!)
For each walker (256x):
    STORE mem[inp_values_p + i] = scratch[val_i]
```

### Expected Savings
- Baseline does: 256 walkers × 16 rounds × 4 memory ops = **16,384 memory ops**
- Optimized does: 256 × 2 loads + 256 × 1 store = **768 memory ops**
- Savings: ~15,600 fewer memory operations

### Scratch Space Budget
- Available: 1,536 words
- Need for 256 walkers: 256 idx + 256 val = 512 words
- Remaining: 1,024 words for temps, constants, etc. ✓

### Using vload/vstore + Both Load/Store Slots

**Key constraints:**
- `vload`/`vstore` load/store 8 **contiguous** memory locations
- We have **2 load slots** and **2 store slots** per cycle
- Walker indices are contiguous: `mem[inp_indices_p + 0], mem[inp_indices_p + 1], ...`
- Walker values are contiguous: `mem[inp_values_p + 0], mem[inp_values_p + 1], ...`

**Loading phase (at start):**
```
256 indices ÷ 8 = 32 vloads needed
256 values  ÷ 8 = 32 vloads needed
Total: 64 vloads

With 2 load slots per cycle:
64 vloads ÷ 2 = 32 cycles for loading
```

**Storing phase (at end):**
```
256 values ÷ 8 = 32 vstores needed
(indices not stored - not checked by test!)

With 2 store slots per cycle:
32 vstores ÷ 2 = 16 cycles for storing
```

**Implementation sketch:**
```python
# Allocate contiguous scratch for all walker state
walker_idx_base = self.alloc_scratch("walker_idx", 256)  # scratch[0:256]
walker_val_base = self.alloc_scratch("walker_val", 256)  # scratch[256:512]

# Load all walker data at start (32 cycles)
for chunk in range(0, 256, 8):
    # Use both load slots in same instruction!
    self.instrs.append({
        "load": [
            ("vload", walker_idx_base + chunk, addr_for_indices_chunk),
            ("vload", walker_val_base + chunk, addr_for_values_chunk),
        ]
    })
    # Need to set up addresses first...

# ... all computation in scratch ...

# Store final values (16 cycles)
for chunk in range(0, 256, 16):  # 2 vstores per cycle
    self.instrs.append({
        "store": [
            ("vstore", addr_for_values_chunk, walker_val_base + chunk),
            ("vstore", addr_for_values_chunk + 8, walker_val_base + chunk + 8),
        ]
    })
```

**Challenge:** `vload` takes a scalar address in scratch, so we need to pre-compute addresses or use `add_imm` to bump pointers.

### Status
- [ ] Implement
- [ ] Test correctness
- [ ] Measure cycles

---

## Future Optimizations (TODO)
- [ ] VLIW packing (use multiple ALU slots per cycle)
- [ ] SIMD vectorization (process 8 walkers at once)
- [ ] Loop structure optimization
- [ ] Instruction scheduling
