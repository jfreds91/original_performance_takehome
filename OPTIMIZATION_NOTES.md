# Optimization Notes

## Baseline
- **Cycles**: 147,734
- **Target**: < 1,487 (to beat Claude Opus 4.5's best)

---

## Optimization 1: Remove Unnecessary Loads/Stores

We can load the idx array (worker position) and val array (worker value) into scratch using vloads, and only write them back to memory once at the end. 1.12x speedup

## Optimization 2: Gather node values from memory and store them in contiguous scratch

This incurs a slight performance hit (down to 1.08) but prepares us for the next step

## Optimization 3: SIMD

SIMD each ALU operation leading up to hash computation: 1.6x. SIMD entire hash algo: 7.5x

## Optimization 4: pack ALU addr computations

gets us to 9.2x

## optim 5: Pack SIMD in hash

A few can be parallelized to use multiple slots