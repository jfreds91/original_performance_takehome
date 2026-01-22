"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    Instruction,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)

# Type alias for an operation assigned to an engine: (engine_name, operation_tuple)
AssignedOp = tuple[Engine, tuple]


class KernelBuilder:
    instrs: list[Instruction]
    scratch: dict[str, int]
    scratch_debug: dict[int, tuple[str, int]]
    scratch_ptr: int
    const_map: dict[int, int]

    def __init__(self) -> None:
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self) -> DebugInfo:
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[AssignedOp], vliw: bool = False) -> list[Instruction]:
        # Simple slot packing that just uses one slot per instruction bundle
        instrs: list[Instruction] = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine: Engine, slot: tuple) -> None:
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name: str | None = None, length: int = 1) -> int:
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val: int, name: str | None = None) -> int:
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(
        self, val_hash_addr: int, tmp1: int, tmp2: int, round: int, i: int
    ) -> list[AssignedOp]:
        slots: list[AssignedOp] = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_hash_simd(
        self, val_vec: int, tmp1_vec: int, tmp2_vec: int, hash_const_vecs: dict[int, int]
    ) -> list[Instruction]:
        """
        SIMD version of hash - operates on 8 values at once.
        val_vec: base address of 8 contiguous values to hash in place
        tmp1_vec, tmp2_vec: base addresses of 8-element temp vectors
        hash_const_vecs: maps scalar constant value -> base address of broadcasted vector
        """
        instrs: list[Instruction] = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            # Pack ops 1 and 2 (both read val_vec, independent)
            # tmp1_vec = val_vec OP1 const1_vec
            # tmp2_vec = val_vec OP3 const3_vec
            instrs.append({
                "valu": [
                    (op1, tmp1_vec, val_vec, hash_const_vecs[val1]),
                    (op3, tmp2_vec, val_vec, hash_const_vecs[val3]),
                ]
            })
            # val_vec = tmp1_vec OP2 tmp2_vec (depends on above)
            instrs.append({"valu": [(op2, val_vec, tmp1_vec, tmp2_vec)]})

        return instrs

    def build_hash_simd_2batch(
        self,
        val_a: int, val_b: int,
        tmp1_a: int, tmp1_b: int,
        tmp2_a: int, tmp2_b: int,
        hash_const_vecs: dict[int, int]
    ) -> list[Instruction]:
        """
        SIMD hash for 2 batches (16 values) in parallel.
        Uses 4 vALU slots per instruction (2 per batch).
        """
        instrs: list[Instruction] = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            # Pack all 4 ops: 2 for batch A, 2 for batch B
            instrs.append({
                "valu": [
                    (op1, tmp1_a, val_a, hash_const_vecs[val1]),
                    (op3, tmp2_a, val_a, hash_const_vecs[val3]),
                    (op1, tmp1_b, val_b, hash_const_vecs[val1]),
                    (op3, tmp2_b, val_b, hash_const_vecs[val3]),
                ]
            })
            # Final combine: 2 ops for A and B
            instrs.append({
                "valu": [
                    (op2, val_a, tmp1_a, tmp2_a),
                    (op2, val_b, tmp1_b, tmp2_b),
                ]
            })

        return instrs

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ) -> None:
        """
        Optimized kernel with software pipelining and double buffering.
        Processes 2 batches (16 walkers) in parallel, overlapping loads
        of batch N+1 with compute of batch N for ~2x speedup.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        
        # Scratch space addresses for memory layout info
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        eight_const = self.scratch_const(8)
        sixteen_const = self.scratch_const(16)

        # Allocate scratch arrays for all walker state (256 each)
        walker_idx = self.alloc_scratch("walker_idx", batch_size)
        walker_val = self.alloc_scratch("walker_val", batch_size)
        
        # Temp addresses for vload/vstore
        addr_idx = self.alloc_scratch("addr_idx")
        addr_val = self.alloc_scratch("addr_val")
        
        # ============ LOAD PHASE: Load all walker state into scratch ============
        self.add("alu", ("+", addr_idx, self.scratch["inp_indices_p"], zero_const))
        self.add("alu", ("+", addr_val, self.scratch["inp_values_p"], zero_const))
        
        for chunk in range(0, batch_size, VLEN):
            self.instrs.append({
                "load": [
                    ("vload", walker_idx + chunk, addr_idx),
                    ("vload", walker_val + chunk, addr_val),
                ]
            })
            self.instrs.append({
                "alu": [
                    ("+", addr_idx, addr_idx, eight_const),
                    ("+", addr_val, addr_val, eight_const),
                ]
            })

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting main loop with software pipelining"))

        # ============ DOUBLE BUFFER ALLOCATION ============
        # Buffer A: for even pipeline stages
        tree_addrs_A = self.alloc_scratch("tree_addrs_A", VLEN * 2)  # 16 addresses
        node_vals_A = self.alloc_scratch("node_vals_A", VLEN * 2)    # 16 values
        
        # Buffer B: for odd pipeline stages  
        tree_addrs_B = self.alloc_scratch("tree_addrs_B", VLEN * 2)
        node_vals_B = self.alloc_scratch("node_vals_B", VLEN * 2)
        
        # Per-batch temp vectors for parallel processing
        # Batch A temps (for first 8 walkers in double-batch)
        tmp1_vec_a = self.alloc_scratch("tmp1_vec_a", VLEN)
        tmp2_vec_a = self.alloc_scratch("tmp2_vec_a", VLEN)
        tmp3_vec_a = self.alloc_scratch("tmp3_vec_a", VLEN)
        
        # Batch B temps (for second 8 walkers in double-batch)
        tmp1_vec_b = self.alloc_scratch("tmp1_vec_b", VLEN)
        tmp2_vec_b = self.alloc_scratch("tmp2_vec_b", VLEN)
        tmp3_vec_b = self.alloc_scratch("tmp3_vec_b", VLEN)
        
        # Broadcast constants to vectors
        zero_vec = self.alloc_scratch("zero_vec", VLEN)
        one_vec = self.alloc_scratch("one_vec", VLEN)
        two_vec = self.alloc_scratch("two_vec", VLEN)
        n_nodes_vec = self.alloc_scratch("n_nodes_vec", VLEN)
        
        self.instrs.append({"valu": [("vbroadcast", zero_vec, zero_const)]})
        self.instrs.append({"valu": [("vbroadcast", one_vec, one_const)]})
        self.instrs.append({"valu": [("vbroadcast", two_vec, two_const)]})
        self.instrs.append({"valu": [("vbroadcast", n_nodes_vec, self.scratch["n_nodes"])]})
        
        # Pre-broadcast all hash constants to vectors
        hash_const_vecs: dict[int, int] = {}
        for (op1, val1, op2, op3, val3) in HASH_STAGES:
            for const_val in [val1, val3]:
                if const_val not in hash_const_vecs:
                    const_scalar = self.scratch_const(const_val)
                    const_vec = self.alloc_scratch(f"hash_const_{const_val}", VLEN)
                    self.instrs.append({"valu": [("vbroadcast", const_vec, const_scalar)]})
                    hash_const_vecs[const_val] = const_vec

        # ============ HELPER FUNCTIONS FOR PIPELINE ============
        def emit_addr_calc(double_batch: int, tree_addrs: int) -> None:
            """Compute 16 tree addresses for a double-batch (uses ALU, 2 cycles)."""
            batch_offset = double_batch * VLEN * 2
            idx_base_a = walker_idx + batch_offset
            idx_base_b = walker_idx + batch_offset + VLEN
            
            # First 12 addresses in one cycle
            self.instrs.append({
                "alu": [
                    ("+", tree_addrs + j, self.scratch["forest_values_p"], idx_base_a + j)
                    for j in range(VLEN)
                ] + [
                    ("+", tree_addrs + VLEN + j, self.scratch["forest_values_p"], idx_base_b + j)
                    for j in range(4)  # Only first 4 of batch B (12 total)
                ]
            })
            # Remaining 4 addresses
            self.instrs.append({
                "alu": [
                    ("+", tree_addrs + VLEN + 4 + j, self.scratch["forest_values_p"], idx_base_b + 4 + j)
                    for j in range(4)
                ]
            })
        
        def emit_loads(tree_addrs: int, node_vals: int) -> list[Instruction]:
            """Load 16 scattered tree values (uses LOAD, 8 cycles). Returns list of instructions."""
            instrs = []
            for j in range(0, VLEN * 2, 2):
                instrs.append({
                    "load": [
                        ("load", node_vals + j, tree_addrs + j),
                        ("load", node_vals + j + 1, tree_addrs + j + 1),
                    ]
                })
            return instrs
        
        def emit_compute(double_batch: int, node_vals: int) -> list[Instruction]:
            """Compute phase for a double-batch (XOR, hash, post-hash). Returns list of instructions."""
            instrs = []
            batch_offset = double_batch * VLEN * 2
            idx_a = walker_idx + batch_offset
            idx_b = walker_idx + batch_offset + VLEN
            val_a = walker_val + batch_offset
            val_b = walker_val + batch_offset + VLEN
            node_a = node_vals
            node_b = node_vals + VLEN
            
            # XOR: val ^= node_val (pack both batches)
            instrs.append({
                "valu": [
                    ("^", val_a, val_a, node_a),
                    ("^", val_b, val_b, node_b),
                ]
            })
            
            # Hash phase: 2 batches in parallel
            hash_instrs = self.build_hash_simd_2batch(
                val_a, val_b, tmp1_vec_a, tmp1_vec_b, tmp2_vec_a, tmp2_vec_b, hash_const_vecs
            )
            instrs.extend(hash_instrs)
            
            # Post-hash phase (optimized ordering)
            # Pack independent ops: mod and mul are independent
            instrs.append({
                "valu": [
                    ("%", tmp1_vec_a, val_a, two_vec),
                    ("%", tmp1_vec_b, val_b, two_vec),
                    ("*", idx_a, idx_a, two_vec),
                    ("*", idx_b, idx_b, two_vec),
                ]
            })
            
            # eq operations
            instrs.append({
                "valu": [
                    ("==", tmp1_vec_a, tmp1_vec_a, zero_vec),
                    ("==", tmp1_vec_b, tmp1_vec_b, zero_vec),
                ]
            })
            
            # vselect 1 for batch A (flow bottleneck - only 1 slot)
            instrs.append({"flow": [("vselect", tmp3_vec_a, tmp1_vec_a, one_vec, two_vec)]})
            
            # vselect 1 for batch B + add for batch A (overlap flow with vALU!)
            instrs.append({
                "flow": [("vselect", tmp3_vec_b, tmp1_vec_b, one_vec, two_vec)],
                "valu": [("+", idx_a, idx_a, tmp3_vec_a)],
            })
            
            # add for batch B
            instrs.append({"valu": [("+", idx_b, idx_b, tmp3_vec_b)]})
            
            # lt operations (pack both)
            instrs.append({
                "valu": [
                    ("<", tmp1_vec_a, idx_a, n_nodes_vec),
                    ("<", tmp1_vec_b, idx_b, n_nodes_vec),
                ]
            })
            
            # vselect 2 for batch A
            instrs.append({"flow": [("vselect", idx_a, tmp1_vec_a, idx_a, zero_vec)]})
            
            # vselect 2 for batch B
            instrs.append({"flow": [("vselect", idx_b, tmp1_vec_b, idx_b, zero_vec)]})
            
            return instrs

        # ============ MAIN LOOP WITH SOFTWARE PIPELINING ============
        n_double_batches = batch_size // (VLEN * 2)  # 16 double-batches of 16 walkers each
        
        for round_idx in range(rounds):
            # Use alternating buffers
            buffers = [
                (tree_addrs_A, node_vals_A),
                (tree_addrs_B, node_vals_B),
            ]
            
            # BOOTSTRAP: Load first double-batch (no compute to overlap with)
            emit_addr_calc(0, buffers[0][0])
            load_instrs = emit_loads(buffers[0][0], buffers[0][1])
            self.instrs.extend(load_instrs)
            
            # STEADY STATE: Interleave load(N+1) with compute(N)
            for db in range(n_double_batches - 1):
                curr_buf = buffers[db % 2]
                next_buf = buffers[(db + 1) % 2]
                
                # Get compute instructions for current double-batch
                compute_instrs = emit_compute(db, curr_buf[1])
                
                # Get load instructions for next double-batch
                # First emit addr calc (uses ALU, can overlap with first compute instrs)
                next_batch_offset = (db + 1) * VLEN * 2
                idx_base_a = walker_idx + next_batch_offset
                idx_base_b = walker_idx + next_batch_offset + VLEN
                
                # Interleave: addr_calc + first compute instruction
                # Cycle 1: first 12 addr calcs + XOR
                first_compute = compute_instrs[0]  # XOR instruction
                self.instrs.append({
                    "alu": [
                        ("+", next_buf[0] + j, self.scratch["forest_values_p"], idx_base_a + j)
                        for j in range(VLEN)
                    ] + [
                        ("+", next_buf[0] + VLEN + j, self.scratch["forest_values_p"], idx_base_b + j)
                        for j in range(4)
                    ],
                    "valu": first_compute.get("valu", []),
                })
                
                # Cycle 2: remaining 4 addr calcs + hash stage 1 (ops 1&2)
                second_compute = compute_instrs[1]  # First hash instruction
                self.instrs.append({
                    "alu": [
                        ("+", next_buf[0] + VLEN + 4 + j, self.scratch["forest_values_p"], idx_base_b + 4 + j)
                        for j in range(4)
                    ],
                    "valu": second_compute.get("valu", []),
                })
                
                # Cycles 3-10: loads + remaining hash instructions
                load_instrs = emit_loads(next_buf[0], next_buf[1])  # 8 load instructions
                compute_idx = 2  # Start from 3rd compute instruction
                
                for load_instr in load_instrs:
                    if compute_idx < len(compute_instrs):
                        # Merge load with compute
                        merged = dict(load_instr)
                        comp = compute_instrs[compute_idx]
                        for engine, slots in comp.items():
                            if engine in merged:
                                merged[engine].extend(slots)
                            else:
                                merged[engine] = list(slots)
                        self.instrs.append(merged)
                        compute_idx += 1
                    else:
                        # Just load, no more compute to overlap
                        self.instrs.append(load_instr)
                
                # Remaining compute instructions (after loads are done)
                while compute_idx < len(compute_instrs):
                    self.instrs.append(compute_instrs[compute_idx])
                    compute_idx += 1
            
            # DRAIN: Compute final double-batch (no next batch to load)
            last_buf = buffers[(n_double_batches - 1) % 2]
            compute_instrs = emit_compute(n_double_batches - 1, last_buf[1])
            self.instrs.extend(compute_instrs)

        # ============ STORE PHASE: Write final values to memory ============
        self.add("alu", ("+", addr_val, self.scratch["inp_values_p"], zero_const))
        
        for chunk in range(0, batch_size, VLEN):
            self.instrs.append({
                "store": [("vstore", addr_val, walker_val + chunk)]
            })
            self.add("alu", ("+", addr_val, addr_val, eight_const))

        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734


def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
) -> int:
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
