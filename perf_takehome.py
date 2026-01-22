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

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ) -> None:
        """
        Optimized kernel that loads walker state into scratch once,
        operates on scratch throughout, and stores results once at end.
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

        # Allocate scratch arrays for all walker state (256 each)
        walker_idx = self.alloc_scratch("walker_idx", batch_size)  # scratch[walker_idx:walker_idx+256]
        walker_val = self.alloc_scratch("walker_val", batch_size)  # scratch[walker_val:walker_val+256]
        
        # Temp addresses for vload/vstore (need scalar addresses in scratch)
        addr_idx = self.alloc_scratch("addr_idx")  # address pointer for indices
        addr_val = self.alloc_scratch("addr_val")  # address pointer for values
        
        # ============ LOAD PHASE: Load all walker state into scratch ============
        # Initialize address pointers to start of idx and val arrays in memory
        self.add("alu", ("+", addr_idx, self.scratch["inp_indices_p"], zero_const))
        self.add("alu", ("+", addr_val, self.scratch["inp_values_p"], zero_const))
        
        # Load 256 indices and 256 values using vload (8 at a time)
        # Use both load slots per cycle: load indices and values in parallel
        for chunk in range(0, batch_size, VLEN):
            # vload 8 indices and 8 values in the same cycle (2 load slots)
            self.instrs.append({
                "load": [
                    ("vload", walker_idx + chunk, addr_idx),
                    ("vload", walker_val + chunk, addr_val),
                ]
            })
            # Increment both address pointers by 8
            self.instrs.append({
                "alu": [
                    ("+", addr_idx, addr_idx, eight_const),
                    ("+", addr_val, addr_val, eight_const),
                ]
            })

        # Pause for debug sync
        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting main loop"))

        # ============ COMPUTE PHASE: Work entirely in scratch ============
        # Process walkers in batches of VLEN (8) using SIMD
        
        # Vector temp registers for SIMD operations
        tmp1_vec = self.alloc_scratch("tmp1_vec", VLEN)  # 8-element vector temp
        tmp2_vec = self.alloc_scratch("tmp2_vec", VLEN)  # 8-element vector temp for hash
        tmp3_vec = self.alloc_scratch("tmp3_vec", VLEN)  # 8-element vector for select result
        
        # Broadcast constants to vectors (need vector versions for vALU)
        zero_vec = self.alloc_scratch("zero_vec", VLEN)
        one_vec = self.alloc_scratch("one_vec", VLEN)
        two_vec = self.alloc_scratch("two_vec", VLEN)
        n_nodes_vec = self.alloc_scratch("n_nodes_vec", VLEN)
        
        # Broadcast scalar constants to vectors
        self.instrs.append({"valu": [("vbroadcast", zero_vec, zero_const)]})
        self.instrs.append({"valu": [("vbroadcast", one_vec, one_const)]})
        self.instrs.append({"valu": [("vbroadcast", two_vec, two_const)]})
        self.instrs.append({"valu": [("vbroadcast", n_nodes_vec, self.scratch["n_nodes"])]})
        
        # Pre-broadcast all hash constants to vectors
        hash_const_vecs: dict[int, int] = {}
        for (op1, val1, op2, op3, val3) in HASH_STAGES:
            for const_val in [val1, val3]:
                if const_val not in hash_const_vecs:
                    # Allocate vector and broadcast constant
                    const_scalar = self.scratch_const(const_val)
                    const_vec = self.alloc_scratch(f"hash_const_{const_val}", VLEN)
                    self.instrs.append({"valu": [("vbroadcast", const_vec, const_scalar)]})
                    hash_const_vecs[const_val] = const_vec
        
        # Scratch space for 8 tree addresses and 8 node values (for gathering)
        tree_addrs = self.alloc_scratch("tree_addrs", VLEN)  # 8 addresses
        node_vals = self.alloc_scratch("node_vals", VLEN)    # 8 gathered node values

        for round in range(rounds):
            # Process 256 walkers in batches of 8
            for batch in range(batch_size // VLEN):  # 32 batches
                batch_offset = batch * VLEN
                idx_base = walker_idx + batch_offset  # Base of 8 contiguous indices
                val_base = walker_val + batch_offset  # Base of 8 contiguous values
                
                # === GATHER PHASE: Load 8 scattered tree node values ===
                # Compute 8 tree addresses in ONE cycle (pack all 8 ALU ops)
                # tree_addr[j] = forest_values_p + idx[j]
                self.instrs.append({
                    "alu": [
                        ("+", tree_addrs + j, self.scratch["forest_values_p"], idx_base + j)
                        for j in range(VLEN)
                    ]
                })
                
                # Load 8 tree node values using both load slots (4 cycles)
                for j in range(0, VLEN, 2):
                    self.instrs.append({
                        "load": [
                            ("load", node_vals + j, tree_addrs + j),
                            ("load", node_vals + j + 1, tree_addrs + j + 1),
                        ]
                    })
                
                # === PRE-HASH SIMD: XOR values with node values ===
                # val_vec ^= node_vals (8 XORs in one vALU slot)
                self.instrs.append({
                    "valu": [("^", val_base, val_base, node_vals)]
                })
                
                # === HASH PHASE: Now fully SIMD! ===
                hash_instrs = self.build_hash_simd(val_base, tmp1_vec, tmp2_vec, hash_const_vecs)
                self.instrs.extend(hash_instrs)
                
                # === POST-HASH SIMD: Compute next indices ===
                # tmp1_vec = val_vec % 2
                self.instrs.append({"valu": [("%", tmp1_vec, val_base, two_vec)]})
                
                # tmp1_vec = (tmp1_vec == 0)
                self.instrs.append({"valu": [("==", tmp1_vec, tmp1_vec, zero_vec)]})
                
                # tmp3_vec = tmp1_vec ? 1 : 2 (vselect)
                self.instrs.append({"flow": [("vselect", tmp3_vec, tmp1_vec, one_vec, two_vec)]})
                
                # idx_vec = idx_vec * 2
                self.instrs.append({"valu": [("*", idx_base, idx_base, two_vec)]})
                
                # idx_vec = idx_vec + tmp3_vec
                self.instrs.append({"valu": [("+", idx_base, idx_base, tmp3_vec)]})
                
                # tmp1_vec = idx_vec < n_nodes_vec
                self.instrs.append({"valu": [("<", tmp1_vec, idx_base, n_nodes_vec)]})
                
                # idx_vec = tmp1_vec ? idx_vec : 0 (vselect)
                self.instrs.append({"flow": [("vselect", idx_base, tmp1_vec, idx_base, zero_vec)]})

        # ============ STORE PHASE: Write final values to memory ============
        # Only need to store values (indices not checked by test)
        # Reset address pointer
        self.add("alu", ("+", addr_val, self.scratch["inp_values_p"], zero_const))
        
        for chunk in range(0, batch_size, VLEN):
            # vstore 8 values per cycle (using 1 store slot)
            self.instrs.append({
                "store": [
                    ("vstore", addr_val, walker_val + chunk),
                ]
            })
            # Increment address pointer by 8
            self.add("alu", ("+", addr_val, addr_val, eight_const))

        # Required to match with the yield in reference_kernel2
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
