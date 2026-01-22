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
        body = []  # array of slots

        # Temp scratch registers for computation
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        for round in range(rounds):
            for i in range(batch_size):
                # Read idx and val from SCRATCH (not memory)
                scratch_idx_addr = walker_idx + i  # this is just unrolling, it's not changing a scratch value
                scratch_val_addr = walker_val + i
                
                # Copy walker's idx from storage to working register: tmp_idx = scratch[walker_idx + i]
                # ALU with +0 copies: scratch[tmp_idx] = scratch[scratch_idx_addr] + scratch[zero_const]
                body.append(("alu", ("+", tmp_idx, scratch_idx_addr, zero_const)))
                
                # Copy walker's val from storage to working register: tmp_val = scratch[walker_val + i]
                body.append(("alu", ("+", tmp_val, scratch_val_addr, zero_const)))
                
                # node_val = mem[forest_values_p + idx] - MUST hit memory
                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                
                # val = myhash(val ^ node_val)
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("alu", ("%", tmp1, tmp_val, two_const)))
                body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                
                # idx = 0 if idx >= n_nodes else idx
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                
                # Store idx and val back to SCRATCH (not memory)
                # Copy tmp_idx to scratch[walker_idx + i]
                # We need: scratch[scratch_idx_addr] = tmp_idx
                # ALU writes to dest, so: scratch[scratch_idx_addr] = tmp_idx + 0
                body.append(("alu", ("+", scratch_idx_addr, tmp_idx, zero_const)))
                body.append(("alu", ("+", scratch_val_addr, tmp_val, zero_const)))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)

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
