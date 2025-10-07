from enum import Enum


class Direction(Enum):
    A_TO_B = "A to B"
    B_TO_A = "B to A"

    def to_ptrace_index(self) -> int:
        return 1 if self == Direction.A_TO_B else 0

    def to_str(self) -> str:
        return "AtoB" if self == Direction.A_TO_B else "BtoA"
