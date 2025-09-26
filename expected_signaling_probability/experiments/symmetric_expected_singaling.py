from expected_signaling_probability.utils.math import expected_signaling_probability
from expected_signaling_probability.utils.math import Direction


def main():
    N = 100
    d_As_and_Bs = [i for i in range(2, 6)]

    for d in d_As_and_Bs:
        esp, _ = expected_signaling_probability(N, d, d, Direction.A_TO_B)
        print(f"d={d}, <S>={esp}")


if __name__ == "__main__":
    main()
