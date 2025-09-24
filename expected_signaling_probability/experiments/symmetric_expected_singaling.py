from expected_signaling_probability.utils.math import expected_signaling_probability


def main():
    N = 10
    d_As_and_Bs = [i for i in range(2, 8)]

    for d in d_As_and_Bs:
        esp, _ = expected_signaling_probability(N, d, d, "A to B")
        print(f"d={d}, <S>={esp}")


if __name__ == "__main__":
    main()
