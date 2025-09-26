from expected_signaling_probability import expected_signaling_probability, Direction


def main():
    N = 1000
    d_As_and_Bs = [i for i in range(2, 7)]

    for d in d_As_and_Bs:
        esp, _ = expected_signaling_probability(N, d, d, Direction.A_TO_B)
        print(f"d={d}, <S>={esp}")


if __name__ == "__main__":
    main()
