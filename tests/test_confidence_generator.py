from wild_visual_navigation.utils import ConfidenceGenerator

def test_confidence_generator():
    # Design a long 1D signal

    # 
    cg = ConfidenceGenerator()
    for i in range(100000):
        inp = (
            torch.rand(
                10,
            )
            * 10
        )
        res = cg.update(inp)
        print("inp ", inp, " res ", res, "std", cg.std)


if __name__ == "__main__":
    test_confidence_generator()
