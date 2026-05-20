from TorchQML.synthesis import Architecture, smoke_unitary_distillation


def main():
    result = smoke_unitary_distillation(
        teacher="qft",
        n=2,
        arch=Architecture(depth=1, local="ry_rz", entangler="none"),
        epochs=1,
        seed=0,
    )
    print(result)


if __name__ == "__main__":
    main()
