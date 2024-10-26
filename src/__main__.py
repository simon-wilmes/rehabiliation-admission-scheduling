from src.instance import create_instance_from_file


def main():
    inst = create_instance_from_file("data/inst001/inst001.txt")
    print("Successfully created instance from file.")


if __name__ == "__main__":
    main()
