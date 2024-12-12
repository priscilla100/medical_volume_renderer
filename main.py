from app import App


def main() -> None:
    try:
        app = App()
        app.run()
    except Exception as e:
        raise e


if __name__ == '__main__':
    main()

