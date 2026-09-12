from gui_agent.perception.screen_capture import ScreenCapture


def main() -> None:
    with ScreenCapture() as capture:
        image = capture.capture()

        print(f"Image shape: {image.shape}")
        print(f"Image dtype: {image.dtype}")


if __name__ == "__main__":
    main()