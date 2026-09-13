from gui_agent.locator.coordinate_mapper import CoordinateMapper
from gui_agent.perception.screen_capture import ScreenCapture

region = (500, 300, 800, 600)

with ScreenCapture() as capture:
    frame = capture.capture(region)

print("Frame shape:", frame.image.shape)
print("Image size:", frame.image_size)
print("Desktop region:", frame.region)

