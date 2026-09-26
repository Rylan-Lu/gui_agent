from gui_agent.agent.observation import Observation


def test_create_observation():
    obs = Observation(
        screenshot="screen.png",
        screen_width=2560,
        screen_height=1440,
    )

    assert obs.screenshot == "screen.png"
    assert obs.screen_width == 2560
    assert obs.screen_height == 1440
    assert obs.ocr_result is None
    assert obs.metadata == {}
    assert obs.timestamp > 0


def test_observation_with_ocr():
    ocr_result = [
        {
            "text": "Firefox",
            "confidence": 0.98,
        }
    ]

    obs = Observation(
        screenshot="screen.png",
        ocr_result=ocr_result,
    )

    assert obs.ocr_result == ocr_result


def test_observation_metadata():
    obs = Observation(
        screenshot="screen.png",
        metadata={
            "source": "mss",
            "region": None,
        },
    )

    assert obs.metadata["source"] == "mss"