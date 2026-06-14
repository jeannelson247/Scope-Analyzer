from tool import run


def test_run():
    out = run([0, 1, 2], {"demo": [0, 2, 4]})
    assert "Samples: 3" in out["text"]
    assert "peak=4" in out["text"]


if __name__ == "__main__":
    test_run()
    print("draft tool self-test passed")
