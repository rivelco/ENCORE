def test_import():
    import encore

def test_entry_point():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "encore", "--version"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    