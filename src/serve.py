"""Serve the trained Tacotron model."""

from pathlib import Path

print("[taco serve] module loaded")


def main():
    print("[taco serve] main() started")
    print("[taco serve] Serving script not yet implemented.")

    project_root = Path(__file__).resolve().parent.parent
    model_dir = project_root / "models"
    print(f"[taco serve] Looking for models in: {model_dir}")
    print("[taco serve] main() finished")


if __name__ == "__main__":
    main()
