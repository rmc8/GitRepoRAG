import os
import tomllib

from dotenv import load_dotenv

from libs.rag import Rag

this_dir = os.path.abspath(os.path.dirname(__file__))
env_path = os.path.join(this_dir, ".env")
load_dotenv(dotenv_path=env_path)


def get_settings() -> dict:
    settings_path = os.path.join(this_dir, "settings.toml")
    with open(settings_path, "rb") as f:
        return tomllib.load(f)


def main() -> None:
    settings = get_settings()
    rag = Rag(settings)
    rag.run()


if __name__ == "__main__":
    main()
