import getpass
import subprocess
import sys
from pathlib import Path


class Docker:
    """
    A little docker implementation to compose the ONO system.
    """

    def __init__(self, ono_project_root: str):
        self.__ono_project_root = Path(ono_project_root).expanduser().resolve()

    def compose_up(self, nodes: int, release: bool) -> list[str]:
        """
        Composes the nodes in a release or debug mode using the project's compose file.
        """
        COMPOSE_FILE_PATH = self.__ono_project_root / "docker" / "compose_up.py"

        args = ["sudo", "-S", sys.executable, COMPOSE_FILE_PATH, "--nodes", str(nodes)]
        if release:
            args.append("--release")

        pwd = getpass.getpass("[sudo] Password for writing simulated node addresses to the /etc/hosts file: ")
        subprocess.run(args=args, input=(pwd + "\n").encode())

        BASE_PORT = 40_000
        return [f"node-{i}:{BASE_PORT + i}" for i in range(nodes)]

    def compose_down(self) -> None:
        """
        Drops the running containers.
        """
        subprocess.run(["docker", "compose", "down"], cwd=self.__ono_project_root)
