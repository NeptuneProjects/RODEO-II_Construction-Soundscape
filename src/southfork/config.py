"""Configuration module for loading path settings with secure machine identification.
The actual machine ID is kept private in a .env file.

Usage:
    from config import paths

    data_file = paths["data_dir"] / "input.csv"

    # Check current computer (for debugging)
    from config import current_computer
    print(f"Using configuration for: {current_computer}")

Command-line usage:
    python config.py                  # Show current configuration
    python config.py info             # Show current configuration (same as above)
    python config.py set-id           # Update machine ID
    python config.py set-id -a NAME   # Update machine ID and set alias
    python config.py set-alias NAME   # Set alias without changing machine ID
"""

import os
from pathlib import Path
import tomllib
from dotenv import load_dotenv

CONFIG_FILE = Path(__file__).parents[2] / "config" / "paths.toml"
ENV_FILE = Path(__file__).parents[2] / ".env"


def create_path(key: str, exist_ok: bool = True) -> Path:
    """Create directory for a configured path if it doesn't exist"""
    path = paths[key]
    path.mkdir(parents=True, exist_ok=exist_ok)
    return path


def get_machine_alias() -> str:
    """Get computer alias from .env file or create it if missing"""
    get_or_create_env_file()
    load_dotenv(ENV_FILE)
    return os.getenv("MACHINE_ALIAS", "default")


def get_machine_id() -> str:
    """Get a persistent machine identifier that doesn't change with networks"""
    import uuid
    import platform

    if platform.system() == "Linux":
        try:
            with open("/etc/machine-id") as f:
                return f.read().strip()
        except:
            pass

    if platform.system() == "Windows":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
            ) as key:
                return winreg.QueryValueEx(key, "ProductId")[0]
        except:
            pass

    if platform.system() == "Darwin":  # macOS
        try:
            import subprocess

            return (
                subprocess.check_output(
                    ["/usr/sbin/ioreg", "-rd1", "-c", "IOPlatformExpertDevice"]
                )
                .decode("utf-8")
                .split("IOPlatformUUID")[1]
                .split('"')[2]
            )
        except:
            pass

    mac = uuid.getnode()  # Fallback to MAC address
    return f"mac-{mac}"


def get_or_create_env_file() -> None:
    """Create or verify .env file with machine ID"""
    if ENV_FILE.exists():
        return

    machine_id = get_machine_id()
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(ENV_FILE, "w") as f:
        f.write(f"# Machine identification - DO NOT PUBLISH THIS FILE\n")
        f.write(f"MACHINE_ID={machine_id}\n")
        f.write("MACHINE_ALIAS=default\n")

    print(f"Created new .env file with machine ID: {machine_id}")
    print(f"Please edit {ENV_FILE} and set your MACHINE_ALIAS.")
    print(f"This file contains private information and should not be published.")


def get_path(key, default=None, validate_exists=False):
    """
    Get a path with validation options

    Args:
        key: The path key from the config
        default: Default path if key not found (string or Path)
        validate_exists: If True, validate path exists

    Returns:
        Path object
    """
    if key not in paths and default is None:
        raise KeyError(f"Path key '{key}' not found in configuration")

    path = paths.get(key, Path(default) if default else None)

    if validate_exists and not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    return path


def load_paths() -> dict[str, Path]:
    """Load paths from TOML configuration file."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_FILE}")

    with open(CONFIG_FILE, "rb") as f:
        config = tomllib.load(f)

    path_dict = config.get("default", {})
    machine_alias = get_machine_alias()
    computer_configs = config.get("computer", {})
    if machine_alias in computer_configs:
        path_dict.update(computer_configs[machine_alias])

    return {k: Path(v) for k, v in path_dict.items()}


def print_configuration() -> None:
    """Print the current configuration details"""
    print(f"[🆔] Machine alias: {current_computer}")
    print(f"[🆔] Machine ID: {get_machine_id()}")
    print(f"[🆔] Stored in: {ENV_FILE}")
    print("Configured paths (✅ = created  ❌ = not created)")
    for name, path in paths.items():
        exists = "✅" if path.exists() else "❌"
        print(f"[{exists}] {name}: {path}")


def update_machine_id(new_alias: str | None = None) -> None:
    """
    Update the MACHINE_ID line in the .env file.
    Optionally update the MACHINE_ALIAS as well.
    Preserves all other content in the file.

    Args:
        new_alias: If provided, update the MACHINE_ALIAS to this value
    """
    if not ENV_FILE.exists():
        get_or_create_env_file()
        return

    machine_id = get_machine_id()

    with open(ENV_FILE, "r") as f:
        lines = f.readlines()

    machine_id_exists = False
    alias_exists = False

    for i, line in enumerate(lines):
        if line.strip().startswith("MACHINE_ID="):
            lines[i] = f"MACHINE_ID={machine_id}\n"
            machine_id_exists = True
        elif new_alias is not None and line.strip().startswith("MACHINE_ALIAS="):
            lines[i] = f"MACHINE_ALIAS={new_alias}\n"
            alias_exists = True

    if not machine_id_exists:
        lines.append(f"MACHINE_ID={machine_id}\n")

    if new_alias is not None and not alias_exists:
        lines.append(f"MACHINE_ALIAS={new_alias}\n")

    with open(ENV_FILE, "w") as f:
        f.writelines(lines)

    print(f"Updated machine ID to: {machine_id}")
    if new_alias is not None:
        print(f"Updated computer alias to: {new_alias}")
    else:
        print("Computer alias unchanged")
    print("All other .env content preserved")


paths = load_paths()

current_computer = get_machine_alias()

__all__ = ["paths", "current_computer", "get_path", "create_path", "update_machine_id"]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Path configuration management")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Set ID command
    update_parser = subparsers.add_parser("set-id", help="Update machine ID")
    update_parser.add_argument("--alias", "-a", help="Set computer alias")

    # Set alias command
    alias_parser = subparsers.add_parser("set-alias", help="Set computer alias")
    alias_parser.add_argument("alias", help="Computer alias to set")

    # Info command
    subparsers.add_parser("info", help="Show configuration information")

    args = parser.parse_args()

    if args.command == "set-id":
        update_machine_id(args.alias)
    elif args.command == "set-alias":
        update_machine_id(args.alias)
    elif args.command == "info" or not args.command:
        print_configuration()
    else:
        parser.print_help()
