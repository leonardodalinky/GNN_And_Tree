import os
import stat
import shutil
import argparse
from string import Template
from pathlib import Path
from datetime import datetime

import yaml

SCRIPT_DIR = Path(os.path.realpath(f"{__file__}/.."))
SRC_DIR = (SCRIPT_DIR / "..").absolute()
GENERATED_DIR = SRC_DIR / "generated"
TEMPLATE_DIR = SRC_DIR / "templates"


parser = argparse.ArgumentParser(description="Tool for generating bash config.")
parser.add_argument(
    "config_yaml",
    type=Path,
    metavar="CONFIG_PATH",
)
parser.add_argument(
    "-n",
    "--project-name",
    type=str,
    metavar="NAME",
    default=None,
)


def main():
    args = parser.parse_args()

    yaml_path: Path = Path(args.config_yaml)
    assert yaml_path.exists(), f"Could not access to config: {args.config_yaml}."
    with yaml_path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)
    template = Template((TEMPLATE_DIR / "config_bash.template").read_text(encoding="utf-8"))
    tokens = {
        "tree_name": config["model"]["tree"],
        "gnn_name": config["model"]["gnn"],
        "framework_name": config["model"]["framework"],
        "project_name": args.project_name or config["task"]["dataset"]["name"],
        "timestamp": int(datetime.now().timestamp()),
    }

    out_dir: Path = GENERATED_DIR / yaml_path.stem
    print(f"Create output dir: {out_dir}...")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generate bash scripts...")
    for stage in ("train", "validate", "test"):
        out_path: Path = out_dir / f"{stage}.sh"
        d = dict(tokens)
        d.update({"stage": stage})
        out_path.write_text(template.safe_substitute(d), encoding="utf-8")
        st = os.stat(str(out_path))
        os.chmod(str(out_path), st.st_mode | stat.S_IEXEC)

    print(f"Copy base.py...")
    shutil.copy(str((SCRIPT_DIR / "base.sh")), str(out_dir))
    print(f"Copy {yaml_path} to config.py...")
    shutil.copy(str(yaml_path), str(out_dir / "config.yml"))

    print("Succeed!")


if __name__ == main():
    main()
