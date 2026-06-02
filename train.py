from __future__ import annotations

import sys


def _extract_stage(argv: list[str]) -> tuple[str, list[str]]:
    if not argv:
        return "gpc", argv
    valid = {"gpc", "bb2", "nnunet"}
    if argv[0] in valid:
        return argv[0], argv[1:]
    if "--stage" in argv:
        idx = argv.index("--stage")
        if idx + 1 >= len(argv):
            raise SystemExit("--stage requires one of: gpc, bb2, nnunet")
        stage = argv[idx + 1]
        if stage not in valid:
            raise SystemExit(f"unknown stage: {stage}")
        return stage, argv[:idx] + argv[idx + 2 :]
    return "gpc", argv


def main() -> None:
    stage, rest = _extract_stage(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest
    if stage == "gpc":
        from train_gpc_prior import main as run
    elif stage == "bb2":
        from train_bb2 import main as run
    elif stage == "nnunet":
        from train_nnunet import main as run
    else:
        raise SystemExit(f"unknown stage: {stage}")
    run()


if __name__ == "__main__":
    main()
