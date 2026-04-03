#!/usr/bin/env python3
"""
Cross-platform Git CLI helper (interactive + subcommands).

Design goals:
1) Reduce memorization burden for frequent git commands.
2) Keep behavior explicit and safe (no hidden destructive operations).
3) Run on Windows/macOS/Linux with Python standard library only.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Sequence

GIT_CWD = os.getcwd()


def set_git_cwd(path: str) -> None:
    global GIT_CWD
    GIT_CWD = os.path.abspath(path)


def run_git(
    args: Sequence[str],
    *,
    check: bool = True,
    capture: bool = False,
    cwd: str | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = ["git", "--no-pager", *args]
    env = os.environ.copy()
    # Disable pager to avoid hanging in interactive-less contexts.
    env["GIT_PAGER"] = "cat"
    env["PAGER"] = "cat"
    kwargs = {
        "text": True,
        "cwd": cwd or GIT_CWD,
        "env": env,
    }
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    proc = subprocess.run(cmd, **kwargs)  # type: ignore[arg-type]
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed: {shlex.join(cmd)}")
    return proc


def in_git_repo(path: str | None = None) -> bool:
    proc = run_git(["rev-parse", "--is-inside-work-tree"], check=False, capture=True, cwd=path)
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def ensure_git_repo(path: str | None = None) -> None:
    repo = os.path.abspath(path or GIT_CWD)
    if not in_git_repo(repo):
        print(f"[ERROR] 目录不是 git 仓库: {repo}", file=sys.stderr)
        sys.exit(1)


def get_repo_root(path: str) -> str:
    proc = run_git(["rev-parse", "--show-toplevel"], capture=True, cwd=path)
    return proc.stdout.strip()


def resolve_repo_interactive(repo_arg: str | None) -> str:
    """
    Resolve target repo path with a user-friendly flow:
    1) If -C/--repo provided: use it directly.
    2) If current dir is a repo: default to current dir, allow override.
    3) If current dir is not a repo: prompt user to input a repo path.
    """
    if repo_arg:
        p = os.path.abspath(repo_arg)
        if in_git_repo(p):
            return get_repo_root(p)
        return p

    cwd = os.getcwd()
    if in_git_repo(cwd):
        default_root = get_repo_root(cwd)
        entered = input(f"目标仓库路径（回车使用当前仓库根目录）[{default_root}]: ").strip()
        selected = os.path.abspath(entered or default_root)
        if in_git_repo(selected):
            return get_repo_root(selected)
        return selected

    while True:
        entered = input("当前目录不是 git 仓库，请输入目标仓库路径: ").strip()
        if not entered:
            print("[ERROR] 请输入有效路径。")
            continue
        candidate = os.path.abspath(entered)
        if not os.path.isdir(candidate):
            print(f"[ERROR] 路径不存在或不是目录: {candidate}")
            continue
        if not in_git_repo(candidate):
            print(f"[ERROR] 不是 git 仓库: {candidate}")
            continue
        return get_repo_root(candidate)


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def print_panel(title: str, body: str | None = None) -> None:
    line = "=" * 56
    print(line)
    print(title)
    print(line)
    if body:
        print(body)
        print(line)


def wait_back_to_menu() -> None:
    input("\n按回车返回菜单...")


def read_key() -> str:
    """Read one key and normalize to UP/DOWN/ENTER/ESC/CHAR:*."""
    if os.name == "nt":
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ("\r", "\n"):
            return "ENTER"
        if ch in ("\x00", "\xe0"):
            nxt = msvcrt.getwch()
            if nxt == "H":
                return "UP"
            if nxt == "P":
                return "DOWN"
            return f"CHAR:{nxt}"
        if ch == "\x1b":
            return "ESC"
        return f"CHAR:{ch}"

    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch in ("\r", "\n"):
            return "ENTER"
        if ch == "\x1b":
            # Arrow keys on Unix terminals usually arrive as ESC [ A/B.
            if select.select([sys.stdin], [], [], 0.03)[0]:
                seq1 = sys.stdin.read(1)
                if seq1 == "[" and select.select([sys.stdin], [], [], 0.03)[0]:
                    seq2 = sys.stdin.read(1)
                    if seq2 == "A":
                        return "UP"
                    if seq2 == "B":
                        return "DOWN"
            return "ESC"
        return f"CHAR:{ch}"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def select_with_arrows(title: str, options: Sequence[str], hint: str) -> int:
    """
    Return selected index via arrow keys.
    Falls back to numeric choice when stdin/stdout is not a TTY.
    """
    if not options:
        raise ValueError("options 不能为空")

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        clear_screen()
        print_panel(title)
        for i, opt in enumerate(options, 1):
            print(f"{i}) {opt}")
        print()
        raw = input("选择操作编号: ").strip()
        if not raw.isdigit():
            return 0
        idx = int(raw) - 1
        if idx < 0 or idx >= len(options):
            return 0
        return idx

    idx = 0
    while True:
        clear_screen()
        print_panel(title, f"Repo: {GIT_CWD}")
        print("使用 ↑/↓ 选择，Enter 执行，Esc 退出\n")
        for i, opt in enumerate(options):
            prefix = "›" if i == idx else " "
            print(f"{prefix} {opt}")
        print(f"\n{hint}")
        key = read_key()
        if key == "UP":
            idx = (idx - 1) % len(options)
        elif key == "DOWN":
            idx = (idx + 1) % len(options)
        elif key == "ENTER":
            return idx
        elif key == "ESC":
            return len(options) - 1


def ask_yes_no(prompt: str, default_no: bool = True) -> bool:
    suffix = " [y/N]: " if default_no else " [Y/n]: "
    raw = input(prompt + suffix).strip().lower()
    if not raw:
        return not default_no
    return raw in {"y", "yes"}


def current_branch() -> str:
    proc = run_git(["branch", "--show-current"], capture=True)
    return proc.stdout.strip()


@dataclass
class Upstream:
    remote: str
    branch: str


def get_upstream() -> Upstream | None:
    proc = run_git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], check=False, capture=True)
    if proc.returncode != 0:
        return None
    raw = proc.stdout.strip()
    if "/" not in raw:
        return None
    remote, branch = raw.split("/", 1)
    return Upstream(remote=remote, branch=branch)


def get_changed_files() -> List[str]:
    proc = run_git(["status", "--porcelain"], capture=True)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    files: List[str] = []
    seen = set()
    for ln in lines:
        path = ln[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path not in seen:
            seen.add(path)
            files.append(path)
    return files


def parse_selection(expr: str, files: Sequence[str]) -> List[str]:
    if expr.strip().lower() == "a":
        return list(files)

    picked = set()
    tokens = [x.strip() for x in expr.split(",") if x.strip()]
    if not tokens:
        return []

    for tk in tokens:
        if tk.isdigit():
            idx = int(tk)
            if idx < 1 or idx > len(files):
                raise ValueError(f"超出范围的编号: {idx}")
            picked.add(files[idx - 1])
            continue
        if "-" in tk:
            left, right = tk.split("-", 1)
            if not (left.isdigit() and right.isdigit()):
                raise ValueError(f"非法区间: {tk}")
            a, b = int(left), int(right)
            if a > b or a < 1 or b > len(files):
                raise ValueError(f"非法区间: {tk}")
            for i in range(a, b + 1):
                picked.add(files[i - 1])
            continue
        raise ValueError(f"非法选择: {tk}")

    return [f for f in files if f in picked]


def stage_interactive() -> bool:
    files = get_changed_files()
    if not files:
        print("没有待暂存的改动。")
        return False

    print("可暂存文件：")
    for i, f in enumerate(files, 1):
        print(f"  {i:2d}) {f}")
    print("输入示例: 1,3,5-8 或 a(全部)")
    expr = input("选择要暂存的文件: ").strip()
    try:
        selected = parse_selection(expr, files)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return False

    if not selected:
        print("未选择任何文件。")
        return False

    run_git(["add", "--", *selected])
    print(f"已暂存 {len(selected)} 个文件。")
    return True


def cmd_status(_: argparse.Namespace) -> None:
    run_git(["status"])


def cmd_branch(args: argparse.Namespace) -> None:
    if args.all:
        run_git(["branch", "-a"])
    else:
        run_git(["branch"])


def cmd_switch(args: argparse.Namespace) -> None:
    if args.create:
        run_git(["switch", "-c", args.branch])
    else:
        run_git(["switch", args.branch])


def cmd_add(args: argparse.Namespace) -> None:
    if args.interactive:
        ok = stage_interactive()
        if not ok:
            sys.exit(1)
        return
    if args.all:
        run_git(["add", "-A"])
        return
    if not args.paths:
        print("[ERROR] add 需要 --all、--interactive 或提供具体路径。", file=sys.stderr)
        sys.exit(1)
    run_git(["add", "--", *args.paths])


def cmd_commit(args: argparse.Namespace) -> None:
    msg = args.message.strip()
    if not msg:
        print("[ERROR] commit message 不能为空。", file=sys.stderr)
        sys.exit(1)
    run_git(["commit", "-m", msg])


def cmd_push(args: argparse.Namespace) -> None:
    if args.remote and args.branch:
        cmd = ["push"]
        if args.set_upstream:
            cmd.append("-u")
        cmd.extend([args.remote, args.branch])
        run_git(cmd)
        return
    if args.remote or args.branch:
        print("[ERROR] --remote 和 --branch 需要同时提供。", file=sys.stderr)
        sys.exit(1)
    run_git(["push"])


def cmd_pull(args: argparse.Namespace) -> None:
    cmd = ["pull"]
    if args.rebase:
        cmd.append("--rebase")
    run_git(cmd)


def cmd_log(args: argparse.Namespace) -> None:
    cmd = ["log", f"-n{args.n}"]
    if args.oneline:
        cmd.append("--oneline")
    run_git(cmd)


def cmd_diff(args: argparse.Namespace) -> None:
    cmd = ["diff"]
    if args.cached:
        cmd.append("--cached")
    if args.paths:
        cmd.extend(["--", *args.paths])
    run_git(cmd)


def cmd_stash(args: argparse.Namespace) -> None:
    action = args.action
    if action == "save":
        cmd = ["stash", "push"]
        if args.message:
            cmd.extend(["-m", args.message])
        run_git(cmd)
    elif action == "list":
        run_git(["stash", "list"])
    elif action == "pop":
        run_git(["stash", "pop"])
    elif action == "apply":
        if args.ref:
            run_git(["stash", "apply", args.ref])
        else:
            run_git(["stash", "apply"])
    elif action == "drop":
        if not args.ref:
            print("[ERROR] stash drop 需要 --ref，例如 stash@{0}", file=sys.stderr)
            sys.exit(1)
        if not ask_yes_no(f"确认删除 {args.ref} 吗？", default_no=True):
            print("已取消。")
            return
        run_git(["stash", "drop", args.ref])
    else:
        print(f"[ERROR] 未知 stash action: {action}", file=sys.stderr)
        sys.exit(1)


def cmd_quickpush(_: argparse.Namespace) -> None:
    print("== Quick Push 向导 ==")
    run_git(["status"])
    print()

    mode = input("暂存方式: [1]全部 [2]按文件选择 [3]取消: ").strip()
    if mode == "1":
        run_git(["add", "-A"])
    elif mode == "2":
        if not stage_interactive():
            sys.exit(1)
    else:
        print("已取消。")
        return

    if run_git(["diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("没有已暂存改动，退出。")
        return

    run_git(["diff", "--cached", "--stat"])
    if not ask_yes_no("确认提交这些改动吗？", default_no=True):
        print("已取消。")
        return

    msg = input("请输入 commit message: ").strip()
    if not msg:
        print("[ERROR] commit message 不能为空。", file=sys.stderr)
        sys.exit(1)
    run_git(["commit", "-m", msg])

    up = get_upstream()
    br = current_branch()
    default_remote = up.remote if up else "origin"
    default_branch = up.branch if up else br
    remote = input(f"Remote [{default_remote}]: ").strip() or default_remote
    branch = input(f"Remote branch [{default_branch}]: ").strip() or default_branch

    if up and up.remote == remote and up.branch == branch:
        push_cmd = ["push"]
    else:
        push_cmd = ["push", "-u", remote, branch]

    print("将执行:", "git " + " ".join(push_cmd))
    if not ask_yes_no("确认推送吗？", default_no=True):
        print("已提交到本地，未推送。")
        return

    run_git(push_cmd)
    print("推送完成。")


def cmd_menu(_: argparse.Namespace) -> None:
    menu_items = [
        "status",
        "branch list",
        "switch/create branch",
        "add (interactive)",
        "commit",
        "push (quickpush 向导)",
        "pull --rebase",
        "log --oneline",
        "diff",
        "stash list",
        "切换目标仓库",
        "exit",
    ]

    while True:
        choice_idx = select_with_arrows("Git CLI 菜单", menu_items, "提示: 在任意结果页按回车返回菜单")

        try:
            if choice_idx == 0:
                clear_screen()
                print_panel("status")
                run_git(["status"])
                wait_back_to_menu()
            elif choice_idx == 1:
                clear_screen()
                print_panel("branch -a")
                run_git(["branch", "-a"])
                wait_back_to_menu()
            elif choice_idx == 2:
                clear_screen()
                print_panel("switch/create branch")
                name = input("分支名: ").strip()
                create = ask_yes_no("是否新建分支并切换？", default_no=False)
                if create:
                    run_git(["switch", "-c", name])
                else:
                    run_git(["switch", name])
                wait_back_to_menu()
            elif choice_idx == 3:
                clear_screen()
                print_panel("add (interactive)")
                stage_interactive()
                wait_back_to_menu()
            elif choice_idx == 4:
                clear_screen()
                print_panel("commit")
                msg = input("commit message: ").strip()
                if msg:
                    run_git(["commit", "-m", msg])
                else:
                    print("message 为空，已取消。")
                wait_back_to_menu()
            elif choice_idx == 5:
                clear_screen()
                print_panel("quickpush")
                cmd_quickpush(argparse.Namespace())
                wait_back_to_menu()
            elif choice_idx == 6:
                clear_screen()
                print_panel("pull --rebase")
                run_git(["pull", "--rebase"])
                wait_back_to_menu()
            elif choice_idx == 7:
                clear_screen()
                print_panel("log --oneline")
                n = input("显示条数 [20]: ").strip() or "20"
                run_git(["log", f"-n{n}", "--oneline"])
                wait_back_to_menu()
            elif choice_idx == 8:
                clear_screen()
                print_panel("diff")
                cached = ask_yes_no("查看 staged diff 吗？", default_no=True)
                run_git(["diff", "--cached"] if cached else ["diff"])
                wait_back_to_menu()
            elif choice_idx == 9:
                clear_screen()
                print_panel("stash list")
                run_git(["stash", "list"])
                wait_back_to_menu()
            elif choice_idx == 10:
                clear_screen()
                print_panel("切换目标仓库")
                new_repo = input("输入新的仓库路径: ").strip()
                if not new_repo:
                    print("已取消。")
                    wait_back_to_menu()
                    continue
                new_repo = os.path.abspath(new_repo)
                if not os.path.isdir(new_repo):
                    print(f"[ERROR] 路径不存在或不是目录: {new_repo}")
                    wait_back_to_menu()
                    continue
                if not in_git_repo(new_repo):
                    print(f"[ERROR] 不是 git 仓库: {new_repo}")
                    wait_back_to_menu()
                    continue
                root = get_repo_root(new_repo)
                set_git_cwd(root)
                print(f"已切换仓库: {root}")
                wait_back_to_menu()
            elif choice_idx == 11:
                print("Bye.")
                return
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            wait_back_to_menu()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="git_cli.py",
        description="跨平台 Git 交互式辅助工具（减少命令记忆负担）。",
        epilog=(
            "示例:\n"
            "  python tools/git_cli.py menu\n"
            "  python tools/git_cli.py -C /path/to/repo menu\n"
            "  python tools/git_cli.py quickpush\n"
            "  python tools/git_cli.py add -i\n"
            "  python tools/git_cli.py commit -m \"update docs\"\n"
            "  python tools/git_cli.py push -r origin -b feat/demo -u"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "-C",
        "--repo",
        default=None,
        help="目标 git 仓库路径（可选；不填则启动时交互选择）",
    )
    sub = p.add_subparsers(dest="command")

    sp = sub.add_parser("status", help="显示仓库状态")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("branch", help="查看分支")
    sp.add_argument("-a", "--all", action="store_true", help="显示本地+远程分支")
    sp.set_defaults(func=cmd_branch)

    sp = sub.add_parser("switch", help="切换分支")
    sp.add_argument("branch", help="目标分支名")
    sp.add_argument("-c", "--create", action="store_true", help="创建并切换")
    sp.set_defaults(func=cmd_switch)

    sp = sub.add_parser("add", help="暂存改动")
    sp.add_argument("paths", nargs="*", help="文件路径")
    sp.add_argument("-A", "--all", action="store_true", help="暂存全部改动")
    sp.add_argument("-i", "--interactive", action="store_true", help="交互选择文件")
    sp.set_defaults(func=cmd_add)

    sp = sub.add_parser("commit", help="提交已暂存改动")
    sp.add_argument("-m", "--message", required=True, help="提交信息")
    sp.set_defaults(func=cmd_commit)

    sp = sub.add_parser("push", help="推送代码")
    sp.add_argument("-r", "--remote", help="远程名，例如 origin")
    sp.add_argument("-b", "--branch", help="远程分支名")
    sp.add_argument("-u", "--set-upstream", action="store_true", help="设置 upstream")
    sp.set_defaults(func=cmd_push)

    sp = sub.add_parser("pull", help="拉取代码")
    sp.add_argument("--rebase", action="store_true", help="使用 rebase")
    sp.set_defaults(func=cmd_pull)

    sp = sub.add_parser("log", help="查看提交历史")
    sp.add_argument("-n", type=int, default=20, help="显示条数")
    sp.add_argument("--oneline", action="store_true", default=True, help="单行显示")
    sp.set_defaults(func=cmd_log)

    sp = sub.add_parser("diff", help="查看改动")
    sp.add_argument("--cached", action="store_true", help="查看 staged diff")
    sp.add_argument("paths", nargs="*", help="指定路径")
    sp.set_defaults(func=cmd_diff)

    sp = sub.add_parser("stash", help="stash 操作")
    sp.add_argument("action", choices=["save", "list", "pop", "apply", "drop"], help="stash 子操作")
    sp.add_argument("-m", "--message", help="stash message（仅 save）")
    sp.add_argument("--ref", help="stash 引用，如 stash@{0}（apply/drop）")
    sp.set_defaults(func=cmd_stash)

    sp = sub.add_parser("quickpush", help="一键交互：add -> commit -> push")
    sp.set_defaults(func=cmd_quickpush)

    sp = sub.add_parser("menu", help="交互式主菜单")
    sp.set_defaults(func=cmd_menu)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    repo_path = resolve_repo_interactive(args.repo)
    if not os.path.isdir(repo_path):
        print(f"[ERROR] 路径不存在或不是目录: {repo_path}", file=sys.stderr)
        sys.exit(1)
    set_git_cwd(repo_path)
    ensure_git_repo(repo_path)

    # 默认进入交互菜单
    if not args.command:
        cmd_menu(argparse.Namespace())
        return

    args.func(args)


if __name__ == "__main__":
    main()
