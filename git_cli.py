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
LANG = "zh"


def tr(zh: str, en: str) -> str:
    return zh if LANG == "zh" else en


def detect_default_lang() -> str:
    lc = (
        os.environ.get("LC_ALL")
        or os.environ.get("LC_MESSAGES")
        or os.environ.get("LANG")
        or ""
    ).lower()
    if lc.startswith("zh"):
        return "zh"
    return "en"


def set_git_cwd(path: str) -> None:
    global GIT_CWD
    GIT_CWD = os.path.abspath(path)


def set_language(lang: str) -> None:
    global LANG
    if lang in {"zh", "en"}:
        LANG = lang


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
        print(f"[ERROR] {tr('目录不是 git 仓库', 'Not a git repository')}: {repo}", file=sys.stderr)
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
        entered = input(
            f"{tr('目标仓库路径（回车使用当前仓库根目录）', 'Target repository path (Enter to use current repo root)')}[{default_root}]: "
        ).strip()
        selected = os.path.abspath(entered or default_root)
        if in_git_repo(selected):
            return get_repo_root(selected)
        return selected

    while True:
        entered = input(tr("当前目录不是 git 仓库，请输入目标仓库路径: ", "Current directory is not a git repository, enter target repo path: ")).strip()
        if not entered:
            print(tr("[ERROR] 请输入有效路径。", "[ERROR] Please enter a valid path."))
            continue
        candidate = os.path.abspath(entered)
        if not os.path.isdir(candidate):
            print(f"[ERROR] {tr('路径不存在或不是目录', 'Path does not exist or is not a directory')}: {candidate}")
            continue
        if not in_git_repo(candidate):
            print(f"[ERROR] {tr('不是 git 仓库', 'Not a git repository')}: {candidate}")
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


def normalize_commit_message(raw: str) -> str:
    msg = raw.strip()
    if len(msg) >= 2 and ((msg[0] == msg[-1] == '"') or (msg[0] == msg[-1] == "'")):
        msg = msg[1:-1].strip()
    return msg


def has_staged_changes() -> bool:
    # git diff --cached --quiet: 0 => no staged changes, 1 => has staged changes
    return run_git(["diff", "--cached", "--quiet"], check=False).returncode != 0


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


def prompt_text(
    prompt: str,
    *,
    default: str | None = None,
    strip: bool = True,
    allow_esc_cancel: bool = True,
) -> str | None:
    """
    Read one line from terminal.
    - Enter: submit
    - Esc: cancel (returns None) when allow_esc_cancel=True
    """
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raw = input(prompt)
        text = raw.strip() if strip else raw
        if not text and default is not None:
            return default
        return text

    print(prompt, end="", flush=True)
    buf: List[str] = []
    while True:
        key = read_key()
        if key == "ENTER":
            print()
            text = "".join(buf)
            if strip:
                text = text.strip()
            if not text and default is not None:
                return default
            return text
        if key == "ESC" and allow_esc_cancel:
            print()
            return None
        if key.startswith("CHAR:"):
            ch = key[5:]
            if ch in {"\x08", "\x7f"}:
                if buf:
                    buf.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            if ch and ch.isprintable():
                buf.append(ch)
                sys.stdout.write(ch)
                sys.stdout.flush()


def select_with_arrows(title: str, options: Sequence[str], hint: str) -> int:
    """
    Return selected index via arrow keys.
    Falls back to numeric choice when stdin/stdout is not a TTY.
    """
    if not options:
        raise ValueError(tr("options 不能为空", "options cannot be empty"))

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        clear_screen()
        print_panel(title)
        for i, opt in enumerate(options, 1):
            print(f"{i}) {opt}")
        print()
        raw = input(tr("选择操作编号: ", "Select option number: ")).strip()
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
        print(tr("使用 ↑/↓ 选择，Enter 执行，Esc 退出\n", "Use ↑/↓ to select, Enter to run, Esc to exit\n"))
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


def choose_language_menu() -> None:
    options = ["中文", "English", tr("返回", "Back")]
    idx = select_with_arrows(tr("语言", "Language"), options, tr("选择界面语言", "Choose interface language"))
    if idx == 0:
        set_language("zh")
    elif idx == 1:
        set_language("en")


def ask_yes_no(prompt: str, default_no: bool = True, allow_cancel: bool = False) -> bool | None:
    suffix = " [y/N]: " if default_no else " [Y/n]: "
    raw = prompt_text(prompt + suffix, allow_esc_cancel=True)
    if raw is None:
        if allow_cancel:
            return None
        return False
    raw = raw.strip().lower()
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


@dataclass
class BranchSyncStatus:
    branch: str
    upstream: str | None
    ahead: int
    behind: int


def get_branch_sync_status() -> BranchSyncStatus:
    branch = current_branch()
    upstream = get_upstream()
    if not upstream:
        return BranchSyncStatus(branch=branch, upstream=None, ahead=0, behind=0)

    upstream_ref = f"{upstream.remote}/{upstream.branch}"
    proc = run_git(["rev-list", "--left-right", "--count", f"{upstream_ref}...HEAD"], check=False, capture=True)
    if proc.returncode != 0:
        return BranchSyncStatus(branch=branch, upstream=upstream_ref, ahead=0, behind=0)
    raw = proc.stdout.strip().split()
    if len(raw) != 2:
        return BranchSyncStatus(branch=branch, upstream=upstream_ref, ahead=0, behind=0)
    behind = int(raw[0])
    ahead = int(raw[1])
    return BranchSyncStatus(branch=branch, upstream=upstream_ref, ahead=ahead, behind=behind)


def has_working_tree_changes() -> bool:
    proc = run_git(["status", "--porcelain"], capture=True)
    return bool(proc.stdout.strip())


def render_sync_dashboard() -> None:
    s = get_branch_sync_status()
    print_panel(tr("仓库状态总览", "Repository Overview"))
    print(f"{tr('当前分支', 'Current branch')}:         {s.branch}")
    print(f"{tr('上游分支', 'Upstream branch')}:         {s.upstream or tr('未设置 upstream', 'upstream not set')}")
    print(f"{tr('本地领先', 'Ahead')}:             {s.ahead}")
    print(f"{tr('本地落后', 'Behind')}:             {s.behind}")
    print(f"{tr('工作区有改动', 'Working tree has changes')}:     {tr('是', 'Yes') if has_working_tree_changes() else tr('否', 'No')}")
    print("=" * 56)


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


def get_branch_list_text() -> str:
    proc = run_git(["branch", "-a"], check=False, capture=True)
    text = (proc.stdout or "").strip()
    return text or tr("(无分支输出)", "(no branch output)")


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
                raise ValueError(tr(f"超出范围的编号: {idx}", f"Index out of range: {idx}"))
            picked.add(files[idx - 1])
            continue
        if "-" in tk:
            left, right = tk.split("-", 1)
            if not (left.isdigit() and right.isdigit()):
                raise ValueError(tr(f"非法区间: {tk}", f"Invalid range: {tk}"))
            a, b = int(left), int(right)
            if a > b or a < 1 or b > len(files):
                raise ValueError(tr(f"非法区间: {tk}", f"Invalid range: {tk}"))
            for i in range(a, b + 1):
                picked.add(files[i - 1])
            continue
        raise ValueError(tr(f"非法选择: {tk}", f"Invalid selection: {tk}"))

    return [f for f in files if f in picked]


def stage_interactive() -> bool:
    files = get_changed_files()
    if not files:
        print(tr("没有待暂存的改动。", "No changes to stage."))
        return False

    print(tr("可暂存文件：", "Files available to stage:"))
    for i, f in enumerate(files, 1):
        print(f"  {i:2d}) {f}")
    print(tr("输入示例: 1,3,5-8 或 a", "Example: 1,3,5-8 or a"))
    print(tr("直接回车默认选择 a", "Press Enter to use default a"))
    expr_raw = prompt_text(tr("选择要暂存的文件: ", "Select files to stage: "), default="a")
    if expr_raw is None:
        print(tr("已取消。", "Canceled."))
        return False
    expr = expr_raw.strip()
    try:
        selected = parse_selection(expr, files)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return False

    if not selected:
        print(tr("未选择任何文件。", "No files selected."))
        return False

    run_git(["add", "--", *selected])
    print(tr(f"已暂存 {len(selected)} 个文件。", f"Staged {len(selected)} files."))
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
    msg = normalize_commit_message(args.message)
    if not msg:
        print("[ERROR] commit message 不能为空。", file=sys.stderr)
        sys.exit(1)
    if not has_staged_changes():
        print("没有已暂存改动，跳过 commit。")
        return
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


def quickpush_flow() -> tuple[bool, str]:
    print(tr("== Push 向导 ==", "== Push Wizard =="))
    render_sync_dashboard()
    sync = get_branch_sync_status()
    dirty = has_working_tree_changes()

    if dirty:
        print(tr("当前有工作区改动，可选操作：", "Working tree has changes. Choose an action:"))
        print("  1) add -> commit -> push")
        if sync.ahead > 0:
            print(tr("  2) 只推送已提交内容", "  2) push committed changes only"))
            print(tr("  3) 取消", "  3) cancel"))
            mode_raw = prompt_text(tr("请选择 [1/2/3]: ", "Select [1/2/3]: "))
            mode = (mode_raw or "").strip()
        else:
            print(tr("  2) 取消", "  2) cancel"))
            mode_raw = prompt_text(tr("请选择 [1/2]: ", "Select [1/2]: "))
            mode = (mode_raw or "").strip()
            mode = "3" if mode == "2" else mode
    else:
        if sync.ahead > 0:
            print(tr("工作区干净，检测到本地有未推送提交。", "Working tree is clean and local commits are ahead."))
            print(tr("  1) 立即推送到远程", "  1) push to remote now"))
            print(tr("  2) 取消", "  2) cancel"))
            mode_raw = prompt_text(tr("请选择 [1/2]: ", "Select [1/2]: "))
            mode = (mode_raw or "").strip()
            mode = "2" if mode == "2" else "push_only"
        else:
            return False, tr("工作区干净且没有领先提交：当前没有可推送内容。", "Working tree is clean and there is nothing to push.")

    if mode in {"3", ""}:
        return False, tr("已取消。", "Canceled.")

    created_new_commit = False
    if mode == "1":
        stage_mode_raw = prompt_text(tr("暂存方式 [1]全部 [2]按文件选择 [3]取消: ", "Stage mode [1]all [2]select files [3]cancel: "))
        stage_mode = (stage_mode_raw or "").strip()
        if stage_mode == "1":
            run_git(["add", "-A"])
        elif stage_mode == "2":
            if not stage_interactive():
                return False, tr("未暂存任何文件。", "No files staged.")
        else:
            return False, tr("已取消。", "Canceled.")

        if not has_staged_changes():
            if sync.ahead > 0:
                if not ask_yes_no(tr("没有新的 staged 改动，仅推送已有提交，继续吗？", "No new staged changes, push existing commits only?"), default_no=False):
                    return False, tr("已取消。", "Canceled.")
            else:
                return False, tr("没有已暂存改动，退出。", "No staged changes. Exit.")
        else:
            run_git(["diff", "--cached", "--stat"])
            if not ask_yes_no(tr("确认提交这些改动吗？", "Commit these staged changes?"), default_no=True):
                return False, tr("已取消提交。", "Commit canceled.")

            msg_raw = prompt_text(tr("请输入 commit message: ", "Enter commit message: "))
            msg = normalize_commit_message((msg_raw or "").strip())
            if not msg:
                print(tr("[ERROR] commit message 不能为空。", "[ERROR] commit message cannot be empty."), file=sys.stderr)
                return False, tr("[ERROR] commit message 不能为空。", "[ERROR] commit message cannot be empty.")
            run_git(["commit", "-m", msg])
            created_new_commit = True

    up = get_upstream()
    br = current_branch()
    default_remote = up.remote if up else "origin"
    default_branch = up.branch if up else br
    remote_raw = prompt_text(f"Remote [{default_remote}]: ", default=default_remote)
    if remote_raw is None:
        return False, tr("已取消。", "Canceled.")
    remote = remote_raw.strip() or default_remote
    branch_raw = prompt_text(f"Remote branch [{default_branch}]: ", default=default_branch)
    if branch_raw is None:
        return False, tr("已取消。", "Canceled.")
    branch = branch_raw.strip() or default_branch

    if up and up.remote == remote and up.branch == branch:
        push_cmd = ["push"]
    else:
        push_cmd = ["push", "-u", remote, branch]

    print("\n" + tr("将推送到远程仓库:", "About to push to remote:"))
    print(f"  {tr('远程', 'Remote')}: {remote}")
    print(f"  {tr('分支', 'Branch')}: {branch}")
    print(tr("将执行命令: ", "Command: ") + "git " + " ".join(push_cmd))
    if not ask_yes_no(tr("确认推送到远程吗？", "Confirm push to remote?"), default_no=False):
        if created_new_commit:
            return False, tr("已提交到本地，但未推送到远程。", "Committed locally, not pushed to remote.")
        return False, tr("未执行远程推送。", "Remote push not executed.")

    proc = run_git(push_cmd, check=False, capture=True)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    detail = "\n".join([x for x in (out, err) if x]).strip()
    if proc.returncode != 0:
        return False, tr("推送失败。\n", "Push failed.\n") + (detail or tr("(无输出)", "(no output)"))
    return True, tr("推送完成。\n", "Push completed.\n") + (detail or tr("(无输出)", "(no output)"))


def cmd_quickpush(_: argparse.Namespace) -> None:
    ok, message = quickpush_flow()
    if ok:
        print(message)
    else:
        print(message)


def run_git_capture_text(args: Sequence[str]) -> tuple[bool, str]:
    proc = run_git(args, check=False, capture=True)
    out = (proc.stdout or "").rstrip()
    err = (proc.stderr or "").rstrip()
    parts = [x for x in (out, err) if x]
    text = "\n".join(parts) if parts else tr("(无输出)", "(no output)")
    return proc.returncode == 0, text


def tail_lines(text: str, max_lines: int = 20) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join([tr("...(已折叠前文)...", "...(previous output folded)..."), *lines[-max_lines:]])


def cmd_menu(_: argparse.Namespace) -> None:
    menu_items = [
        "status",
        "branch list",
        "switch/create branch",
        "add",
        "commit",
        "push",
        "pull --rebase",
        "log --oneline",
        "diff",
        "stash list",
        "switch repo",
        "language",
        "exit",
    ]
    last_title = tr("就绪", "Ready")
    last_output = tr("使用 ↑/↓ 选择功能，Enter 执行。", "Use ↑/↓ to choose and Enter to run.")

    while True:
        hint = f"{tr('最近结果', 'Latest result')}: {last_title}\n{tail_lines(last_output)}"
        choice_idx = select_with_arrows(tr("Git CLI 菜单", "Git CLI Menu"), menu_items, hint)

        try:
            if choice_idx == 0:
                ok, out = run_git_capture_text(["status"])
                last_title = "status"
                last_output = out if ok else f"[ERROR]\n{out}"
            elif choice_idx == 1:
                ok, out = run_git_capture_text(["branch", "-a"])
                last_title = "branch -a"
                last_output = out if ok else f"[ERROR]\n{out}"
            elif choice_idx == 2:
                clear_screen()
                print_panel("switch/create branch")
                print(tr("可切换分支列表：", "Switchable branches:"))
                print(get_branch_list_text())
                print("=" * 56)
                name_raw = prompt_text(tr("分支名: ", "Branch name: "))
                if name_raw is None:
                    last_title = "switch/create branch"
                    last_output = tr("已取消。", "Canceled.")
                    continue
                name = name_raw.strip()
                if not name:
                    last_title = "switch/create branch"
                    last_output = tr("已取消：分支名为空。", "Canceled: branch name is empty.")
                    continue
                create = ask_yes_no(
                    tr("是否新建分支并切换？", "Create and switch to a new branch?"),
                    default_no=False,
                    allow_cancel=True,
                )
                if create is None:
                    last_title = "switch/create branch"
                    last_output = tr("已取消。", "Canceled.")
                    continue
                if create:
                    ok, out = run_git_capture_text(["switch", "-c", name])
                else:
                    ok, out = run_git_capture_text(["switch", name])
                last_title = "switch/create branch"
                last_output = out if ok else f"[ERROR]\n{out}"
            elif choice_idx == 3:
                clear_screen()
                print_panel("add")
                ok = stage_interactive()
                last_title = "add"
                if not ok:
                    last_output = tr("未暂存任何文件。", "No files staged.")
                else:
                    _, out = run_git_capture_text(["diff", "--cached", "--name-only"])
                    last_output = tr("已暂存文件:\n", "Staged files:\n") + (out if out.strip() else tr("(无)", "(none)"))
            elif choice_idx == 4:
                clear_screen()
                print_panel("commit")
                msg_raw = prompt_text("commit message: ")
                if msg_raw is None:
                    last_title = "commit"
                    last_output = tr("已取消。", "Canceled.")
                    continue
                msg = normalize_commit_message(msg_raw.strip())
                if not msg:
                    last_title = "commit"
                    last_output = tr("已取消：message 为空。", "Canceled: message is empty.")
                elif not has_staged_changes():
                    last_title = "commit"
                    last_output = tr("没有已暂存改动，未执行 commit。", "No staged changes. Commit not executed.")
                else:
                    ok, out = run_git_capture_text(["commit", "-m", msg])
                    last_title = "commit"
                    last_output = out if ok else f"[ERROR]\n{out}"
            elif choice_idx == 5:
                clear_screen()
                print_panel("push")
                ok, msg = quickpush_flow()
                last_title = "push"
                _, out = run_git_capture_text(["status", "--short", "--branch"])
                prefix = tr("执行成功", "Succeeded") if ok else tr("执行结束，未完成推送", "Finished without push")
                last_output = f"{prefix}\n{msg}\n\n{out}"
            elif choice_idx == 6:
                ok, out = run_git_capture_text(["pull", "--rebase"])
                last_title = "pull --rebase"
                last_output = out if ok else f"[ERROR]\n{out}"
            elif choice_idx == 7:
                clear_screen()
                print_panel("log --oneline")
                n_raw = prompt_text(tr("显示条数 [20]: ", "Show lines [20]: "), default="20")
                if n_raw is None:
                    last_title = "log --oneline"
                    last_output = tr("已取消。", "Canceled.")
                    continue
                n = n_raw.strip() or "20"
                ok, out = run_git_capture_text(["log", f"-n{n}", "--oneline"])
                last_title = f"log -n{n} --oneline"
                last_output = out if ok else f"[ERROR]\n{out}"
            elif choice_idx == 8:
                clear_screen()
                print_panel("diff")
                cached = ask_yes_no(tr("查看 staged diff 吗？", "Show staged diff?"), default_no=True)
                cmd = ["diff", "--cached"] if cached else ["diff"]
                ok, out = run_git_capture_text(cmd)
                last_title = "diff --cached" if cached else "diff"
                last_output = out if ok else f"[ERROR]\n{out}"
            elif choice_idx == 9:
                ok, out = run_git_capture_text(["stash", "list"])
                last_title = "stash list"
                last_output = out if ok else f"[ERROR]\n{out}"
            elif choice_idx == 10:
                clear_screen()
                print_panel("switch repo")
                new_repo_raw = prompt_text(tr("输入新的仓库路径: ", "Enter new repository path: "))
                if new_repo_raw is None:
                    last_title = "switch repo"
                    last_output = tr("已取消。", "Canceled.")
                    continue
                new_repo = new_repo_raw.strip()
                if not new_repo:
                    last_title = "switch repo"
                    last_output = tr("已取消。", "Canceled.")
                    continue
                new_repo = os.path.abspath(new_repo)
                if not os.path.isdir(new_repo):
                    last_title = "switch repo"
                    last_output = f"[ERROR] {tr('路径不存在或不是目录', 'Path does not exist or is not a directory')}: {new_repo}"
                    continue
                if not in_git_repo(new_repo):
                    last_title = "switch repo"
                    last_output = f"[ERROR] {tr('不是 git 仓库', 'Not a git repository')}: {new_repo}"
                    continue
                root = get_repo_root(new_repo)
                set_git_cwd(root)
                last_title = "switch repo"
                last_output = tr(f"已切换仓库: {root}", f"Switched repo: {root}")
            elif choice_idx == 11:
                choose_language_menu()
                last_title = tr("语言", "Language")
                last_output = tr("语言已更新。", "Language updated.")
            elif choice_idx == 12:
                print("Bye.")
                return
        except RuntimeError as e:
            last_title = "执行失败"
            last_output = str(e)


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
        help="Target git repository path",
    )
    p.add_argument(
        "--lang",
        choices=["zh", "en"],
        default=None,
        help="UI language: zh or en",
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
    set_language(args.lang or detect_default_lang())
    repo_path = resolve_repo_interactive(args.repo)
    if not os.path.isdir(repo_path):
        print(f"[ERROR] {tr('路径不存在或不是目录', 'Path does not exist or is not a directory')}: {repo_path}", file=sys.stderr)
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
