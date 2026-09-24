#!/usr/bin/env python
"""提交前自查：在仓库内以「与 CI 一致的口径」跑 Ruff。.

= 为什么需要它（代码审查报告 M-6）=

CI 通过 ``.github/workflows/format.yml`` → ``ultralytics/actions`` 执行
Ruff（外加 docformatter / prettier / codespell），而仓库此前**既没装 Ruff、也没有任何配置**，
于是本地口径与 CI 口径静默分叉：

* 本地 ``ruff`` 回落到**默认 88 列 + 版本相关默认规则集** → ``ruff format --check .`` 报
  235 / 271 个文件需重排（绝大多数是本就合规的上游代码）；
* CI 按上游 ``pyproject.toml`` 的 120 列执行 → 只会改动本项目自有/改过的那些 ``.py``；
* 没有任何「提交前门禁」，可自动修复的问题长期累积。

本脚本把口径固定在仓库根的 ``pyproject.toml``（``[tool.ruff]`` 等，**唯一真源**），并打印
**实际解析到的设置**（列宽 / 目标版本）—— 因为「配置文件存在」不等于「配置被读到」
（父目录配置、拼错表名、``--isolated`` 都会让它回落到默认）。只有把生效值打出来，
才能一眼看出自己是不是在「用 88 列自查、然后被 CI 用 120 列改写」。

= 用法 =

    python pytools/lint.py                 # 报告：口径指纹 + 逐规则统计 + 格式检查（不改文件）
    python pytools/lint.py --strict        # 门禁：有任何发现即退出码 1（可用于 git hook / CI）
    python pytools/lint.py --fix           # 应用 Ruff 安全修复 + ruff format（**会改文件**）
    python pytools/lint.py --show-config   # 只看口径指纹（版本 / 列宽 / 目标版本）

``--ruff <path>`` 或环境变量 ``RUFF`` 可指定 Ruff 可执行文件。

= 不要「顺手」做全量 ``--fix`` =

``ruff check --fix`` 会连带删掉 RUF100 认定的「无用 noqa」，而本项目在若干处**有意**保留
``# noqa: <不在 select 列表里的规则>``（如 ``base.py`` 的 ``# noqa: BLE001``）作为
「此处有意为之」的说明；删掉它等于**静默移除一条校验**（与 M-2 同类问题）。同理 RET5xx 的
自动修复大批落在上游文件上，属无谓改动。故 ``--fix`` 之后必须人工复核 ``git diff``。
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "pyproject.toml"

# PATH 上找不到 ruff 时的兜底（本机托管环境）。与其静默跳过检查，不如给出一条本机可用的
# 确定路径；换机器时用 --ruff 或环境变量 RUFF 覆盖。
_FALLBACK_RUFF = Path.home() / ".workbuddy" / "binaries" / "python" / "envs" / "default" / "Scripts" / "ruff.exe"

# 从 `ruff check --show-settings` 里提取的关键项 —— 用来证明「配置真的生效了」
_FINGERPRINT_KEYS = ("linter.line_length", "linter.unresolved_target_version")
_CONCISE_RULE = re.compile(r":\d+:\d+: ([A-Z]+\d+)")
_FORMAT_SUMMARY = re.compile(r"(\d+) files? would be reformatted")
_FORMAT_PATH = re.compile(r"^\s*--> (.+?):\d+:\d+", re.MULTILINE)


def _digits(text: object) -> str:
    """把 ``py38`` / ``3.8`` 归一成 ``38``：Ruff 打印 ``3.8`` 而 pyproject 写 ``py38``。."""
    return re.sub(r"\D", "", str(text))


def ruff_path(explicit: str | None = None) -> Path:
    """定位 Ruff 可执行文件；找不到时 :class:`SystemExit`（**绝不静默跳过**）。.

    ``--ruff`` / 环境变量 ``RUFF`` 属**用户显式指定**：指向不存在的文件时直接报错， 不再回落到 PATH —— 否则"我明明指了路径"会静默变成"用了另一份 ruff"，口径又不可知。
    """
    for source, raw in (("--ruff", explicit), ("环境变量 RUFF", os.environ.get("RUFF"))):
        if raw:
            path = Path(raw)
            if not path.is_file():
                raise SystemExit(f"{source} 指定的 Ruff 不存在：{path}（显式指定不会再回落到 PATH）")
            print(f"ruff  : {path}（来源：{source}）")
            return path
    for source, raw in (("PATH", shutil.which("ruff")), ("托管环境兜底", str(_FALLBACK_RUFF))):
        if raw and Path(raw).is_file():
            print(f"ruff  : {raw}（来源：{source}）")
            return Path(raw)
    raise SystemExit(
        "未找到 Ruff 可执行文件。任选其一后重试：\n"
        "  - pip install ruff\n"
        "  - python pytools/lint.py --ruff <ruff 可执行文件绝对路径>\n"
        "  - 设置环境变量 RUFF=<同上>\n"
        "（本脚本不会「跳过」检查后报告成功 —— 静默跳过等于门禁不存在。）"
    )


def _run(ruff: Path, args: list[str]) -> subprocess.CompletedProcess:
    """在仓库根执行 ruff 并捕获输出（打印完整命令，便于复现）。."""
    cmd = [str(ruff), *args]
    print(f"\n$ {shlex.join(cmd)}")
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def config_expectations() -> dict:
    """读仓库根 ``pyproject.toml`` 里**我们希望生效**的值（唯一真源，不在此另抄一份）。."""
    with CONFIG_PATH.open("rb") as handle:
        data = tomllib.load(handle)
    ruff_cfg = data.get("tool", {}).get("ruff", {})
    return {
        "line-length": ruff_cfg.get("line-length"),
        "target-version": ruff_cfg.get("target-version"),
        "extend-exclude": ruff_cfg.get("extend-exclude", []),
        "select": ruff_cfg.get("lint", {}).get("select", []),
        "ignore": ruff_cfg.get("lint", {}).get("ignore", []),
    }


def resolved_settings(ruff: Path) -> dict[str, str]:
    """Ruff **实际生效**的关键设置 —— M-6 那类静默分叉的守卫。."""
    proc = _run(ruff, ["check", "--no-cache", "--show-settings", "pytools/lint.py"])
    found: dict[str, str] = {}
    for key in _FINGERPRINT_KEYS:
        match = re.search(rf"^{re.escape(key)} = (.+)$", proc.stdout or "", re.MULTILINE)
        if match:
            found[key] = match.group(1).strip().strip('"')
    return found


def show_fingerprint(ruff: Path) -> bool:
    """打印口径指纹并与 pyproject.toml 对照；返回 ``True`` 表示两者一致。."""
    expected = config_expectations()
    actual = resolved_settings(ruff)
    line = actual.get("linter.line_length", "<未解析到>")
    target = actual.get("linter.unresolved_target_version", "<未解析到>")
    print("\n== 口径指纹（本地自查口径必须 == CI 口径）==")
    print(f"  pyproject.toml: line-length={expected['line-length']} target-version={expected['target-version']}")
    print(f"  ruff 实际生效 : line-length={line} target-version={target}")
    consistent = str(expected["line-length"]) == str(line) and _digits(expected["target-version"]) == _digits(target)
    if consistent:
        print("  [OK] 一致")
    else:
        print("  [!!] 不一致：配置没有被读到（或已被改坏）—— 本地会以与 CI 不同的口径自查。")
    return consistent


def _rule_counts(concise_stdout: str) -> dict[str, int]:
    """把 concise 输出按规则码计数（自己数，不依赖 Ruff 汇总文案的措辞）。."""
    counts: dict[str, int] = {}
    for line in concise_stdout.splitlines():
        match = _CONCISE_RULE.search(line)
        if match:
            code = match.group(1)
            counts[code] = counts.get(code, 0) + 1
    return counts


def _report_external_tools() -> None:
    """Docformatter / codespell 也在 CI 里跑；缺失时**明确说"已跳过"**而不是假装通过。."""
    print("\n== CI 的其余步骤（本地可选）==")
    for tool in ("docformatter", "codespell"):
        exe = shutil.which(tool)
        if exe:
            print(f"  {tool}: 已安装 → {exe}（未自动执行；CI 会跑，需要时自行调用）")
        else:
            print(f"  {tool}: 未安装 → 本次**已跳过**（CI 仍会执行，勿据此认为通过）")


def _apply_fix(ruff: Path) -> None:
    """应用 Ruff 安全修复 + 格式化，并提醒复核。."""
    _run(ruff, ["check", "--no-cache", "--fix", "."])
    _run(ruff, ["format", "--no-cache", "."])
    print(
        "\n[注意] 已应用修复，请务必 `git diff` 人工复核：\n"
        "  - RUF100 会删掉「无用 noqa」，而本项目有意保留的 `# noqa: <不在 select 的规则>` 属此类；\n"
        "  - RET5xx 等风格修复会落到上游文件上，确认是否属于本次改动的范围。"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="仓库提交前自查（Ruff 口径取仓库根 pyproject.toml）")
    parser.add_argument("--ruff", help="Ruff 可执行文件路径（默认 PATH / 环境变量 RUFF / 托管环境兜底）")
    parser.add_argument("--fix", action="store_true", help="应用 Ruff 安全修复并格式化（会改文件，之后请复核 diff）")
    parser.add_argument("--strict", action="store_true", help="有任何发现即退出码 1（默认只报告）")
    parser.add_argument("--show-config", action="store_true", help="只打印口径指纹后退出")
    args = parser.parse_args(argv)

    if not CONFIG_PATH.is_file():
        raise SystemExit(f"缺少 {CONFIG_PATH}：本地将回落 Ruff 默认口径（88 列），正是 M-6 的根因。")

    ruff = ruff_path(args.ruff)
    print(f"  {(_run(ruff, ['--version']).stdout or '').strip()}")
    consistent = show_fingerprint(ruff)
    if args.show_config:
        return 0 if consistent else 1
    if not consistent:
        print("\n[警告] 口径不一致，以下统计来自**与 CI 不同**的口径，仅供参考。")

    if args.fix:
        _apply_fix(ruff)

    check = _run(ruff, ["check", "--no-cache", "--output-format", "concise", "."])
    counts = _rule_counts(check.stdout or "")
    total = sum(counts.values())
    print(f"\n== ruff check：{total} 项发现 ==")
    for code, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {n:4d}  {code}")
    if not counts:
        print("  （无发现）")

    fmt = _run(ruff, ["format", "--no-cache", "--check", "--quiet", "."])
    summary = _FORMAT_SUMMARY.search(fmt.stdout or "")
    unformatted = int(summary.group(1)) if summary else len(set(_FORMAT_PATH.findall(fmt.stdout or "")))
    print(f"\n== ruff format --check：{unformatted} 个文件需要重排 ==")
    for path in sorted(set(_FORMAT_PATH.findall(fmt.stdout or ""))):
        print(f"  {path}")

    _report_external_tools()

    dirty = total + unformatted
    print(f"\n== 结论：{dirty} 项待处理（{total} lint + {unformatted} 格式）==")
    if not args.strict:
        return 0
    return 1 if dirty else 0


if __name__ == "__main__":
    sys.exit(main())
