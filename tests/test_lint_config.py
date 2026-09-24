"""仓库 lint/format 配置的守卫（对应复审报告 M-6）。.

为什么需要静态守卫：M-6 的失效是**静默**的 —— 配置文件缺失、或列宽被改回 88 时，
不会有任何运行时报错，只会让本地 Ruff 与 CI 的口径悄悄分叉（本地报 235 / 271 个文件需重排，
而 CI 按 120 列只会动本项目自有/改过的 23 个），"提交前自查"看起来一切正常。

这里钉住三件事：

1. 配置存在性与关键值（line-length / target-version / 中文歧义规则忽略 / markdown 排除 / 显式 select）；
2. 本文件**只**是工具配置，不得长出打包元数据或继承上游的 pytest 配置；
3. 端到端：在本仓库里真跑一次 Ruff，断言它**实际解析到**的列宽（``--show-settings``）等于
   pyproject.toml 里写的值 —— 只有这一条能抓到"配置文件在、但 Ruff 没读到、回落 88"。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "pyproject.toml"
PYTOOLS = REPO / "pytools"

sys.path.insert(0, str(PYTOOLS))
import lint as lint_tool  # noqa: E402


def _config() -> dict:
    """读取仓库根 pyproject.toml（不存在时直接失败：缺它正是 M-6 的根因）。."""
    assert CONFIG.is_file(), f"缺少 {CONFIG}：本地 Ruff 会回落到默认 88 列口径（M-6 根因）"
    with CONFIG.open("rb") as handle:
        return tomllib.load(handle)


def _ruff_or_skip() -> Path:
    """本机没有 Ruff 时跳过端到端断言（静态断言仍会跑）。."""
    try:
        return lint_tool.ruff_path()
    except SystemExit:
        pytest.skip("本机未找到 Ruff，跳过端到端口径断言")


# --------------------------------------------------------------------------- 静态守卫


def test_pyproject_exists_at_repo_root():
    assert CONFIG.is_file()


def test_ruff_line_length_and_target_version_match_upstream():
    """列宽必须是上游的 120（缺它 → 回落 88，一次 PR 会重排 235 个文件）。."""
    cfg = _config()["tool"]["ruff"]
    assert cfg["line-length"] == 120
    assert cfg["target-version"] == "py38"


def test_cjk_ambiguity_rules_are_ignored():
    """中文注释里的全角标点会被判为"歧义 Unicode"，必须忽略否则纯噪声。."""
    ignore = _config()["tool"]["ruff"]["lint"]["ignore"]
    assert {"RUF001", "RUF002", "RUF003"} <= set(ignore)


def test_markdown_is_excluded_from_ruff():
    """README 的示例代码块刻意对齐了行内注释，不能被 Ruff 压成单空格。."""
    assert "*.md" in _config()["tool"]["ruff"]["extend-exclude"]


def test_lint_select_is_explicit_not_version_default():
    """显式枚举规则码：依赖 Ruff 默认集会得到"随版本变化"的自查口径。."""
    select = _config()["tool"]["ruff"]["lint"]["select"]
    assert isinstance(select, list) and select, "select 必须是非空列表"
    assert "ALL" not in select


def test_no_packaging_metadata():
    """本文件只承载工具配置：长出 [project]/[build-system] 会改变 `pip install -e .` 语义。."""
    data = _config()
    assert "project" not in data and "build-system" not in data
    assert "setuptools" not in data.get("tool", {})


def test_upstream_pytest_addopts_not_inherited():
    """上游该节含 `--doctest-modules`，会让 pytest 去收集 ultralytics/ 全部模块（慢且起冲突）。."""
    assert "pytest" not in _config().get("tool", {})


def test_ruff_cache_is_gitignored():
    """必须**行为上**真被忽略：`.gitignore` 不支持行尾注释，写错就把注释并进模式而静默失效。."""
    proc = subprocess.run(
        ["git", "check-ignore", "-q", ".ruff_cache"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 128:
        pytest.skip("当前环境没有可用的 git")
    assert proc.returncode == 0, "`.ruff_cache` 未被 .gitignore 忽略（检查是否有行尾注释被并入模式）"


# --------------------------------------------------------------------------- 端到端守卫


def test_lint_tool_locates_ruff():
    assert _ruff_or_skip().is_file()


def test_ruff_actually_resolves_config_line_length():
    """抓"配置在、但没被读到"：Ruff 实际生效的 line-length 必须等于 pyproject 里写的值。."""
    ruff = _ruff_or_skip()
    expected = _config()["tool"]["ruff"]
    actual = lint_tool.resolved_settings(ruff)
    assert str(expected["line-length"]) == str(actual.get("linter.line_length")), (
        f"Ruff 实际生效 line-length={actual.get('linter.line_length')}，"
        f"与 pyproject 的 {expected['line-length']} 不一致（配置未被读到 / 被父目录配置覆盖）"
    )
    got_target = lint_tool._digits(actual.get("linter.unresolved_target_version"))
    assert got_target == lint_tool._digits(expected["target-version"])


def test_lint_tool_show_config_exits_zero():
    """端到端：把自己当用户跑一遍入口，断言"口径一致"确实被判定出来。."""
    ruff = _ruff_or_skip()
    proc = subprocess.run(
        [sys.executable, str(PYTOOLS / "lint.py"), "--show-config", "--ruff", str(ruff)],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "[OK] 一致" in proc.stdout


def test_lint_tool_never_silently_skips_without_ruff(monkeypatch, tmp_path):
    """找不到 Ruff 必须报错退出，不能"跳过检查后报告成功"（静默跳过 = 门禁不存在）。.

    用 monkeypatch 把三个来源全部置空，避免依赖"本机恰好没装 ruff"。
    """
    monkeypatch.delenv("RUFF", raising=False)
    monkeypatch.setattr(lint_tool.shutil, "which", lambda _name: None)
    monkeypatch.setattr(lint_tool, "_FALLBACK_RUFF", tmp_path / "nope" / "ruff.exe")
    with pytest.raises(SystemExit) as excinfo:
        lint_tool.ruff_path()
    assert "未找到 Ruff" in str(excinfo.value)


def test_explicit_ruff_path_never_falls_back(monkeypatch, tmp_path):
    """显式 --ruff 指向不存在的文件时直接报错：否则会静默改用 PATH 上的另一份 Ruff。."""
    monkeypatch.delenv("RUFF", raising=False)
    monkeypatch.setattr(lint_tool.shutil, "which", lambda _name: str(tmp_path / "other_ruff.exe"))
    with pytest.raises(SystemExit) as excinfo:
        lint_tool.ruff_path(str(tmp_path / "does_not_exist"))
    assert "不存在" in str(excinfo.value)


def test_external_ci_steps_are_reported_as_skipped_not_passed(capsys):
    """Docformatter / codespell 缺失时必须显式说「已跳过」，而不是从报告里消失。."""
    lint_tool._report_external_tools()
    out = capsys.readouterr().out
    for tool in ("docformatter", "codespell"):
        assert tool in out
        if shutil.which(tool) is None:
            assert f"{tool}: 未安装" in out and "已跳过" in out
