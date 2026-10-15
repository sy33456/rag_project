"""文档加载：把原始文件变成 (文本, 来源) 列表。"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple


def load_file(path: str | Path) -> List[Tuple[str, str]]:
    """加载单个 .txt / .md 文件，返回 [(文本, 来源名), ...]。"""
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    return [(text, p.name)]


def load_dir(dir_path: str | Path) -> List[Tuple[str, str]]:
    """加载目录下所有 .txt / .md。"""
    p = Path(dir_path)
    docs: List[Tuple[str, str]] = []
    for f in sorted(p.glob("*")):
        if f.suffix.lower() in (".txt", ".md"):
            docs.extend(load_file(f))
    return docs
