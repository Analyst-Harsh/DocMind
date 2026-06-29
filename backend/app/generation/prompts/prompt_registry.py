from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from app.retrieval.searcher import RetrievedChunk

_PROMPTS_DIR = Path(__file__).parent
_REGISTRY_PATH = _PROMPTS_DIR / "prompt_registry.json"


@dataclass(frozen=True)
class PromptVersion:
    name: str
    version: str
    file: str
    description: str
    introduced_in: str
    eval_scores: dict[str, float | None] = field(default_factory=dict)


class PromptRegistry:
    """Maps prompt name + version → Jinja2 template + metadata.

    Version resolution: version=None always resolves to the last entry for that
    name in prompt_registry.json, making "latest" a property of insertion order
    rather than a separate field — add a new entry at the bottom to promote it.
    """

    def __init__(self, registry_path: Path = _REGISTRY_PATH) -> None:
        self._env = Environment(
            loader=FileSystemLoader(str(_PROMPTS_DIR)),
            keep_trailing_newline=True,
        )
        self._entries: list[PromptVersion] = self._load(registry_path)

    def _load(self, path: Path) -> list[PromptVersion]:
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return [PromptVersion(**entry) for entry in raw["prompts"]]

    def get(self, name: str, version: str | None = None) -> PromptVersion:
        """Return metadata for a prompt. Resolves to latest when version is None."""
        candidates = [e for e in self._entries if e.name == name]
        if not candidates:
            raise KeyError(f"No prompt named {name!r}")
        if version is None:
            return candidates[-1]
        matches = [e for e in candidates if e.version == version]
        if not matches:
            available = [e.version for e in candidates]
            raise KeyError(
                f"Prompt {name!r} has no version {version!r}. Available: {available}"
            )
        return matches[0]

    def render(self, name: str, version: str | None = None, **kwargs: Any) -> str:
        """Render a prompt template by name and optional version."""
        entry = self.get(name, version)
        # entry.file is "prompts/v1_foo.jinja2"; strip the directory prefix so
        # FileSystemLoader (rooted at _PROMPTS_DIR) can resolve the filename.
        template_name = Path(entry.file).name
        template = self._env.get_template(template_name)
        return template.render(**kwargs)

    def list_versions(self, name: str) -> list[PromptVersion]:
        return [e for e in self._entries if e.name == name]


# Module-level singleton — loaded once at import time, shared across the app.
_registry = PromptRegistry()


def get_registry() -> PromptRegistry:
    return _registry


def build_qa_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    version: str | None = None,
) -> str:
    """Render the grounded-QA prompt. Defaults to the latest registered version."""
    return _registry.render(
        "grounded_qa",
        version=version,
        question=question,
        chunks=chunks,
    )
