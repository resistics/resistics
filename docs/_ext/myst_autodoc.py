"""Allow standard Sphinx autodoc to render selected docstrings as MyST.

Sphinx autodoc generates reStructuredText object directives and normally parses
their docstrings as reStructuredText too.  During the migration we wrap only
the selected docstrings in a nested MyST directive, leaving autodoc responsible
for discovery, signatures, aliases, overloads, and source links.
"""

from __future__ import annotations

from re import Pattern
from re import compile as compile_pattern
from typing import TYPE_CHECKING, Any, ClassVar

from docutils import nodes
from docutils.parsers.rst import Parser
from myst_parser.mocking import MockStateMachine
from myst_parser.parsers.sphinx_ import MystParser
from sphinx.util.docutils import SphinxDirective, new_document

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sphinx.application import Sphinx


class MystDocstringDirective(SphinxDirective):
    """Parse the directive body as MyST and return its document nodes."""

    has_content = True
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec: ClassVar[dict[str, Any]] = {}

    def run(self) -> list[nodes.Node]:
        """Render the nested MyST content in the current Sphinx environment."""
        source, _ = self.get_source_info()
        # Autodoc identifies docstring input as ``path:docstring of object``.
        # Matplotlib derives plot filenames from the document source, where
        # those spaces would produce invalid image targets.  Keep the real
        # source path and let the plot directive's per-document counter make
        # filenames unique.
        source_path = source.partition(":docstring of ")[0]
        document = new_document(source_path, self.state.document.settings)
        MystParser().parse("\n".join(self.content), document)
        return list(document.children)


def _parse_inserted_rst(
    state_machine: MockStateMachine,
    input_lines: Iterable[str],
    source: str,
) -> None:
    """Parse RST inserted by a directive running inside a MyST document.

    Matplotlib's plot directive emits generated reStructuredText through
    ``state_machine.insert_input``.  MyST's state-machine adapter intentionally
    lacks that method, so parse the generated fragment into the active node.
    """
    document = new_document(source, state_machine.document.settings)
    Parser().parse("\n".join(input_lines), document)
    state_machine.node.extend(document.children)


def _compile_parser_rules(app: Sphinx) -> list[tuple[Pattern[str], str]]:
    """Compile the configured per-object parser rules once per build."""
    rules = [
        (compile_pattern(pattern), parser)
        for pattern, parser in app.config.myst_autodoc_docstring_parser_regexes
    ]
    app.env.temp_data["myst_autodoc_parser_rules"] = rules
    return rules


def _parser_for_name(app: Sphinx, name: str) -> str:
    """Return the first configured parser matching an autodoc object name."""
    rules = app.env.temp_data.get("myst_autodoc_parser_rules")
    if rules is None:
        rules = _compile_parser_rules(app)
    for pattern, parser in rules:
        if pattern.fullmatch(name):
            return parser
    return "rst"


def _wrap_myst_docstring(
    app: Sphinx,
    _what: str,
    name: str,
    _obj: object,
    _options: object,
    lines: list[str],
) -> None:
    """Wrap selected autodoc content in the nested MyST directive."""
    if not lines or _parser_for_name(app, name) != "myst":
        return
    lines[:] = [".. myst-docstring::", "", *(f"   {line}" for line in lines)]


def setup(app: Sphinx) -> dict[str, bool]:
    """Register the directive, parser routing, and plot compatibility hook."""
    app.add_config_value(
        "myst_autodoc_docstring_parser_regexes",
        [(r".*", "rst")],
        "env",
    )
    app.add_directive("myst-docstring", MystDocstringDirective)
    app.connect("autodoc-process-docstring", _wrap_myst_docstring)

    # Remove this compatibility assignment once MyST implements insert_input.
    MockStateMachine.insert_input = _parse_inserted_rst  # type: ignore[attr-defined]
    return {"parallel_read_safe": True, "parallel_write_safe": True}
