"""Allow standard Sphinx autodoc to render every docstring as MyST.

Sphinx autodoc generates reStructuredText object directives and normally parses
their docstrings as reStructuredText too. We wrap their content in a nested
MyST directive, leaving autodoc responsible for discovery, signatures, aliases,
overloads, and source links.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from docutils import nodes
from docutils.parsers.rst import Parser
from docutils.parsers.rst import directives
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


class MystAutoModuleDirective(SphinxDirective):
    """Expose standard autodoc module generation through a MyST directive."""

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec: ClassVar[dict[str, Any]] = {
        "members": directives.flag,
        "undoc-members": directives.flag,
        "show-inheritance": directives.flag,
    }

    def run(self) -> list[nodes.Node]:
        """Run standard ``automodule`` in an RST document node container."""
        source, _ = self.get_source_info()
        input_lines = [f".. automodule:: {self.arguments[0]}"]
        input_lines.extend(f"   :{name}:" for name in self.options)
        document = new_document(source, self.state.document.settings)
        Parser().parse("\n".join(input_lines), document)
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


def _wrap_myst_docstring(
    _app: Sphinx,
    _what: str,
    _name: str,
    _obj: object,
    _options: object,
    lines: list[str],
) -> None:
    """Wrap autodoc content in the nested MyST directive."""
    if not lines:
        return
    lines[:] = [".. myst-docstring::", "", *(f"   {line}" for line in lines)]


def setup(app: Sphinx) -> dict[str, bool]:
    """Register the catch-all parser and plot compatibility hook."""
    app.add_directive("myst-docstring", MystDocstringDirective)
    app.add_directive("myst-automodule", MystAutoModuleDirective)
    app.connect("autodoc-process-docstring", _wrap_myst_docstring)

    # Remove this compatibility assignment once MyST implements insert_input.
    MockStateMachine.insert_input = _parse_inserted_rst  # type: ignore[attr-defined]
    return {"parallel_read_safe": True, "parallel_write_safe": True}
