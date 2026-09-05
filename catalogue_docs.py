"""Shared docstring/equation rendering helpers for the catalogue generators.

Domains, fluid equilibria, and perturbations all now expose their own
``mapping_markdown()`` / ``formula_markdown()`` classmethods (via
``struphy.geometry.base.Domain``, ``struphy.fields_background.base.FluidEquilibrium``,
and ``struphy.initial.base.Perturbation``) that already parse a class's ``..
math::`` blocks into real LaTeX. Generators should call those directly for a
class's defining equation rather than re-parsing the raw docstring themselves.

The general class description (the prose before the ``Parameters`` section)
still has to be pulled out of the docstring by hand -- struphy has no
"intro-only" accessor for that -- so `class_description` keeps doing that part.
"""

from __future__ import annotations

import inspect
import re

from struphy.utils.docstring_converter import (_extract_math_directives,
                                               rst_to_html)


def extract_math(rst: str) -> tuple[str, list[dict]]:
    """Pull raw LaTeX out of ``.. math::`` blocks and ``:math:`...``` roles.

    Each occurrence is replaced with a placeholder token so the surrounding
    RST can still go through struphy's ``rst_to_html`` unharmed. The real
    LaTeX is returned separately for the site to render with KaTeX (real
    typesetting) instead of struphy's Unicode-approximation converter, which
    exists only for VS Code hover tooltips that can't load a math renderer.
    """
    math_items: list[dict] = []

    def save_block(content: str) -> str:
        raw = "\n".join(
            line.strip() for line in content.strip().split("\n") if line.strip()
        )
        if raw.lstrip().startswith("\\begin{"):
            latex = raw
        else:
            groups = [
                " ".join(line.strip() for line in group.split("\n") if line.strip())
                for group in re.split(r"\n\s*\n", content.strip())
            ]
            needs_aligned = len(groups) > 1 or any("&" in group for group in groups)
            latex = r" \\ ".join(groups)
            if needs_aligned:
                latex = f"\\begin{{aligned}} {latex} \\end{{aligned}}"
        token = f"@@MATH{len(math_items)}@@"
        math_items.append({"token": token, "latex": latex, "display": True})
        return token

    stripped = _extract_math_directives(rst, save_block)

    def save_inline(match: re.Match) -> str:
        token = f"@@MATH{len(math_items)}@@"
        math_items.append({"token": token, "latex": match.group(1), "display": False})
        return token

    stripped = re.sub(r":math:`([^`]+)`", save_inline, stripped)
    return stripped, math_items


def class_description(cls: type) -> tuple[str, list[dict]]:
    """Convert the docstring intro (before ``Parameters``) to HTML, with LaTeX
    extracted for KaTeX rendering. Sphinx ``.. image::`` directives reference
    internal documentation assets that don't resolve here, so they are dropped.
    """
    doc = inspect.getdoc(cls) or ""
    intro = re.split(r"\nParameters\n-+\n", doc, maxsplit=1)[0]
    intro = re.sub(r"\n\.\. image::[^\n]*\n?", "\n", intro)
    if not intro.strip():
        return "", []
    stripped, math_items = extract_math(intro)
    html = rst_to_html(stripped, forced_heading_level=3).strip()
    return html, math_items


def equation_html(markdown: str) -> tuple[str, list[dict]]:
    """Turn one of struphy's own equation-markdown outputs (``mapping_markdown()``,
    ``formula_markdown()``, ``pde_markdown()``, ...) into the site's placeholder-
    tokenized HTML + LaTeX array, ready for the existing KaTeX ``renderMath()``.

    Those methods already resolve ``.. math::`` blocks into real, correctly
    laid-out LaTeX (aligned systems, matrices, ...), so this only has to split
    on the ``$$...$$`` delimiters they wrap that LaTeX in -- no directive
    parsing or heuristics required. Inline ```:math:`...``` roles are left
    untouched by struphy's own Markdown converter, so those are still pulled
    out here the same way the docstring intro's inline math is.
    """
    math_items: list[dict] = []

    def save_block(match: re.Match) -> str:
        token = f"@@MATH{len(math_items)}@@"
        math_items.append(
            {"token": token, "latex": match.group(1).strip(), "display": True}
        )
        return token

    def save_inline(match: re.Match) -> str:
        token = f"@@MATH{len(math_items)}@@"
        math_items.append({"token": token, "latex": match.group(1), "display": False})
        return token

    stripped = re.sub(r"\$\$([\s\S]*?)\$\$", save_block, markdown or "")
    stripped = re.sub(r":math:`([^`]+)`", save_inline, stripped)
    bold = re.compile(r"\*\*(.+?)\*\*")
    paragraphs = [p.strip() for p in stripped.split("\n\n") if p.strip()]
    html = "".join(f"<p>{bold.sub(r'<strong>\1</strong>', p)}</p>" for p in paragraphs)
    return html, math_items
