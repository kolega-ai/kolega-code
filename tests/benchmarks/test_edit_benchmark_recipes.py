from benchmarks.edit_tools.models import EditOperationSpec, EditRecipeSpec, FileContent
from benchmarks.edit_tools.recipes import (
    apply_recipe,
    derive_recipe,
    payload_line_count,
    render_recipe_prompt,
    text_sha256,
)


def test_recipe_applies_original_coordinates_and_round_trips() -> None:
    before = {"src/example.py": FileContent(text="def first():\n    return 1\n\n\ndef second():\n    return 2\n")}
    recipe = EditRecipeSpec(
        operations=[
            EditOperationSpec(
                id="replace-first",
                kind="replace",
                path="src/example.py",
                start_line=1,
                end_line=2,
                before_sha256=text_sha256("def first():\n    return 1\n"),
                new_text="def first():\n    return 10\n",
            ),
            EditOperationSpec(
                id="insert-helper",
                kind="insert",
                path="src/example.py",
                start_line=4,
                before_sha256=text_sha256("\n"),
                new_text="def helper():\n    return 3\n\n",
            ),
        ]
    )

    after = apply_recipe(before, recipe)
    derived = derive_recipe(before, after)

    assert after["src/example.py"].text == (
        "def first():\n    return 10\n\n\ndef helper():\n    return 3\n\ndef second():\n    return 2\n"
    )
    assert apply_recipe(before, derived) == after


def test_neutral_prompt_contains_exact_locations_and_payloads() -> None:
    before = {"config.txt": FileContent(text="first\nsecond\n")}
    recipe = EditRecipeSpec(
        operations=[
            EditOperationSpec(
                id="replace-second",
                kind="replace",
                path="config.txt",
                start_line=2,
                end_line=2,
                before_sha256=text_sha256("second\n"),
                new_text="replacement\n",
            )
        ]
    )

    prompt = render_recipe_prompt(recipe, before)

    assert "original lines 2–2" in prompt
    assert 'begins with "second"' in prompt
    assert "replacement\n" in prompt
    assert "apply_patch" not in prompt
    assert "SEARCH" not in prompt


def test_payload_size_counts_removed_source_lines() -> None:
    before = {"old.txt": FileContent(text="one\ntwo\nthree\nfour\n")}
    recipe = EditRecipeSpec(
        operations=[
            EditOperationSpec(
                id="remove-three-lines",
                kind="delete",
                path="old.txt",
                start_line=1,
                end_line=3,
                before_sha256=text_sha256("one\ntwo\nthree\n"),
            )
        ]
    )

    assert payload_line_count(recipe, before) == 3
