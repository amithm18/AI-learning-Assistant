import markdown

from app import fix_markdown_lists


def render(md_text):
    return markdown.markdown(fix_markdown_lists(md_text), extensions=["extra", "fenced_code"])


def test_list_right_after_a_label_becomes_a_list():
    html = render("**Key Points:**\n- First\n- Second")
    assert "<li>First</li>" in html and "<li>Second</li>" in html


def test_two_space_sub_points_are_nested():
    html = render("Intro\n- Parent\n  - Child\n- Next")
    assert "<ul>\n<li>Child</li>\n</ul>" in html


def test_code_blocks_are_untouched():
    md_text = "```python\nx = 1\n- y\n  - z\n```"
    assert fix_markdown_lists(md_text) == md_text
