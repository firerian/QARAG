import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


@pytest.fixture
def create_markdown_file():
    def _create(content: str, filename: str = "test.md") -> str:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".md",
            prefix="test_",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(content)
            return f.name
    return _create


@pytest.fixture
def create_html_file():
    def _create(content: str, filename: str = "test.html") -> str:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".html",
            prefix="test_",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(content)
            return f.name
    return _create


MARKDOWN_CONTENT = """# Heading 1

This is the first paragraph under heading 1.
It has multiple lines of text that should be grouped together.

## Heading 2

This is a paragraph under heading 2.

- List item 1
- List item 2
- List item 3

### Heading 3

**Bold text** and *italic text* and `code text`.

```
This is a code block
with multiple lines
```

> This is a blockquote
> with multiple lines

[Link text](https://example.com)

![Image alt](image.png)

---

## Heading 2 Again

Final paragraph with some **bold** and ~~strikethrough~~ text.
"""


HTML_CONTENT = """<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
    <meta charset="utf-8">
</head>
<body>
    <nav>
        <ul>
            <li><a href="/">Home</a></li>
            <li><a href="/about">About</a></li>
        </ul>
    </nav>
    <main>
        <article>
            <h1>Main Article Title</h1>
            <p>This is the first paragraph of the article. It contains important information.</p>
            <p>This is the second paragraph with more details.</p>
            <h2>Section One</h2>
            <p>Content under section one.</p>
            <ul>
                <li>Item A</li>
                <li>Item B</li>
                <li>Item C</li>
            </ul>
            <h2>Section Two</h2>
            <p>Content under section two with <strong>bold text</strong> and <em>italic text</em>.</p>
            <blockquote>This is a blockquote.</blockquote>
            <pre><code>print("Hello World")</code></pre>
            <h3>Subsection</h3>
            <p>Final paragraph.</p>
        </article>
    </main>
    <footer>
        <p>Copyright 2024</p>
    </footer>
    <aside class="sidebar">
        <p>Sidebar content to be ignored.</p>
    </aside>
    <div class="ad-banner">Advertisement content</div>
</body>
</html>
"""


HTML_MINIMAL_CONTENT = """<!DOCTYPE html>
<html>
<body>
    <h1>Simple Page</h1>
    <p>Simple paragraph one.</p>
    <p>Simple paragraph two.</p>
</body>
</html>
"""
