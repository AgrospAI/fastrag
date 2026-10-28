"""Execute standalone published structural recipes against the public SDK."""

import re
from pathlib import Path


def test_published_structure_recipes():
    guide = Path(__file__).parents[1] / "docs/source/guide/structure.md"
    source = guide.read_text()
    recipes = re.findall(
        r"<!-- executable:(\w+) -->\n```python\n(.*?)\n```", source, re.S
    )
    assert {name for name, _ in recipes} == {"budget", "parents", "adapter"}
    for name, code in recipes:
        namespace = {"__name__": f"documentation_{name}"}
        exec(compile(code, str(guide), "exec"), namespace)
