import ast
import json
import re
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "colab_notebooks").rglob("*.ipynb"))


def python_of(source):
    # IPython magics and shell escapes are valid in Colab; replace them so plain Python can parse the rest.
    source = re.sub(r"=\s*!", "= None  #", source)
    return "\n".join(re.sub(r"^(\s*)[!%].*", r"\1pass", line) for line in source.splitlines())


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_and_every_code_cell_parses(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            try:
                ast.parse(python_of("".join(cell["source"])))
            except SyntaxError as error:
                pytest.fail(f"{path.name} cell {i}: {error}")


def test_candidate_notebook_is_generated_from_its_script():
    import subprocess, sys
    root = Path(__file__).resolve().parents[1]
    before = (root / "colab_notebooks/CP3_candidate_models.ipynb").read_text(encoding="utf-8")
    subprocess.run([sys.executable, str(root / "training/make_candidates_notebook.py")], check=True, capture_output=True)
    assert (root / "colab_notebooks/CP3_candidate_models.ipynb").read_text(encoding="utf-8") == before
