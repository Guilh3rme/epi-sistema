import json
import subprocess
import sys
from pathlib import Path

from epi.ambiente import configurar_ultralytics, registrar


def test_registrar_grava_json(tmp_path):
    info = registrar(tmp_path / "ambiente", semente=7)

    arquivos = list((tmp_path / "ambiente").glob("*.json"))
    assert len(arquivos) == 1
    assert json.loads(arquivos[0].read_text(encoding="utf-8")) == info
    assert info["semente"] == 7
    assert info["ultralytics"] and info["torch"]
    assert (info["gpu"] is not None) == info["cuda_disponivel"]


def test_ultralytics_grava_dentro_do_projeto(tmp_path, monkeypatch):
    monkeypatch.setenv("YOLO_CONFIG_DIR", "")  # restaurado ao fim do teste
    configurar_ultralytics(tmp_path)

    codigo = "from ultralytics.utils import DATASETS_DIR; print(DATASETS_DIR)"
    saida = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, check=True)
    assert Path(saida.stdout.strip().splitlines()[-1]) == tmp_path / "data"
