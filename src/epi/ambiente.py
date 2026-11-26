"""Registro do ambiente de cada execução, para reprodutibilidade."""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parents[2]


def em_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ


def configurar_ultralytics(raiz: Path = RAIZ) -> None:
    """Mantém configuração, bases e resultados do Ultralytics dentro do projeto.

    Chamar antes de importar `ultralytics`, que lê esses caminhos na importação.
    """
    config = raiz / "runs" / ".ultralytics"
    if "ultralytics" in sys.modules and os.environ.get("YOLO_CONFIG_DIR") != str(config):
        raise RuntimeError("Chame configurar_ultralytics() antes de importar ultralytics.")
    config.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config)
    # Em outro processo, para que a próxima importação neste já leia os caminhos novos.
    codigo = (
        "import sys; from ultralytics import settings; settings.update("
        "{'datasets_dir': sys.argv[1], 'runs_dir': sys.argv[2], 'weights_dir': sys.argv[3]})"
    )
    caminhos = [raiz / "data", raiz / "runs", raiz / "runs" / "pesos"]
    subprocess.run([sys.executable, "-c", codigo, *map(str, caminhos)], check=True)


def _git(*args: str) -> str | None:
    try:
        saida = subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True)
    except OSError:
        return None
    return saida.stdout.strip() if saida.returncode == 0 else None


def registrar(destino: str | Path, semente: int) -> dict:
    """Grava em `destino` um JSON com versões, hardware, semente e commit do código."""
    agora = datetime.now(timezone.utc)
    gpu = torch.cuda.is_available()
    estado = _git("status", "--porcelain")
    info = {
        "data_utc": agora.isoformat(timespec="seconds"),
        "colab": em_colab(),
        "sistema": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": version("ultralytics"),
        "cuda": torch.version.cuda,
        "cuda_disponivel": gpu,
        "gpu": torch.cuda.get_device_name(0) if gpu else None,
        "semente": semente,
        "commit": _git("rev-parse", "HEAD"),
        # True: o código executado difere do commit registrado.
        "codigo_modificado": None if estado is None else bool(estado),
    }
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    arquivo = destino / f"{agora:%Y%m%dT%H%M%S%fZ}.json"
    arquivo.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info
