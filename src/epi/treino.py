"""Treino retomável de E1 e E2 (Fase 3). Os testes das bases não entram aqui.

O treino grava em `saida` (disco da máquina) e copia o essencial para `copia` (Drive).
Chamar `configurar_ultralytics()` antes.
"""

import csv
import json
import os
import shutil
from pathlib import Path

from epi import dados
from epi.ambiente import RAIZ, registrar

ARQUIVOS = ("weights/last.pt", "weights/best.pt", "results.csv", "args.yaml")
MARCADOR = "concluido.json"
FITNESS = "metrics/mAP50-95(B)"  # critério do `best.pt` e da parada antecipada


def nome_execucao(experimento: str, cfg: dict) -> str:
    return f"{experimento}_{cfg['modelo']}_s{cfg['treino']['seed']}"


def preparar_yaml(experimento: str, preparado: str | Path, divisoes: str | Path,
                  destino: str | Path) -> tuple[Path, dict[str, int]]:
    """YAML do treino a partir de `configs/base_<exp>.yaml`, sem a chave `test`.

    Confere as impressões digitais, gera as listas de caminhos e devolve as contagens.
    """
    from ultralytics.utils import YAML

    base = dados.carregar_config(RAIZ / "configs" / f"base_{experimento}.yaml")
    esperado = {"train": f"{experimento}_treino.txt", "val": f"{experimento}_desenvolvimento.txt"}
    obtido = {k: base.get(k) for k in esperado}
    if obtido != esperado:
        raise ValueError(f"treino e desenvolvimento inesperados em base_{experimento}: {obtido}")
    preparado = Path(preparado).resolve()
    dados.listas_ultralytics(divisoes, preparado)
    contagens = {k: len((preparado / v).read_text(encoding="utf-8").split())
                 for k, v in esperado.items()}
    destino = Path(destino).resolve()
    destino.parent.mkdir(parents=True, exist_ok=True)
    YAML.save(destino, {"path": preparado.as_posix(), **esperado, "names": base["names"]})
    return destino, contagens


def copiar(origem: str | Path, destino: str | Path) -> None:
    """Copia os arquivos do treino; cada um só substitui o anterior depois de completo."""
    for relativo in ARQUIVOS:
        arquivo = Path(origem) / relativo
        if not arquivo.exists():
            continue
        alvo = Path(destino) / relativo
        alvo.parent.mkdir(parents=True, exist_ok=True)
        parcial = alvo.with_name(alvo.name + ".parte")
        shutil.copyfile(arquivo, parcial)
        os.replace(parcial, alvo)


def terminou(ultimo: str | Path) -> bool:
    """Sem épocas a retomar: encerrado (época -1) ou caído depois da última época."""
    import torch

    # Checkpoint gerado por este projeto; o Ultralytics o desserializa do mesmo modo ao retomar.
    ponto = torch.load(ultimo, map_location="cpu", weights_only=False)
    return ponto["epoch"] == -1 or ponto["epoch"] + 1 >= ponto["train_args"]["epochs"]


def resumir_curva(results: str | Path, epocas: int, close_mosaic: int) -> dict:
    with open(results, encoding="utf-8") as arquivo:
        linhas = [{k.strip(): v.strip() for k, v in x.items()} for x in csv.DictReader(arquivo)]
    melhor = max(linhas, key=lambda x: float(x[FITNESS]))
    executadas = int(float(linhas[-1]["epoch"]))
    return {
        "epocas_executadas": executadas,
        "melhor_epoca": int(float(melhor["epoch"])),
        "mosaico_desligado_no_fim": executadas > epocas - close_mosaic,
    }


def _avaliar_desenvolvimento(melhor: Path, yaml: Path, cfg: dict, saida: Path, device) -> dict:
    """Métricas por classe do `best.pt` no desenvolvimento (`split="val"`)."""
    from ultralytics import YOLO

    t = cfg["treino"]
    m = YOLO(melhor).val(data=str(yaml), split="val", imgsz=t["imgsz"], batch=t["batch"],
                         device=device, plots=False, verbose=False, project=str(saida),
                         name="val_desenvolvimento", exist_ok=True)
    caixa = m.box
    por_classe = {
        m.names[int(c)]: {
            "precisao": float(caixa.p[i]), "revocacao": float(caixa.r[i]),
            "map50": float(caixa.ap50[i]), "map50_95": float(caixa.ap[i]),
        }
        for i, c in enumerate(caixa.ap_class_index)
    }
    geral = {"precisao": float(caixa.mp), "revocacao": float(caixa.mr),
             "map50": float(caixa.map50), "map50_95": float(caixa.map)}
    return {"geral": geral, "por_classe": por_classe}


def _gpu() -> str | None:
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def treinar(experimento: str, cfg: dict, preparado, divisoes, saida, copia, peso,
            device=None) -> dict:
    """Inicia, retoma ou pula um experimento e devolve o resumo (sem caminhos locais)."""
    nome = nome_execucao(experimento, cfg)
    saida, copia = Path(saida).resolve(), Path(copia) / nome
    marcador = copia / MARCADOR
    if marcador.exists():
        print(f"{nome}: já concluído")
        return json.loads(marcador.read_text(encoding="utf-8"))

    yaml, contagens = preparar_yaml(experimento, preparado, divisoes, saida / f"{nome}.yaml")
    pasta = saida / nome
    ultimo = pasta / "weights" / "last.pt"
    if not ultimo.exists() and (copia / "weights" / "last.pt").exists():
        copiar(copia, pasta)  # máquina nova: traz de volta o que foi salvo

    registros = copia / "ambiente"
    anteriores = sorted(registros.glob("*.json"))
    if ultimo.exists() and anteriores:  # conferido antes de registrar a sessão nova
        antes = json.loads(anteriores[-1].read_text(encoding="utf-8"))["gpu"]
        agora = _gpu()
        if agora != antes:
            raise RuntimeError(
                f"um experimento não troca de máquina no meio: começou em {antes}, agora {agora}")
    registrar(registros, semente=cfg["treino"]["seed"])
    sessoes = [json.loads(p.read_text(encoding="utf-8"))
               for p in sorted(registros.glob("*.json"))]

    from ultralytics import YOLO

    if not (ultimo.exists() and terminou(ultimo)):

        def conferir(t):
            lidas = {"train": len(t.train_loader.dataset), "val": len(t.test_loader.dataset)}
            if lidas != contagens or t.args.split != "val":
                raise RuntimeError(f"imagens lidas {lidas} (split {t.args.split}), esperado {contagens}")
            if Path(t.save_dir).resolve() != pasta:
                raise RuntimeError(f"pasta do treino inesperada: {Path(t.save_dir).name}")

        def guardar(t):
            if (t.epoch + 1) % cfg["copiar_a_cada"] == 0:
                try:
                    copiar(t.save_dir, copia)
                    print(f"CÓPIA época {t.epoch + 1} salva")
                except OSError as erro:  # o treino segue; a cópia final é conferida
                    print(f"AVISO: cópia da época {t.epoch + 1} falhou: {erro}")

        def cronometrar(t):
            if t.epoch >= t.epochs:  # a validação final repete o evento
                return
            print(f"TEMPO época {t.epoch + 1}/{t.epochs}: {t.epoch_time / 60:.2f} min")

        if ultimo.exists():
            print(f"{nome}: retomando")
            modelo, argumentos = YOLO(ultimo), {"resume": True}
        else:
            print(f"{nome}: iniciando")
            dados.baixar(cfg["peso"]["url"], peso, cfg["peso"]["sha256"])
            modelo = YOLO(peso)
            argumentos = {**cfg["treino"], "data": str(yaml), "project": str(saida),
                          "name": nome, "exist_ok": True}
        if device is not None:
            argumentos["device"] = device
        modelo.add_callback("on_train_start", conferir)
        modelo.add_callback("on_model_save", guardar)
        modelo.add_callback("on_fit_epoch_end", cronometrar)
        modelo.train(**argumentos)

    melhor = pasta / "weights" / "best.pt"
    metricas = _avaliar_desenvolvimento(melhor, yaml, cfg, saida, device)
    copiar(pasta, copia)
    pesos = {}
    for arquivo in ("best.pt", "last.pt"):
        local, remoto = pasta / "weights" / arquivo, copia / "weights" / arquivo
        pesos[arquivo] = dados.impressao(local)
        if dados.impressao(remoto) != pesos[arquivo]:
            raise RuntimeError(f"{arquivo}: a cópia difere do original")
    t = cfg["treino"]
    resumo = {
        "experimento": experimento, "nome": nome, "modelo": cfg["modelo"],
        "parametros": t, "peso_inicial_sha256": cfg["peso"]["sha256"],
        "pesos_sha256": pesos,
        "relatorio_divisoes_sha256": dados._sha_texto(Path(divisoes) / "relatorio.json"),
        "imagens": contagens,
        **resumir_curva(pasta / "results.csv", t["epochs"], t["close_mosaic"]),
        "metricas_desenvolvimento": metricas,
        "sessoes": sessoes,
    }
    parcial = marcador.with_name(marcador.name + ".parte")
    parcial.write_text(json.dumps(resumo, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(parcial, marcador)  # por último: marca o experimento como concluído
    return resumo


def entregar(copia: str | Path, destino: str | Path) -> Path:
    """Zip com resumo, curva e melhor peso de cada experimento concluído."""
    destino = Path(destino)
    pasta = destino.with_suffix("")
    shutil.rmtree(pasta, ignore_errors=True)
    for marcador in sorted(Path(copia).glob(f"*/{MARCADOR}")):
        alvo = pasta / marcador.parent.name
        alvo.mkdir(parents=True)
        shutil.copyfile(marcador, alvo / "resumo.json")
        shutil.copyfile(marcador.parent / "results.csv", alvo / "results.csv")
        shutil.copyfile(marcador.parent / "weights" / "best.pt", alvo / "best.pt")
    if not pasta.exists():
        raise ValueError("nenhum experimento concluído")
    return Path(shutil.make_archive(str(pasta), "zip", pasta))
