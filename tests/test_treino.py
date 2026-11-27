import json
import zipfile

import pytest
import torch

from epi import dados, treino
from epi.ambiente import RAIZ

DIVISOES = RAIZ / "configs" / "divisoes"
CFG = {"modelo": "yolo26n", "treino": {"seed": 0}}


@pytest.mark.parametrize("exp, contagens", [("e1", (5453, 962)), ("e2", (677, 120))])
def test_yaml_do_treino_nao_tem_teste(tmp_path, exp, contagens):
    arquivo, obtido = treino.preparar_yaml(exp, tmp_path, DIVISOES, tmp_path / "t.yaml")

    yaml = dados.carregar_config(arquivo)
    assert set(yaml) == {"path", "train", "val", "names"}
    assert yaml["val"] == f"{exp}_desenvolvimento.txt"
    assert (obtido["train"], obtido["val"]) == contagens
    assert len(yaml["names"]) == (6 if exp == "e1" else 11)


def test_configuracao_fixa_o_que_o_automatico_mudaria():
    cfg = dados.carregar_config(RAIZ / "configs" / "treino.yaml")
    t = cfg["treino"]
    assert t["optimizer"] == "AdamW" and t["batch"] > 0 and t["cache"] == "disk"
    assert t["patience"] < t["epochs"] and "split" not in t and "time" not in t


def test_copiar_substitui_so_o_que_existe(tmp_path):
    origem, destino = tmp_path / "run", tmp_path / "drive"
    (origem / "weights").mkdir(parents=True)
    (origem / "weights" / "last.pt").write_bytes(b"novo")
    (destino / "weights").mkdir(parents=True)
    (destino / "weights" / "best.pt").write_bytes(b"anterior")

    treino.copiar(origem, destino)

    assert (destino / "weights" / "last.pt").read_bytes() == b"novo"
    assert (destino / "weights" / "best.pt").read_bytes() == b"anterior"
    assert not list(destino.rglob("*.parte"))


def test_concluido_e_pulado_sem_treinar(tmp_path):
    pasta = tmp_path / "drive" / "e2_yolo26n_s0"
    pasta.mkdir(parents=True)
    (pasta / treino.MARCADOR).write_text(json.dumps({"ok": 1}), encoding="utf-8")

    resumo = treino.treinar("e2", CFG, "x", "x", tmp_path / "vm", tmp_path / "drive", "x.pt")

    assert resumo == {"ok": 1} and not (tmp_path / "vm").exists()


@pytest.mark.parametrize("epoca, fim", [(-1, True), (99, True), (98, False), (4, False)])
def test_terminou(tmp_path, epoca, fim):
    torch.save({"epoch": epoca, "train_args": {"epochs": 100}}, tmp_path / "last.pt")
    assert treino.terminou(tmp_path / "last.pt") is fim


def test_retomada_em_outra_gpu_e_recusada(tmp_path):
    (tmp_path / "prep").mkdir()
    pasta = tmp_path / "drive" / "e2_yolo26n_s0"
    (pasta / "weights").mkdir(parents=True)
    (pasta / "weights" / "last.pt").write_bytes(b"p")
    (pasta / "ambiente").mkdir()
    (pasta / "ambiente" / "a.json").write_text('{"gpu": "outra"}', encoding="utf-8")

    for _ in range(2):  # recusar não pode contaminar o registro
        with pytest.raises(RuntimeError, match="não troca de máquina"):
            treino.treinar("e2", CFG, tmp_path / "prep", DIVISOES, tmp_path / "vm",
                           tmp_path / "drive", "x.pt")
    assert len(list((pasta / "ambiente").glob("*.json"))) == 1


def test_curva_e_mosaico(tmp_path):
    curva = tmp_path / "results.csv"
    curva.write_text(
        "epoch,time,metrics/mAP50-95(B)\n1,1,0.1\n2,2,0.5\n3,3,0.4\n", encoding="utf-8"
    )
    assert treino.resumir_curva(curva, 100, 10) == {
        "epocas_executadas": 3, "melhor_epoca": 2, "mosaico_desligado_no_fim": False,
    }
    assert treino.resumir_curva(curva, 3, 1)["mosaico_desligado_no_fim"]


def test_entrega_so_inclui_concluidos(tmp_path):
    drive = tmp_path / "f3"
    for nome, pronto in (("e2_yolo26n_s0", True), ("e1_yolo26n_s0", False)):
        (drive / nome / "weights").mkdir(parents=True)
        (drive / nome / "weights" / "best.pt").write_bytes(b"p")
        (drive / nome / "results.csv").write_text("epoch\n1\n", encoding="utf-8")
        if pronto:
            (drive / nome / treino.MARCADOR).write_text("{}", encoding="utf-8")

    with zipfile.ZipFile(treino.entregar(drive, tmp_path / "entrega.zip")) as z:
        assert sorted(x for x in z.namelist() if not x.endswith("/")) == [
            f"e2_yolo26n_s0/{a}" for a in ("best.pt", "results.csv", "resumo.json")]
