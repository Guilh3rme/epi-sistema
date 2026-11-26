import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from epi import clp, fluxo
from epi.associacao import Deteccao
from epi.decisao import BLOQUEADO, INDETERMINADO, LIBERADO
from epi.deteccao import detector_roteirizado, estimar_pose

LARGURA, ALTURA = 200, 200
PESSOA = Deteccao("pessoa", 0.9, (60, 20, 140, 180))


def config(tmp_path, porta=1):
    # Montada aqui: ler o YAML importaria o ultralytics no processo dos testes.
    return {
        "eventos": str(tmp_path / "eventos"), "clp": {"porta": porta},
        "exigidos": ["capacete"], "associacao_ativa": "A", "modelos": {},
        "parametros": {
            "conf_min": 0.25, "conf_ok": 0.5, "conf_ponto": 0.5, "margem_borda": 0.01,
            "altura_min": 0.5, "n": 3, "maioria": 0.6, "k_entrada": 2, "k_saida": 2,
            "max_quadros": 30, "iou_duplicata": 0.5,
        },
    }


def video(tmp_path, quadros):
    caminho = tmp_path / "v.mp4"
    escritor = cv2.VideoWriter(str(caminho), cv2.VideoWriter_fourcc(*"mp4v"), 30, (LARGURA, ALTURA))
    for _ in range(quadros):
        escritor.write(np.zeros((ALTURA, LARGURA, 3), np.uint8))
    escritor.release()
    return caminho


def detector_de(trechos):
    """`trechos`: sequência de (quadros, confiança do capacete | None sem item | False vazio)."""
    plano = [conf for quadros, conf in trechos for _ in range(quadros)]

    def detector(indice, quadro):
        conf = plano[indice] if indice < len(plano) else False
        if conf is False:
            return []
        item = [Deteccao("capacete", conf, (90, 25, 110, 45))] if conf else []
        return [PESSOA, *item]

    return detector


def sem_pose(quadro, pessoa):
    return None


class CLPFalso:
    """Substitui as funções de `clp`; `falhar(valores)` simula escritas perdidas."""

    def __init__(self, monkeypatch, falhar=lambda valores: False):
        self.escritas, self.registradores, self.falhar = [], [1, 0, 0], falhar
        monkeypatch.setattr(clp, "enviar", self.enviar)
        monkeypatch.setattr(clp, "ler", lambda cliente: self.registradores)

    def enviar(self, cliente, decisao, seq, ausentes=()):
        valores = [clp.CODIGOS[decisao], seq, 0]
        ok = not self.falhar(valores)
        if ok:
            self.registradores = valores
            self.escritas.append(valores[0])
        return {"ok": ok, "valores": valores, "latencia_ms": 1.0 if ok else None,
                "erro": None if ok else "falha simulada"}


def linhas(resumo):
    texto = Path(resumo["eventos"]).read_text(encoding="utf-8")
    return [json.loads(linha) for linha in texto.splitlines()]


def arquivos(pasta):
    return sorted(p.suffix for p in pasta.rglob("*") if p.is_file())


def test_tres_decisoes_zeragem_registro_e_nenhuma_imagem(tmp_path, monkeypatch):
    falso = CLPFalso(monkeypatch)
    trechos = [(6, 0.9), (3, False), (6, None), (3, False), (6, 0.3), (3, False)]
    origem = video(tmp_path, 27)

    r = fluxo.executar(origem, config(tmp_path), detector_de(trechos), sem_pose)

    assert r["falha"] is None and r["quadros"] == 27
    ativas = [d for d in r["decisoes"] if d["associacao"] == "A"]
    assert [d["decisao"] for d in ativas] == [LIBERADO, BLOQUEADO, INDETERMINADO]
    assert [d["sequencia"] for d in ativas] == [1, 2, 3]
    # partida, início 1, LIBERADO, zerado no fim da passagem, início 2, BLOQUEADO (retido),
    # início 3, INDETERMINADO (retido), fim do programa.
    assert falso.escritas == [0, 0, 1, 0, 0, 2, 0, 3, 0]
    assert r["registradores_finais"] == [0, 3, 0]
    # B, sem pose, nunca libera nem vai ao CLP.
    b = [d for d in r["decisoes"] if d["associacao"] == "B"]
    assert {d["decisao"] for d in b} == {INDETERMINADO} and all(d["escrita"] is None for d in b)

    primeira = next(x for x in linhas(r) if x["tipo"] == "decisao")
    for campo in ("horario", "passagem", "associacao", "decisao", "motivo", "estados",
                  "tempos", "escrita", "modelos", "config", "commit"):
        assert campo in primeira
    assert primeira["tempos"]["ate_decisao_s"] > 3 / 30
    assert r["fps_video"] == 30 and r["fps_laco"] > 0
    # Só o vídeo de entrada, os eventos e o registro do ambiente.
    assert arquivos(tmp_path) == [".json", ".jsonl", ".mp4"]


@pytest.mark.parametrize("segunda_em", [0, 6])  # desde o começo e depois de liberar
def test_segunda_pessoa_nunca_fica_liberada(tmp_path, monkeypatch, segunda_em):
    falso = CLPFalso(monkeypatch)
    bom = detector_de([(12, 0.9)])
    outra = Deteccao("pessoa", 0.9, (0, 20, 50, 180))
    chamadas = []

    def detector(indice, quadro):
        chamadas.append(indice)
        extra = [outra] if len(chamadas) > 1 and indice >= segunda_em else []
        return bom(indice, quadro) + extra

    r = fluxo.executar(video(tmp_path, 12), config(tmp_path), detector, sem_pose)

    motivos = [x.get("motivo") for x in linhas(r)]
    if segunda_em == 0:
        assert 1 not in falso.escritas
    else:
        assert 1 in falso.escritas and "segunda_pessoa" in motivos
        depois = falso.escritas[falso.escritas.index(1) + 1:]
        assert depois and set(depois) == {0}


def test_quadros_vazios_nao_escondem_fluxo_lento(tmp_path, monkeypatch):
    CLPFalso(monkeypatch)
    bom = detector_de([(6, 0.9), (60, False)])

    def lento(indice, quadro):
        dets = bom(indice, quadro)
        if dets:
            time.sleep(0.05)  # 20 q/s com pessoa, num vídeo de 30
        return dets

    r = fluxo.executar(video(tmp_path, 66), config(tmp_path), lento, sem_pose)
    assert r["fps_laco"] > r["fps_video"] > r["fps_com_pessoa"]
    assert not r["fluxo_acompanha_video"]
    assert not any(d["tempos"]["fluxo_acompanha_video"] for d in r["decisoes"])


def test_exibicao_nao_grava_e_q_encerra_sem_liberar(tmp_path, monkeypatch):
    falso = CLPFalso(monkeypatch)
    telas = []
    monkeypatch.setattr(cv2, "imshow", lambda nome, tela: telas.append(tela))
    monkeypatch.setattr(cv2, "waitKey", lambda ms: ord("q") if len(telas) == 2 else -1)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    c = {**config(tmp_path), "mostrar": True}

    r = fluxo.executar(video(tmp_path, 10), c, detector_de([(10, 0.9)]), sem_pose)

    assert len(telas) == 2 and telas[0].any()
    assert r["falha"] == "interrompido" and falso.escritas[-2:] == [3, 0]
    assert 1 not in falso.escritas
    assert arquivos(tmp_path) == [".json", ".jsonl", ".mp4"]


def test_video_que_nao_abre_e_indeterminado(tmp_path, monkeypatch):
    falso = CLPFalso(monkeypatch)
    r = fluxo.executar(tmp_path / "nao_existe.mp4", config(tmp_path), detector_de([]), sem_pose)
    assert r["falha"].startswith("falha_camera")
    assert {d["decisao"] for d in r["decisoes"]} == {INDETERMINADO}
    assert falso.escritas == [0, 3, 0]


def test_leitura_interrompida_e_indeterminado(tmp_path, monkeypatch):
    falso = CLPFalso(monkeypatch)

    class Interrompida:
        def __init__(self, caminho):
            self.restam = 2

        def isOpened(self):
            return True

        def get(self, prop):
            return {cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: 100,
                    cv2.CAP_PROP_FRAME_HEIGHT: ALTURA, cv2.CAP_PROP_FRAME_WIDTH: LARGURA}[prop]

        def read(self):
            self.restam -= 1
            return (self.restam >= 0), np.zeros((ALTURA, LARGURA, 3), np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", Interrompida)
    r = fluxo.executar("x.mp4", config(tmp_path), detector_de([(2, 0.9)]), sem_pose)
    assert r["falha"].startswith("falha_camera")
    assert falso.escritas == [0, 0, 3, 0]


@pytest.mark.parametrize("quebra_em", [4, 6])  # antes e depois de LIBERADO
def test_falha_do_modelo_nunca_deixa_liberado(tmp_path, monkeypatch, quebra_em):
    falso = CLPFalso(monkeypatch)
    bom = detector_de([(20, 0.9)])
    chamadas = []

    def detector(indice, quadro):
        chamadas.append(indice)
        if len(chamadas) > quebra_em:  # a primeira chamada é o aquecimento
            raise RuntimeError("modelo caiu")
        return bom(indice, quadro)

    r = fluxo.executar(video(tmp_path, 20), config(tmp_path), detector, sem_pose)

    assert r["falha"].startswith("falha_processamento")
    assert falso.registradores[0] == 0
    if quebra_em == 4:  # três quadros processados: passagem aberta, janela completa
        assert 1 in falso.escritas
    assert falso.escritas[-1] == 0


def test_falha_ao_carregar_modelo(tmp_path, monkeypatch):
    falso = CLPFalso(monkeypatch)

    def quebrar(config):
        raise FileNotFoundError("sem pesos")

    monkeypatch.setattr(fluxo, "_carregar_modelos", quebrar)
    r = fluxo.executar(video(tmp_path, 5), config(tmp_path))
    assert r["falha"].startswith("falha_modelo")
    assert falso.escritas == [0, 3, 0]


def test_zeragem_que_falha_e_repetida(tmp_path, monkeypatch):
    perdidas = []

    def falhar(valores):
        # Perde a primeira zeragem posterior ao LIBERADO (fim da passagem).
        if valores[:2] == [0, 1] and not perdidas:
            perdidas.append(valores)
            return True
        return False

    falso = CLPFalso(monkeypatch, falhar)
    relogio = iter(range(0, 1000, 2))
    monkeypatch.setattr(fluxo.time, "monotonic", lambda: next(relogio))
    r = fluxo.executar(video(tmp_path, 12), config(tmp_path),
                       detector_de([(8, 0.9), (4, False)]), sem_pose)
    assert perdidas and falso.registradores[0] == 0
    assert "nova_tentativa" in [x.get("motivo") for x in linhas(r)]


def test_falha_no_registro_do_ambiente_ainda_zera_o_clp(tmp_path, monkeypatch):
    falso = CLPFalso(monkeypatch)

    def quebrar(*args, **kwargs):
        raise OSError("sem disco")

    monkeypatch.setattr(fluxo, "registrar", quebrar)
    r = fluxo.executar(video(tmp_path, 5), config(tmp_path), detector_de([]), sem_pose)
    assert r["falha"].startswith("falha_processamento")
    assert falso.escritas == [3, 0]


def test_liberado_com_escrita_perdida_e_zerado(tmp_path, monkeypatch):
    falso = CLPFalso(monkeypatch, falhar=lambda valores: valores[0] == 1)
    r = fluxo.executar(video(tmp_path, 12), config(tmp_path),
                       detector_de([(8, 0.9), (4, False)]), sem_pose)
    liberado = next(d for d in r["decisoes"] if d["associacao"] == "A")
    assert not liberado["escrita"]["ok"] and r["falhas_clp"] == 1
    assert 1 not in falso.escritas and falso.registradores[0] == 0
    motivos = [x.get("motivo") for x in linhas(r)]
    assert motivos[motivos.index("falha_escrita") + 1] == "todos_presentes"


def test_clp_inacessivel_registra_falha_e_processa_tudo(tmp_path):
    r = fluxo.executar(video(tmp_path, 8), config(tmp_path, porta=1),
                       detector_de([(8, 0.9)]), sem_pose)
    assert r["falha"] is None and r["quadros"] == 8
    assert r["falhas_clp"] >= 3 and r["registradores_finais"] is None
    registros = linhas(r)
    assert any(x["tipo"] == "decisao" and x["ativa"] and not x["escrita"]["ok"] for x in registros)
    assert any(x.get("motivo") == "fim_programa" and not x["escrita"]["ok"] for x in registros)


@pytest.mark.parametrize("mudanca", [
    {"exigidos": []}, {"exigidos": ["capacet"]}, {"associacao_ativa": "C"},
])
def test_configuracao_invalida(tmp_path, mudanca):
    with pytest.raises(ValueError):
        fluxo.executar("v.mp4", {**config(tmp_path), **mudanca})


def test_roteiro_posiciona_itens_na_caixa_da_pessoa():
    itens = detector_roteirizado({
        "itens": {"c": {"classe": "capacete", "conf": 0.8, "caixa": [0, 0, 0.5, 0.1]}},
        "faixas": [{"quadros": 2, "pessoa": False}, {"quadros": 2, "itens": ["c"]}],
    })
    assert itens(1, [PESSOA]) == [] and itens(2, []) == [] and itens(4, [PESSOA]) == []
    assert itens(3, [PESSOA]) == [Deteccao("capacete", 0.8, (60, 20, 100, 36))]


def test_pose_sem_correspondencia_com_a_pessoa():
    class Modelo:
        def predict(self, quadro, **opcoes):
            assert opcoes["save"] is False
            caixas = np.array([[0, 0, 10, 10], [62, 22, 138, 178]])
            pontos = np.stack([np.zeros((17, 3)), np.ones((17, 3))])
            return [SimpleNamespace(
                boxes=SimpleNamespace(xyxy=caixas),
                keypoints=SimpleNamespace(conf=pontos[..., 2], data=pontos),
            )]

    assert estimar_pose(Modelo(), None, PESSOA) == np.ones((17, 3)).tolist()
    assert estimar_pose(Modelo(), None, Deteccao("pessoa", 0.9, (150, 150, 190, 190))) is None
    assert estimar_pose(Modelo(), None, None) is None


def test_arquivos_de_configuracao_sao_validos():
    """Em outro processo, para não importar o ultralytics neste."""
    codigo = """
from epi import fluxo
from epi.associacao import regioes
from epi.deteccao import detector_roteirizado
c = fluxo.carregar_yaml("configs/ponto_de_controle.yaml")
fluxo._validar(c)
regioes([(1, 1, 1)] * 17, (10, 10), c["parametros"])
for arquivo in (fluxo.RAIZ / "configs").glob("roteiro_*.yaml"):
    detector_roteirizado(fluxo.carregar_yaml(arquivo))
"""
    subprocess.run([sys.executable, "-c", codigo], check=True)
