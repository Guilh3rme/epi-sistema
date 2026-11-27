import csv
import json

import numpy as np
import pytest
from PIL import Image

from epi import avaliacao, dados
from epi.ambiente import RAIZ

DIVISOES = RAIZ / "configs" / "divisoes"
CFG = {"principais": ["pessoa", "capacete", "colete", "luvas"],
       "exploratorias": ["calcado", "oculos"], "tolerancia_f3": 0.005,
       "limiares": {"e1": 0.5, "e2": None}}
NOMES = ["pessoa", "capacete", "oculos", "luvas", "calcado", "colete"]


def _classe(i, f1=None):
    c = {"id": i, "imagens": 1, "caixas": 0 if f1 is None else 1}
    if f1 is not None:
        c |= {"ap50": f1, "ap": f1 / 2, "p": [f1] * 11, "r": [f1] * 11, "f1": [f1] * 11}
    return c


def test_limiar_usa_so_as_principais():
    pico = np.exp(-((np.linspace(0, 1, 1000) - 0.3) ** 2) / 0.01).tolist()
    outro = np.linspace(0, 1, 1000).tolist()  # puxaria o limiar para 1
    por_classe = {c: {"f1": pico} for c in CFG["principais"]} | {"oculos": {"f1": outro}}

    limiar, f1 = avaliacao.escolher_limiar(por_classe, CFG["principais"])

    assert limiar == pytest.approx(0.3, abs=0.01) and f1 == pytest.approx(1, abs=0.01)
    with pytest.raises(ValueError, match="sem caixas"):
        avaliacao.escolher_limiar({"pessoa": {"f1": pico}}, CFG["principais"])


def test_faixas_pela_area_com_lado_maior_em_640():
    # Imagem de 1280: a escala é 0,5. Caixas de 40, 120 e 400 px viram 20, 60 e 200 em 640.
    def caixa(x, lado):
        return [0, (x + lado / 2) / 1280, (10 + lado / 2) / 1280, lado / 1280, lado / 1280]

    gt = [{"nome": "a.jpg", "largura": 1280, "altura": 1280,
           "caixas": [caixa(0, 40), caixa(100, 120), caixa(600, 400)]}]
    def pred(x, lado, conf):
        return {"file_name": "a.jpg", "category_id": 1, "bbox": [x, 10, lado, lado], "score": conf}

    r = avaliacao.por_tamanho(gt, [pred(0, 40, 0.9), pred(600, 400, 0.3)],
                              {0: "pessoa", 1: "capacete"}, limiar=0.5)

    assert [r["pessoa", f]["caixas"] for f in avaliacao.FAIXAS] == [1, 1, 1]
    assert r["pessoa", "pequeno"]["revocacao"] == 1 and r["pessoa", "pequeno"]["ap50"] == 1
    assert r["pessoa", "medio"] == {"caixas": 1, "revocacao": 0, "ap50": 0, "ap50_95": 0}
    # Detectada com confiança abaixo do limiar: conta no AP, não na revocação.
    assert r["pessoa", "grande"]["revocacao"] == 0 and r["pessoa", "grande"]["ap50"] == 1
    assert r["capacete", "grande"] == {"caixas": 0, "revocacao": None, "ap50": None, "ap50_95": None}

    vazio = avaliacao.por_tamanho(gt, [], {0: "pessoa"}, limiar=0.5)
    assert vazio["pessoa", "medio"]["ap50"] == 0 and vazio["pessoa", "medio"]["revocacao"] == 0


def test_media_das_principais_e_blocos():
    por_classe = {n: _classe(i, 0.1 * (i + 1)) for i, n in enumerate(NOMES)}
    por_classe["no_helmet"] = _classe(6, 0.9)

    linhas = avaliacao.linhas_por_classe("e2_interno", CFG, por_classe, 0.5)

    media = next(x for x in linhas if x["classe"] == avaliacao.MEDIA)
    assert media["f1"] == pytest.approx(np.mean([0.1, 0.2, 0.6, 0.4]))
    assert [x["bloco"] for x in linhas].count("exploratória") == 2
    assert linhas[-1]["classe"] == "no_helmet" and linhas[-1]["bloco"] == "ausência"
    por_classe["colete"] = _classe(5)  # sem caixas: a média não é calculada
    linhas = avaliacao.linhas_por_classe("e2_cruzado", CFG, por_classe, 0.5)
    assert next(x for x in linhas if x["classe"] == avaliacao.MEDIA)["f1"] is None
    assert not any(x["bloco"] == "ausência" for x in linhas)


@pytest.mark.parametrize("execucao", list(avaliacao.EXECUCOES))
def test_yaml_do_desenvolvimento_nao_tem_teste(tmp_path, execucao):
    for split, chaves in (("val", {"train", "val"}), ("test", {"train", "val", "test"})):
        arquivo, lista = avaliacao.preparar_yaml(execucao, split, tmp_path, DIVISOES,
                                                 tmp_path / "a.yaml")
        assert set(dados.carregar_config(arquivo)) == chaves | {"path", "names"}
        assert lista


def test_marcacoes_descarta_repetidas(tmp_path):
    (tmp_path / "sh17" / "images").mkdir(parents=True)
    (tmp_path / "sh17" / "labels").mkdir()
    Image.new("RGB", (64, 32)).save(tmp_path / "sh17" / "images" / "a.jpg")
    (tmp_path / "sh17" / "labels" / "a.txt").write_text("1 0.5 0.5 0.1 0.1\n" * 2, encoding="utf-8")

    (m,) = avaliacao.marcacoes(tmp_path, ["./sh17/images/a.jpg"])

    assert (m["largura"], m["altura"], len(m["caixas"])) == (64, 32, 1)


def _teste(tmp_path, cfg, dev=None):
    peso = tmp_path / "p.pt"
    peso.write_bytes(b"peso")
    if dev is not None:
        sha = dados.impressao(peso)
        (tmp_path / "f4").mkdir()
        (tmp_path / "f4" / "desenvolvimento.json").write_text(
            json.dumps({"e1": {"peso_sha256": sha, "reproduzido": True, "limiar": 0.5} | dev}),
            encoding="utf-8")
    return avaliacao.avaliar("e1_interno", "test", cfg, {"treino": {"seed": 0}}, tmp_path,
                             DIVISOES, tmp_path / "vm", tmp_path / "f4", peso)


@pytest.mark.parametrize("limiar, dev, mensagem", [
    (None, {}, "grave o limiar"),
    (0.5, None, "não reproduzido"),
    (0.5, {"reproduzido": False}, "não reproduzido"),
    (0.5, {"peso_sha256": "outro"}, "não reproduzido"),
    (0.5, {"limiar": 0.4}, "difere do escolhido"),
])
def test_teste_recusado_antes_de_ler_as_imagens(tmp_path, limiar, dev, mensagem):
    with pytest.raises(RuntimeError, match=mensagem):
        _teste(tmp_path, CFG | {"limiares": {"e1": limiar}}, dev)
    assert not (tmp_path / "vm").exists()


def test_teste_recusa_codigo_modificado(tmp_path, monkeypatch):
    monkeypatch.setattr(avaliacao, "registrar", lambda *a, **k: {"codigo_modificado": True})
    with pytest.raises(RuntimeError, match="sem modificações"):
        _teste(tmp_path, CFG, {})


def test_concluida_nao_roda_de_novo_e_gera_tabelas(tmp_path):
    pasta = tmp_path / "f4" / "e1_interno"
    pasta.mkdir(parents=True)
    arquivos = {
        "curvas.json": {n: _classe(i, 0.5) for i, n in enumerate(NOMES)},
        "marcacoes.json": [{"nome": "a.jpg", "largura": 640, "altura": 640,
                            "caixas": [[0, 0.5, 0.5, 0.5, 0.5]]}],
        "predicoes.json": [],
    }
    for nome, conteudo in arquivos.items():
        (pasta / nome).write_text(json.dumps(conteudo), encoding="utf-8")
    marcador = {"execucao": "e1_interno", "experimento": "e1", "split": "test", "limiar": 0.5,
                "sha256": {n: dados.impressao(pasta / n) for n in arquivos}}
    (pasta / avaliacao.MARCADOR).write_text(json.dumps(marcador), encoding="utf-8")

    feito = avaliacao.avaliar("e1_interno", "test", CFG, {}, "x", "x", tmp_path / "vm",
                              tmp_path / "f4", "x.pt")
    assert feito == marcador and not (tmp_path / "vm").exists()
    with pytest.raises(RuntimeError, match="outra divisão"):
        avaliacao.avaliar("e1_interno", "val", CFG, {}, "x", "x", "vm", tmp_path / "f4", "x.pt")

    assert avaliacao.tabelas(tmp_path / "f4", tmp_path / "t", CFG) == ["e1_interno"]
    with open(tmp_path / "t" / "por_tamanho.csv", encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))
    assert {x["faixa"] for x in linhas} == set(avaliacao.FAIXAS)
    assert "| 0,500 |" in (tmp_path / "t" / "por_classe.md").read_text(encoding="utf-8")
    assert "não é resultado" not in (tmp_path / "t" / "por_classe.md").read_text(encoding="utf-8")

    (pasta / "predicoes.json").write_text("[1]", encoding="utf-8")
    with pytest.raises(ValueError, match="difere do registrado"):
        avaliacao.tabelas(tmp_path / "f4", tmp_path / "t", CFG)


def test_reproducao_separa_ap_de_precisao_e_revocacao():
    def metricas(p, ap):
        m = {"precisao": p, "revocacao": 0.5, "map50": ap, "map50_95": 0.4}
        return {"geral": m, "por_classe": {"pessoa": m}}

    ap, pr = avaliacao.diferencas(metricas(0.80, 0.60), metricas(0.82, 0.601))
    assert ap == pytest.approx(0.001) and pr == pytest.approx(0.02)
    assert avaliacao.diferencas(metricas(0.8, 0.6), metricas(0.8, 0.61))[0] == pytest.approx(0.01)
    with pytest.raises(RuntimeError, match="classes diferentes"):
        avaliacao.diferencas(metricas(0.8, 0.6), metricas(0.8, 0.6) | {"por_classe": {}})
