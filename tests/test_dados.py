import hashlib
import io
import random

import cv2
import numpy as np
import pytest
import requests
from PIL import Image

from epi import dados
from epi.ambiente import RAIZ

FAMILIAS = ["ultralytics-ppe-nov2026-", "source-"]


@pytest.fixture
def config(monkeypatch):
    pasta = RAIZ / "runs" / ".ultralytics"
    pasta.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(pasta))
    return dados.carregar_config(RAIZ / "configs" / "dados.yaml")


def imagem(largura=64, altura=48, formato="JPEG", semente=0, orientacao=None):
    ruido = np.random.default_rng(semente).integers(0, 256, (6, 8, 3), np.uint8)
    figura = Image.fromarray(ruido).resize((largura, altura), Image.Resampling.BICUBIC)
    saida, extras = io.BytesIO(), {}
    if orientacao:
        exif = Image.Exif()
        exif[0x0112] = orientacao
        extras["exif"] = exif
    figura.save(saida, formato, **extras)
    return saida.getvalue()


def test_mapas_do_projeto(config):
    # ids nativos de sh17.yaml e construction-ppe.yaml -> pessoa, capacete, óculos, luvas, calçado, colete
    assert config["sh17"]["mapa"] == {0: 0, 10: 1, 8: 2, 9: 3, 14: 4, 16: 5}
    assert config["cppe"]["mapa"] == {5: 0, 0: 1, 4: 2, 1: 3, 3: 4, 2: 5, 6: 6, 7: 7, 8: 8, 9: 9, 11: 10}
    linhas, nativas = dados.converter_rotulos(
        "10 .5 .5 .1 .1\r\n6 .2 .2 .1 .1\r\n\r\n0 .1 .2 .3 .4\r\n", config["sh17"]["mapa"]
    )
    assert linhas == ["1 .5 .5 .1 .1", "0 .1 .2 .3 .4"]  # foot (6) descartada
    assert nativas == {10: 1, 6: 1, 0: 1}
    with pytest.raises(ValueError):
        dados.converter_rotulos("0 .1 .1 .2 .2 .3 .3", {0: 0})


def test_yamls_dos_experimentos(config):
    for nome, treino, teste, classes in (
        ("e1", "e1", "e1_teste", 6), ("e1_cppe", "e1", "e1_teste_cruzado", 6),
        ("e2", "e2", "e2_teste", 11), ("e2_sh17", "e2", "e2_teste_cruzado", 11),
    ):
        y = dados.carregar_config(RAIZ / "configs" / f"base_{nome}.yaml")
        assert (y["train"], y["val"], y["test"]) == (
            f"{treino}_treino.txt", f"{treino}_desenvolvimento.txt", f"{teste}.txt")
        assert len(y["names"]) == classes
        assert [y["names"][i] for i in range(6)] == [
            "pessoa", "capacete", "oculos", "luvas", "calcado", "colete"]


def test_lista_com_crlf_e_nomes_inseguros():
    assert dados.ler_lista(b"a.jpeg\r\nb c.png\r\n\r\n") == ["a.jpeg", "b c.png"]
    for ruim in (b"../x.jpg\r\n", b"p/x.jpg\n", b"C:x.jpg\n", b"a.jpg\na.jpg\n"):
        with pytest.raises(ValueError):
            dados.ler_lista(ruim)


def test_zip_nao_sai_do_destino():
    itens = dados.itens_cppe(["d/images/train/a.jpg", "d/labels/train/a.txt", "d/LICENSE", "images/val/"])
    assert itens == [{"parte": "train", "nome": "train/a.jpg", "imagem": "d/images/train/a.jpg",
                      "rotulo": "d/labels/train/a.txt"}]
    for ruim in ("../images/train/a.jpg", "images/train/../../a.jpg", "/images/train/a.jpg",
                 "images\\train\\a.jpg"):
        with pytest.raises(ValueError):
            dados.itens_cppe([ruim])


def test_redimensionar():
    saida, original, final = dados.redimensionar(imagem(4000, 1000, orientacao=6), 1280, 95)
    assert original == (1000, 4000) and final == (320, 1280)  # EXIF girou antes de medir
    with Image.open(io.BytesIO(saida)) as copia:
        assert copia.size == final and copia.format == "JPEG" and 0x0112 not in copia.getexif()
    saida, original, final = dados.redimensionar(imagem(300, 200, "PNG"), 1280, 95)
    assert original == final == (300, 200)  # sem ampliar
    assert saida.startswith(b"\x89PNG")


def test_phash_e_pares_iguais_a_forca_bruta():
    base = imagem(640, 480, semente=1)
    menor = dados.redimensionar(base, 320, 80)[0]
    assert (dados.phash(base) ^ dados.phash(menor)).bit_count() <= 4
    assert (dados.phash(base) ^ dados.phash(imagem(640, 480, semente=2))).bit_count() > 10

    sorteio = random.Random(0)
    hashes = [sorteio.getrandbits(64) for _ in range(200)]
    for i in range(0, 60, 3):  # vizinhos plantados a 0..19 bits
        hashes.append(hashes[i] ^ sum(1 << b for b in sorteio.sample(range(64), i // 3)))
    n = len(hashes)
    distancias = {(i, j): (hashes[i] ^ hashes[j]).bit_count() for i in range(n) for j in range(i + 1, n)}
    pares, histograma = dados.pares_proximos(hashes, 10, bloco=7)
    assert pares == [(i, j, d) for (i, j), d in distancias.items() if d <= 10]
    assert histograma.sum() == len(distancias)
    assert histograma[:11].sum() == len(pares)


def par(a, b, distancia, identico=False):
    return {"a": a, "b": b, "distancia": distancia, "identico": identico}


def test_revisao_preserva_decisoes_e_exige_todas(tmp_path):
    arquivo = tmp_path / "revisao.csv"
    pares = [par("a", "b", 3), par("a", "c", 7), par("d", "e", 12, identico=True)]
    faixa = dados.faixa_revisao(pares, 4, 10)
    assert faixa == [pares[1]]
    dados.gravar_revisao(faixa, arquivo)
    with pytest.raises(ValueError, match="1 pares"):
        dados.decidir(pares, dados.ler_decisoes(arquivo), 4, 10)

    # O autor salva pelo Excel em português (BOM e ponto e vírgula); depois surge um par novo.
    arquivo.write_text("n;distancia;a;b;duplicata\n1;7;a;c;Sim\n", encoding="utf-8-sig")
    antes = arquivo.read_bytes()
    dados.gravar_revisao(faixa, arquivo)
    assert arquivo.read_bytes() == antes
    linhas = dados.gravar_revisao(faixa + [par("f", "g", 9)], arquivo)
    assert [(x["n"], x["duplicata"]) for x in linhas] == [("1", "Sim"), ("2", "")]

    marcados = dados.decidir(pares, dados.ler_decisoes(arquivo), 4, 10)
    assert [p["duplicata"] for p in marcados] == [True, True, True]
    assert not dados.decidir(pares, {}, 4, 6)[1]["duplicata"]  # faixa reduzida

    # Precaução: vale só acima da revisão e não troca um "nao" do autor.
    outros = [par("a", "c", 10), par("a", "d", 12), par("a", "e", 14)]
    marcados = dados.decidir(outros, {("a", "c"): False}, 4, 10, 12)
    assert [(p["duplicata"], p["origem"]) for p in marcados] == [
        (False, "revisao"), (True, "precaucao"), (False, "acima do limiar")]

    arquivo.write_text("n,distancia,a,b,duplicata\n1,7,a,c,talvez\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sim ou nao"):
        dados.ler_decisoes(arquivo)


def manifesto_sintetico():
    sorteio, m = random.Random(1), {}
    for base, parte, primeiro, total in (("sh17", "train", 0, 400), ("sh17", "val", 1000, 80),
                                         ("cppe", "train", 0, 200), ("cppe", "val", 0, 50)):
        for i in range(primeiro, primeiro + total):
            nome = f"{i:04d}.jpg" if base == "sh17" else f"{parte}/{FAMILIAS[i % 2]}{i:04d}.jpg"
            classes = "0 " + sorteio.choice(["1", "2", "2", "3", "3", "3"]) if i % 20 else ""
            m[f"{base}/{nome}"] = {"base": base, "parte": parte, "nome": nome, "classes": classes.strip()}
    return m


def dup(a, b, duplicata=True):
    return par(a, b, 0) | {"duplicata": duplicata, "origem": "automatica"}


def S(i):
    return f"sh17/{i:04d}.jpg"


def C(i, parte="train"):
    return f"cppe/{parte}/{FAMILIAS[i % 2]}{i:04d}.jpg"


def test_regras_de_remocao_e_divisao():
    m = manifesto_sintetico()
    pares = [
        dup(S(1), C(1)),           # treino x treino entre bases: sai só de E1 (CPPE é teste de E1)
        dup(S(1000), C(2)),        # teste SH17 x treino CPPE: sai só de E2
        dup(S(3), S(1001)),        # treino x teste interno de E1
        dup(C(4), C(0, "val")),    # treino x teste interno de E2
        dup(S(5), S(6)), dup(S(6), S(7)),  # grupo no treino de E1
        dup(C(8), C(9)),
        dup(S(10), S(1002), duplicata=False),
        dup(S(1003), S(1004)),     # dentro do teste: nada muda
    ]
    listas, efeitos = dados.dividir(m, pares, 0.15, 0, FAMILIAS)

    assert efeitos["e1"]["removidas"] == sorted([S(1), S(3)])
    assert efeitos["e2"]["removidas"] == sorted([C(2), C(4)])
    assert len(listas["e1_teste"]) == 80 and len(listas["e2_teste"]) == 50
    assert len(listas["e1_teste_cruzado"]) == 250 and len(listas["e2_teste_cruzado"]) == 80
    assert all(x.startswith("cppe6/") for x in listas["e1_teste_cruzado"])
    assert all(x.startswith("cppe/") for x in listas["e2_treino"])
    assert "cppe6/" + C(1)[5:] in listas["e1_teste_cruzado"]  # o teste não perde o par
    assert C(1) in listas["e2_treino"] + listas["e2_desenvolvimento"]

    for e, total in (("e1", 398), ("e2", 198)):
        treino, dev = set(listas[f"{e}_treino"]), set(listas[f"{e}_desenvolvimento"])
        assert not treino & dev and len(treino | dev) == total
        assert abs(len(dev) - 0.15 * total) <= 2
        assert not (treino | dev) & set(listas[f"{e}_teste"] + listas[f"{e}_teste_cruzado"])
    for grupo, e in (([S(5), S(6), S(7)], "e1"), ([C(8), C(9)], "e2")):
        assert len({x in listas[f"{e}_desenvolvimento"] for x in grupo}) == 1

    embaralhado = dict(random.Random(9).sample(list(m.items()), len(m)))
    assert dados.dividir(embaralhado, pares[::-1], 0.15, 0, FAMILIAS)[0] == listas
    assert dados.dividir(m, pares, 0.15, 1, FAMILIAS)[0] != listas


def test_divisao_estratificada():
    grupos = [[f"raro{i}"] for i in range(40)] + [[f"comum{i}"] for i in range(400)]
    dev = dados.separar(grupos, lambda g: g[0].rstrip("0123456789"), 0.15, 0)
    assert len(dev) == 66 and sum(x.startswith("raro") for x in dev) == 6


def test_preparar_congelar_e_conferir(tmp_path):
    origem = {}
    itens = dados.itens_sh17(["a.jpeg", "b.png"], ["c.jpg"])
    for n, item in enumerate(itens):
        origem[item["imagem"]] = imagem(2000, 1000, "PNG" if item["nome"].endswith("png") else "JPEG", n)
        origem[item["rotulo"]] = b"10 .5 .5 .1 .1\r\n6 .5 .5 .2 .2\r\n"
        largura = 999 if n == 2 else 2000
        origem[item["voc"]] = f"<a><size><width>{largura}</width><height>1000</height></size></a>".encode()
        origem[item["meta"]] = b'{"photographer_id": %d}' % (7 if n != 1 else 8)
    preparado = tmp_path / "data" / "preparado"
    resumo = dados.preparar("sh17", itens, origem.__getitem__, preparado, {10: 1, 0: 0}, 1280, 95)
    assert resumo["imagens"] == {"train": 2, "val": 1} and resumo["rotulos"] == 3
    assert resumo["caixas_projeto"] == {"train": {"1": 2}, "val": {"1": 1}}
    assert resumo["caixas_nativas"]["val"] == {"10": 1, "6": 1}
    assert resumo["voc_divergentes"] == ["c.jpg"] and not resumo["rotulos_fora_da_faixa"]
    assert resumo["fotografos"] == {"train": 2, "val": 1, "em_comum": 1,
                                    "imagens_val_de_fotografo_do_train": 1}
    assert "photographer" not in (preparado / "sh17.csv").read_text() + (preparado / "sh17.json").read_text()

    zipado = {"images/train/source-1.jpg": imagem(), "images/val/source-2.jpg": imagem(semente=3),
              "labels/train/source-1.txt": b"0 .5 .5 .1 .1\n6 .4 .4 .1 .1\n"}
    dados.preparar("cppe", dados.itens_cppe(list(zipado)), zipado.__getitem__, preparado,
                   {0: 1, 6: 6}, 1280, 95, vista_comum="cppe6")
    assert (preparado / "cppe/labels/train/source-1.txt").read_text().split()[::5] == ["1", "6"]
    assert (preparado / "cppe6/labels/train/source-1.txt").read_text().split()[::5] == ["1"]
    assert (preparado / "cppe/labels/val/source-2.txt").read_text() == ""  # fundo

    m = dados.ler_manifesto(preparado)
    assert dados.conferir_preparado(preparado, m) == 5
    pacote = tmp_path / "drive" / "bases.tar"
    dados.empacotar(preparado, pacote)
    dados.restaurar(pacote, tmp_path / "restaurado")
    assert dados.conferir_preparado(tmp_path / "restaurado" / "preparado", m) == 5

    pares = [p | {"duplicata": False, "origem": "x"} for p in dados.achar_pares(m, 64)[0]]
    listas, efeitos = dados.dividir(m, pares, 0.5, 0, FAMILIAS)
    divisoes = tmp_path / "divisoes"
    relatorio = dados.congelar(divisoes, listas, efeitos, pares, m, {"semente": 0})
    assert relatorio["listas"]["e1_teste_cruzado"] == {"imagens": 2, "caixas": {1: 1}}
    dados.congelar(divisoes, listas, efeitos, pares, m, {"semente": 0})  # repetir é permitido
    with pytest.raises(FileExistsError):
        dados.congelar(divisoes, listas | {"e1_teste": []}, efeitos, pares, m, {})

    alvo = divisoes / "e1_teste.txt"
    alvo.write_bytes(alvo.read_bytes().replace(b"\n", b"\r\n"))  # conversão do git no Windows
    dados.listas_ultralytics(divisoes, preparado)
    assert (preparado / "e1_teste.txt").read_text() == "./sh17/images/c.jpg\n"
    alvo.write_text("sh17/outra.jpg\n")
    with pytest.raises(ValueError, match="e1_teste.txt"):
        dados.verificar_divisoes(divisoes)


class Resposta:
    def __init__(self, status, corpo, cabecalhos):
        self.status_code, self.corpo, self.headers = status, corpo, cabecalhos

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, _):
        yield self.corpo


@pytest.mark.parametrize("aceita_range", [True, False])
def test_baixar_retoma_e_confere(tmp_path, monkeypatch, aceita_range):
    conteudo, pedidos = bytes(range(100)), []

    def get(url, headers, **_):
        pedidos.append(headers.get("Range"))
        if len(pedidos) == 1:  # conexão cai no meio
            return Resposta(200, conteudo[:40], {"Content-Length": "100"})
        if aceita_range and headers.get("Range"):
            return Resposta(206, conteudo[40:],
                            {"Content-Length": "60", "Content-Range": "bytes 40-99/100"})
        return Resposta(200, conteudo, {"Content-Length": "100"})

    monkeypatch.setattr(requests, "get", get)
    md5 = hashlib.md5(conteudo).hexdigest()
    destino = dados.baixar("u", tmp_path / "x.zip", md5, "md5", espera=0)
    assert destino.read_bytes() == conteudo and pedidos == [None, "bytes=40-"]
    with pytest.raises(ValueError):
        dados.baixar("u", destino, "0" * 32, "md5")
    assert destino.exists()  # arquivo pronto nunca é apagado

    pedidos[:] = ["", ""]
    with pytest.raises(ValueError, match="parcial apagado"):
        dados.baixar("u", tmp_path / "y.zip", "0" * 32, "md5", espera=0)
    assert not list(tmp_path.glob("y.zip*"))


def test_divisoes_do_repositorio():
    """Vale depois que o autor trouxer do Colab os arquivos de configs/divisoes/."""
    pasta = RAIZ / "configs" / "divisoes"
    if not (pasta / "relatorio.json").exists():
        pytest.skip("divisões ainda não geradas")
    relatorio = dados.verificar_divisoes(pasta)
    assert relatorio["modo"] == "completo"

    def lista(nome):
        linhas = (pasta / f"{nome}.txt").read_text(encoding="utf-8").split()
        return {x.replace("cppe6/", "cppe/", 1) for x in linhas}

    tamanhos = {"e1_teste": 1620, "e1_teste_cruzado": 1000, "e2_teste": 200, "e2_teste_cruzado": 1620}
    assert {n: len(lista(n)) for n in tamanhos} == tamanhos
    duplicadas = [x for x in dados._ler_csv(pasta / "duplicatas.csv") if x["duplicata"] == "sim"]
    for e in ("e1", "e2"):
        treino, dev = lista(f"{e}_treino"), lista(f"{e}_desenvolvimento")
        teste = lista(f"{e}_teste") | lista(f"{e}_teste_cruzado")
        assert not treino & dev and not (treino | dev) & teste
        for x in duplicadas:
            for a, b in ((x["a"], x["b"]), (x["b"], x["a"])):
                assert not (a in treino | dev and b in teste)
                assert not (a in treino and b in dev)
