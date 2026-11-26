"""Preparação das bases (Fase 2): download, cópia, classes, duplicatas e divisões.

Uma imagem é identificada pela chave `base/nome`, com `nome` relativo à pasta `images` da base.
"""

import csv
import hashlib
import io
import json
import os
import random
import re
import shutil
import tarfile
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageOps

# A origem é fixada por impressão digital e tem fotos de 8K ou mais.
Image.MAX_IMAGE_PIXELS = 300_000_000
# Nome genérico de cliente; o Kaggle responde 404 ao nome padrão do `requests`.
CABECALHOS = {"User-Agent": "downloader", "Accept-Encoding": "identity"}
EXTENSOES = {".jpg", ".jpeg", ".png"}
# Origem (base, parte) do conjunto de treino e dos testes de cada experimento.
EXPERIMENTOS = {
    "e1": {
        "treino": [("sh17", "train")],
        "teste": [("sh17", "val")],
        "teste_cruzado": [("cppe", "train"), ("cppe", "val")],
    },
    "e2": {
        "treino": [("cppe", "train")],
        "teste": [("cppe", "val")],
        "teste_cruzado": [("sh17", "val")],
    },
}
# Pasta lida em lugar da base: E1 testa na Construction-PPE só com os ids 0-5.
VISTAS = {("e1", "cppe"): "cppe6"}
CLASSES_COMUNS = 6


def carregar_config(caminho: str | Path) -> dict:
    from ultralytics.utils import YAML  # aqui, por causa de configurar_ultralytics()

    return YAML.load(caminho)


# --- download e leitura ------------------------------------------------------------------


def impressao(caminho: str | Path, algoritmo: str = "sha256") -> str:
    resumo = hashlib.new(algoritmo)
    with open(caminho, "rb") as arquivo:
        while bloco := arquivo.read(1 << 22):
            resumo.update(bloco)
    return resumo.hexdigest()


def baixar(url: str, destino: str | Path, esperado: str, algoritmo: str = "sha256",
           tentativas: int = 10, espera: float = 5) -> Path:
    """Baixa com retomada e só entrega `destino` depois de conferir a impressão digital."""
    destino = Path(destino)
    if destino.exists():
        if impressao(destino, algoritmo) != esperado:
            raise ValueError(f"{destino.name} já existe e o {algoritmo} difere do esperado.")
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_name(destino.name + ".parte")
    for tentativa in range(1, tentativas + 1):
        inicio = parcial.stat().st_size if parcial.exists() else 0
        cabecalhos = CABECALHOS | ({"Range": f"bytes={inicio}-"} if inicio else {})
        try:
            with requests.get(url, headers=cabecalhos, stream=True, timeout=60) as resposta:
                if resposta.status_code == 416:  # nada a retomar: o parcial já está completo
                    break
                resposta.raise_for_status()
                retoma = resposta.status_code == 206  # 200: o servidor ignorou o Range
                if retoma and not resposta.headers["Content-Range"].startswith(f"bytes {inicio}-"):
                    raise requests.ConnectionError("trecho diferente do pedido")
                total = (inicio if retoma else 0) + int(resposta.headers.get("Content-Length", 0))
                with open(parcial, "ab" if retoma else "wb") as arquivo:
                    for bloco in resposta.iter_content(1 << 20):
                        arquivo.write(bloco)
            if parcial.stat().st_size < total:
                raise requests.ConnectionError("download incompleto")
            break
        except requests.RequestException as erro:
            status = erro.response.status_code if isinstance(erro, requests.HTTPError) else 0
            if (400 <= status < 500 and status != 429) or tentativa == tentativas:  # recusa: não insiste
                raise
            print(f"tentativa {tentativa} falhou ({type(erro).__name__}); retomando")
            time.sleep(espera * tentativa)
    obtido = impressao(parcial, algoritmo)
    if obtido != esperado:
        parcial.unlink()
        raise ValueError(f"{destino.name}: {algoritmo} {obtido} difere de {esperado}; parcial apagado.")
    parcial.replace(destino)
    return destino


def baixar_avulso(url: str) -> bytes:
    """Um arquivo do Kaggle, que pode vir dentro de um zip."""
    resposta = requests.get(url, headers=CABECALHOS, timeout=120)
    resposta.raise_for_status()
    if not resposta.content.startswith(b"PK\x03\x04"):
        return resposta.content
    with zipfile.ZipFile(io.BytesIO(resposta.content)) as pacote:
        (membro,) = pacote.namelist()
        return pacote.read(membro)


def nome_seguro(nome: str) -> str:
    """Recusa nomes que gravariam fora da pasta de destino."""
    if nome in ("", ".", "..") or any(c in nome for c in "/\\:\0"):
        raise ValueError(f"nome de arquivo inválido: {nome!r}")
    return nome


def ler_lista(dados: bytes) -> list[str]:
    linhas = dados.decode("utf-8-sig").splitlines()
    nomes = [nome_seguro(linha.strip()) for linha in linhas if linha.strip()]
    if len(set(nomes)) != len(nomes):
        raise ValueError("lista com nomes repetidos")
    return nomes


def itens_sh17(treino: list[str], val: list[str]) -> list[dict]:
    return [
        {
            "parte": parte, "nome": nome_seguro(nome), "imagem": f"images/{nome}",
            "rotulo": f"labels/{Path(nome).stem}.txt",
            "voc": f"voc_labels/{Path(nome).stem}.xml",
            "meta": f"meta-data/{Path(nome).stem}.json",
        }
        for parte, nomes in (("train", treino), ("val", val))
        for nome in nomes
    ]


def itens_cppe(membros: list[str]) -> list[dict]:
    """Imagens do zip da Construction-PPE, em `[prefixo/]images/<parte>/<nome>`."""
    itens = []
    for membro in sorted(membros):
        caminho = PurePosixPath(membro)
        if caminho.is_absolute() or ".." in caminho.parts or "\\" in membro:
            raise ValueError(f"caminho inseguro no zip: {membro!r}")
        if len(caminho.parts) < 3 or caminho.parts[-3] != "images":
            continue
        if caminho.suffix.lower() not in EXTENSOES:
            continue
        parte, nome = caminho.parts[-2], nome_seguro(caminho.name)
        itens.append({
            "parte": parte, "nome": f"{parte}/{nome}", "imagem": membro,
            "rotulo": str(caminho.parents[2] / "labels" / parte / f"{caminho.stem}.txt"),
        })
    return itens


# --- cópia preparada ---------------------------------------------------------------------


def converter_rotulos(texto: str, mapa: dict[int, int]) -> tuple[list[str], Counter]:
    """Troca os ids nativos pelos do projeto e descarta os que não estão no mapa."""
    linhas, nativas = [], Counter()
    for linha in texto.splitlines():
        campos = linha.split()
        if not campos:
            continue
        if len(campos) != 5:
            raise ValueError(f"rótulo YOLO com {len(campos)} campos: {linha!r}")
        nativa = int(campos[0])
        nativas[nativa] += 1
        if nativa in mapa:
            linhas.append(" ".join([str(mapa[nativa]), *campos[1:]]))
    return linhas, nativas


def redimensionar(dados: bytes, lado: int, qualidade: int):
    """Aplica o EXIF e limita o lado maior. Devolve (bytes, dimensões originais, finais)."""
    with Image.open(io.BytesIO(dados)) as imagem:
        formato = "PNG" if imagem.format == "PNG" else "JPEG"
        largura, altura = imagem.size
        if imagem.getexif().get(0x0112, 1) in (5, 6, 7, 8):  # orientações que giram 90 graus
            largura, altura = altura, largura
        escala = min(1.0, lado / max(largura, altura))
        final = (round(largura * escala), round(altura * escala))
        if escala < 1 and imagem.format == "JPEG":
            # Decodifica já reduzido, mantendo ao menos o dobro do tamanho final.
            imagem.draft("RGB", (int(imagem.width * escala * 2), int(imagem.height * escala * 2)))
        copia = ImageOps.exif_transpose(imagem).convert("RGB")
    if copia.size != final:
        copia = copia.resize(final, Image.Resampling.LANCZOS)
    saida = io.BytesIO()
    copia.save(saida, formato, **({"quality": qualidade} if formato == "JPEG" else {}))
    return saida.getvalue(), (largura, altura), final


def phash(dados: bytes) -> int:
    """Hash perceptual de 64 bits: DCT 32x32, bloco 8x8 comparado com a mediana."""
    cinza = cv2.imdecode(np.frombuffer(dados, np.uint8), cv2.IMREAD_GRAYSCALE)
    if cinza is None:
        raise ValueError("imagem ilegível")
    pequena = cv2.resize(cinza, (32, 32), interpolation=cv2.INTER_AREA)
    baixas = cv2.dct(np.float32(pequena))[:8, :8]
    return int.from_bytes(np.packbits(baixas > np.median(baixas)).tobytes(), "big")


def _gravar_csv(caminho: Path, linhas: list[dict], campos: list[str], separador: str = ",") -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    parcial = caminho.with_name(caminho.name + ".parte")
    with open(parcial, "w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, campos, delimiter=separador, lineterminator="\n",
                                  extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(linhas)
    parcial.replace(caminho)


def _ler_csv(caminho: Path) -> list[dict]:
    texto = Path(caminho).read_text(encoding="utf-8-sig", errors="replace")
    separador = ";" if ";" in texto.split("\n", 1)[0] else ","  # Excel em português usa ";"
    return list(csv.DictReader(io.StringIO(texto), delimiter=separador))


def preparar(base: str, itens: list[dict], ler, destino: str | Path, mapa: dict,
             lado: int, qualidade: int, vista_comum: str | None = None) -> dict:
    """Grava imagens, rótulos, manifesto e resumo de uma base em `destino`.

    `ler(caminho)` devolve os bytes de um arquivo da origem (zip ou pasta). O resumo é
    gravado por último e marca a preparação como completa.
    """
    destino = Path(destino)
    mapa = {int(k): int(v) for k, v in mapa.items()}

    def gravar(vista, nome, imagem, linhas):
        alvo = destino / vista / "images" / nome
        rotulo = (destino / vista / "labels" / nome).with_suffix(".txt")
        alvo.parent.mkdir(parents=True, exist_ok=True)
        rotulo.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(imagem, bytes):
            alvo.write_bytes(imagem)
        else:  # mesma imagem em outra vista: ligação física, sem ocupar disco
            alvo.unlink(missing_ok=True)
            try:
                os.link(imagem, alvo)
            except OSError:
                shutil.copyfile(imagem, alvo)
        rotulo.write_text("".join(f"{linha}\n" for linha in linhas), encoding="utf-8")
        return alvo

    def um(item):
        try:
            original = ler(item["imagem"])
            copia, (largura, altura), (w, h) = redimensionar(original, lado, qualidade)
            try:
                texto, tem_rotulo = ler(item["rotulo"]).decode("utf-8-sig"), True
            except (KeyError, FileNotFoundError):  # sem arquivo de rótulo: imagem de fundo
                texto, tem_rotulo = "", False
            linhas, nativas = converter_rotulos(texto, mapa)
            alvo = gravar(base, item["nome"], copia, linhas)
            if vista_comum:
                comuns = [x for x in linhas if int(x.split()[0]) < CLASSES_COMUNS]
                gravar(vista_comum, item["nome"], alvo, comuns)
            # Ultralytics descarta a imagem cujo rótulo sai desta faixa.
            fora = any(not -0.01 <= float(v) <= 1.01 for x in linhas for v in x.split()[1:])
            voc_difere = False
            if "voc" in item:
                xml = ler(item["voc"])  # só as dimensões: sem interpretar o XML
                voc = tuple(int(re.search(rb"<%s>\s*(\d+)" % t, xml)[1]) for t in (b"width", b"height"))
                voc_difere = voc != (largura, altura)
            autor = json.loads(ler(item["meta"])).get("photographer_id") if "meta" in item else None
            linha = {
                "base": base, "parte": item["parte"], "nome": item["nome"],
                "largura": largura, "altura": altura,
                "largura_preparada": w, "altura_preparada": h,
                "sha256": hashlib.sha256(original).hexdigest(),
                "phash": f"{phash(copia):016x}",
                "classes": " ".join(x.split()[0] for x in linhas),
            }
        except Exception as erro:
            raise RuntimeError(f"falha em {base}: {item['imagem']}") from erro
        return linha, nativas, tem_rotulo, voc_difere, autor, len(original), fora

    inicio = time.perf_counter()
    with ThreadPoolExecutor(os.cpu_count()) as grupo:
        resultados = list(grupo.map(um, itens))
    segundos = time.perf_counter() - inicio

    resumo = {
        "imagens": dict(Counter(item["parte"] for item in itens)),
        "rotulos": sum(r[2] for r in resultados),
        "caixas_nativas": defaultdict(Counter),
        "caixas_projeto": defaultdict(Counter),
        "voc_divergentes": sorted(r[0]["nome"] for r in resultados if r[3]),
        "rotulos_fora_da_faixa": sorted(r[0]["nome"] for r in resultados if r[6]),
        "bytes_originais": sum(r[5] for r in resultados),
        "segundos": round(segundos, 1),
    }
    autores = defaultdict(list)
    for linha, nativas, _, _, autor, *_ in resultados:
        resumo["caixas_nativas"][linha["parte"]].update(nativas)
        resumo["caixas_projeto"][linha["parte"]].update(map(int, linha["classes"].split()))
        if autor is not None:
            autores[linha["parte"]].append(autor)
    if autores:  # só contagens: nomes e ids de fotógrafos não saem daqui
        comuns = set(autores["train"]) & set(autores["val"])
        resumo["fotografos"] = {
            "train": len(set(autores["train"])), "val": len(set(autores["val"])),
            "em_comum": len(comuns),
            "imagens_val_de_fotografo_do_train": sum(a in comuns for a in autores["val"]),
        }
    campos = list(resultados[0][0])
    linhas = sorted((r[0] for r in resultados), key=lambda x: x["nome"])
    _gravar_csv(destino / f"{base}.csv", linhas, campos)
    resumo = json.loads(json.dumps(resumo))  # chaves em texto, como na releitura
    (destino / f"{base}.json").write_text(json.dumps(resumo, indent=2), encoding="utf-8")
    return resumo


def ler_manifesto(preparado: str | Path, bases=("sh17", "cppe")) -> dict[str, dict]:
    manifesto = {}
    for base in bases:
        for linha in _ler_csv(Path(preparado) / f"{base}.csv"):
            manifesto[f"{base}/{linha['nome']}"] = linha
    return manifesto


def conferir_preparado(preparado: str | Path, manifesto: dict[str, dict]) -> int:
    """Lê todas as imagens preparadas, confere as dimensões e a presença dos rótulos."""
    preparado = Path(preparado)

    def um(linha):
        vistas = [linha["base"]] + (["cppe6"] if linha["base"] == "cppe" else [])
        for vista in vistas:
            caminho = preparado / vista / "images" / linha["nome"]
            imagem = cv2.imdecode(np.fromfile(caminho, np.uint8), cv2.IMREAD_COLOR)
            esperado = (int(linha["altura_preparada"]), int(linha["largura_preparada"]))
            if imagem is None or imagem.shape[:2] != esperado:
                raise ValueError(f"imagem preparada inválida: {vista}/{linha['nome']}")
            if not (preparado / vista / "labels" / linha["nome"]).with_suffix(".txt").is_file():
                raise ValueError(f"rótulo ausente: {vista}/{linha['nome']}")

    with ThreadPoolExecutor(os.cpu_count()) as grupo:
        list(grupo.map(um, manifesto.values()))
    return len(manifesto)


def empacotar(preparado: str | Path, destino: str | Path) -> None:
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_name(destino.name + ".parte")
    with tarfile.open(parcial, "w") as pacote:
        pacote.add(preparado, arcname="preparado")
    parcial.replace(destino)


def restaurar(pacote: str | Path, pasta_dados: str | Path) -> None:
    with tarfile.open(pacote) as arquivo:
        arquivo.extractall(pasta_dados, filter="data")  # recusa caminhos fora do destino


# --- duplicatas --------------------------------------------------------------------------


def pares_proximos(hashes: list[int], maximo: int, bloco: int = 1024):
    """Todos os pares (i < j) a até `maximo` bits, e o histograma das distâncias."""
    h = np.array(hashes, dtype=np.uint64)
    pares, histograma = [], np.zeros(65, np.int64)
    for i0 in range(0, len(h), bloco):
        distancias = np.bitwise_count(h[i0:i0 + bloco, None] ^ h[None, i0:])
        acima = np.triu(np.ones(distancias.shape, bool), 1)
        histograma += np.bincount(distancias[acima], minlength=65)
        for i, j in zip(*np.nonzero((distancias <= maximo) & acima)):
            pares.append((i0 + int(i), i0 + int(j), int(distancias[i, j])))
    return pares, histograma


def achar_pares(manifesto: dict[str, dict], maximo: int):
    chaves = sorted(manifesto)
    pares, histograma = pares_proximos([int(manifesto[c]["phash"], 16) for c in chaves], maximo)
    return [
        {
            "a": chaves[i], "b": chaves[j], "distancia": d,
            "identico": manifesto[chaves[i]]["sha256"] == manifesto[chaves[j]]["sha256"],
        }
        for i, j, d in pares
    ], histograma


def faixa_revisao(pares: list[dict], duplicata_ate: int, revisao_ate: int) -> list[dict]:
    faixa = [p for p in pares if not p["identico"] and duplicata_ate < p["distancia"] <= revisao_ate]
    return sorted(faixa, key=lambda p: (p["distancia"], p["a"], p["b"]))


def gravar_revisao(faixa: list[dict], caminho: str | Path) -> list[dict]:
    """Cria ou completa o CSV de revisão sem apagar decisões já registradas."""
    caminho = Path(caminho)
    linhas = _ler_csv(caminho) if caminho.exists() else []
    existentes = {(x["a"], x["b"]) for x in linhas}
    novos = [p for p in faixa if (p["a"], p["b"]) not in existentes]
    if novos or not caminho.exists():  # sem novidade, o arquivo do autor não é reescrito
        proximo = max((int(x["n"]) for x in linhas), default=0) + 1
        linhas += [
            {"n": n, "distancia": p["distancia"], "a": p["a"], "b": p["b"], "duplicata": ""}
            for n, p in enumerate(novos, proximo)
        ]
        # ";" abre direto em colunas no Excel em português.
        _gravar_csv(caminho, linhas, ["n", "distancia", "a", "b", "duplicata"], ";")
    return _ler_csv(caminho)


def ler_decisoes(caminho: str | Path) -> dict[tuple[str, str], bool]:
    valores = {"sim": True, "s": True, "nao": False, "não": False, "n": False}
    decisoes = {}
    for linha in _ler_csv(caminho):
        valor = (linha["duplicata"] or "").strip().lower()
        if not valor:
            continue
        if valor not in valores:
            raise ValueError(f"par {linha['n']}: use sim ou nao na coluna duplicata, não {valor!r}")
        decisoes[linha["a"], linha["b"]] = valores[valor]
    return decisoes


def decidir(pares: list[dict], decisoes: dict, duplicata_ate: int, revisao_ate: int,
            precaucao_ate: int = 0) -> list[dict]:
    """Marca cada par como duplicata ou não; interrompe se faltar decisão na faixa de revisão.

    Acima de `revisao_ate` e até `precaucao_ate`, o par é duplicata sem revisão visual.
    """
    marcados, pendentes = [], 0
    for par in pares:
        if par["identico"] or par["distancia"] <= duplicata_ate:
            duplicata, origem = True, "automatica"
        elif par["distancia"] <= revisao_ate:
            duplicata, origem = decisoes.get((par["a"], par["b"])), "revisao"
            pendentes += duplicata is None
        elif par["distancia"] <= precaucao_ate:
            duplicata, origem = True, "precaucao"
        else:
            duplicata, origem = False, "acima do limiar"
        marcados.append(par | {"duplicata": duplicata, "origem": origem})
    if pendentes:
        raise ValueError(f"{pendentes} pares da faixa de revisão sem decisão; divisão não gerada.")
    return marcados


# --- divisão -----------------------------------------------------------------------------


def familia(chave: str, familias: list[str]) -> str:
    nome = PurePosixPath(chave).name
    return next((prefixo for prefixo in familias if nome.startswith(prefixo)), "outra")


def separar(grupos: list[list[str]], estrato, fracao: float, semente: int) -> set[str]:
    """Sorteia grupos inteiros para desenvolvimento, estrato a estrato."""
    estratos = defaultdict(list)
    for grupo in sorted(sorted(g) for g in grupos):
        estratos[estrato(grupo)].append(grupo)
    sorteio, escolhidas, vistas = random.Random(semente), set(), 0
    for chave in sorted(estratos):
        sorteio.shuffle(estratos[chave])
        vistas += sum(map(len, estratos[chave]))
        # Cota acumulada: o arredondamento de um estrato é compensado no seguinte.
        falta = round(fracao * vistas) - len(escolhidas)
        for grupo in estratos[chave]:
            if len(grupo) <= falta:
                escolhidas.update(grupo)
                falta -= len(grupo)
    return escolhidas


def dividir(manifesto: dict[str, dict], pares: list[dict], fracao: float, semente: int,
            familias: list[str]) -> tuple[dict[str, list[str]], dict]:
    """Listas de cada experimento (linhas `vista/nome`) e o que foi removido ou agrupado.

    Sai do treino e do desenvolvimento a imagem com duplicata em um teste do mesmo
    experimento. Os testes nunca perdem imagens.
    """
    pai: dict[str, str] = {}

    def raiz(chave):
        while pai.setdefault(chave, chave) != chave:
            pai[chave] = pai[pai[chave]]
            chave = pai[chave]
        return chave

    vizinhos = defaultdict(set)
    for par in pares:
        if par["duplicata"]:
            vizinhos[par["a"]].add(par["b"])
            vizinhos[par["b"]].add(par["a"])
            pai[raiz(par["b"])] = raiz(par["a"])

    listas, efeitos = {}, {}
    for experimento, origens in EXPERIMENTOS.items():
        def de(papel):
            return sorted(c for c, m in manifesto.items() if (m["base"], m["parte"]) in origens[papel])

        testes = set(de("teste")) | set(de("teste_cruzado"))
        removidas = {c for c in de("treino") if vizinhos[c] & testes}
        restantes = [c for c in de("treino") if c not in removidas]
        contagem = Counter(x for c in restantes for x in manifesto[c]["classes"].split())
        grupos = defaultdict(list)
        for chave in restantes:
            grupos[raiz(chave)].append(chave)

        def estrato(grupo):
            presentes = {x for c in grupo for x in manifesto[c]["classes"].split()}
            rara = min(presentes, key=lambda x: (contagem[x], int(x)), default="fundo")
            da_cppe = manifesto[grupo[0]]["base"] == "cppe"
            return rara, familia(grupo[0], familias) if da_cppe else ""

        desenvolvimento = separar(list(grupos.values()), estrato, fracao, semente)
        conjuntos = {
            "treino": [c for c in restantes if c not in desenvolvimento],
            "desenvolvimento": sorted(desenvolvimento),
            "teste": de("teste"),
            "teste_cruzado": de("teste_cruzado"),
        }
        for papel, chaves in conjuntos.items():
            listas[f"{experimento}_{papel}"] = [
                f"{VISTAS.get((experimento, base), base)}/{nome}"
                for base, nome in (c.split("/", 1) for c in chaves)
            ]
        # Imagens mantidas ligadas a um teste só por uma cadeia de duplicatas (informativo).
        raizes_de_teste = {raiz(c) for c in testes if c in vizinhos}
        efeitos[experimento] = {
            "removidas": sorted(removidas),
            "mantidas_em_cadeia_com_teste": sum(raiz(c) in raizes_de_teste for c in restantes),
            "grupos_com_mais_de_uma": sum(len(g) > 1 for g in grupos.values()),
        }
    return listas, efeitos


# --- registro e uso das divisões ---------------------------------------------------------


def _sha_texto(caminho: Path) -> str:
    # Fim de linha normalizado: o git no Windows pode convertê-lo.
    return hashlib.sha256(caminho.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def congelar(destino: str | Path, listas: dict, efeitos: dict, pares: list[dict],
             manifesto: dict, relatorio: dict) -> dict:
    """Grava listas, manifesto, duplicatas e relatório. Não substitui lista já congelada."""
    destino = Path(destino)
    textos = {f"{nome}.txt": "".join(f"{x}\n" for x in linhas) for nome, linhas in listas.items()}
    for nome, texto in textos.items():
        arquivo = destino / nome
        if arquivo.exists() and arquivo.read_bytes().replace(b"\r\n", b"\n") != texto.encode():
            raise FileExistsError(f"{nome} já está congelada com outro conteúdo.")
    destino.mkdir(parents=True, exist_ok=True)
    for nome, texto in textos.items():
        (destino / nome).write_bytes(texto.encode())

    campos = list(next(iter(manifesto.values())))
    _gravar_csv(destino / "manifesto.csv", [manifesto[c] for c in sorted(manifesto)], campos)
    removidas = {e: set(v["removidas"]) for e, v in efeitos.items()}
    treinos = {e: {c for c, m in manifesto.items() if (m["base"], m["parte"]) in o["treino"]}
               for e, o in EXPERIMENTOS.items()}
    linhas = []
    for par in pares:
        acoes = []
        for e, origens in EXPERIMENTOS.items():
            for lado, outro in (("a", "b"), ("b", "a")):
                m = manifesto[par[outro]]
                em_teste = (m["base"], m["parte"]) in origens["teste"] + origens["teste_cruzado"]
                if par["duplicata"] and par[lado] in removidas[e] and em_teste:
                    acoes.append(f"{e}: remove {lado}")
            if par["duplicata"] and {par["a"], par["b"]} <= treinos[e] - removidas[e]:
                acoes.append(f"{e}: agrupa")
        linhas.append(par | {
            "identico": "sim" if par["identico"] else "nao",
            "duplicata": "sim" if par["duplicata"] else "nao",
            "acao": "; ".join(acoes),
        })
    _gravar_csv(destino / "duplicatas.csv", linhas,
                ["a", "b", "distancia", "identico", "duplicata", "origem", "acao"])

    def contagens(linhas_da_lista):
        chaves = (x.replace("cppe6/", "cppe/", 1) for x in linhas_da_lista)
        todas = Counter(int(k) for c in chaves for k in manifesto[c]["classes"].split())
        comum = linhas_da_lista and linhas_da_lista[0].startswith("cppe6/")
        return {k: n for k, n in sorted(todas.items()) if not comum or k < CLASSES_COMUNS}

    relatorio = relatorio | {
        "experimentos": {e: v | {"removidas": len(v["removidas"])} for e, v in efeitos.items()},
        "listas": {nome: {"imagens": len(x), "caixas": contagens(x)} for nome, x in listas.items()},
        "sha256": {n: _sha_texto(destino / n) for n in [*textos, "manifesto.csv", "duplicatas.csv"]},
    }
    (destino / "relatorio.json").write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return relatorio


def verificar_divisoes(pasta: str | Path) -> dict:
    """Confere as impressões digitais registradas no relatório."""
    pasta = Path(pasta)
    relatorio = json.loads((pasta / "relatorio.json").read_text(encoding="utf-8"))
    for nome, esperado in relatorio["sha256"].items():
        if _sha_texto(pasta / nome) != esperado:
            raise ValueError(f"{nome} difere da impressão digital registrada.")
    return relatorio


def listas_ultralytics(divisoes: str | Path, preparado: str | Path) -> None:
    """Converte as listas congeladas em listas de caminhos relativos para o Ultralytics."""
    for nome in verificar_divisoes(divisoes)["sha256"]:
        if not nome.endswith(".txt"):
            continue
        linhas = (Path(divisoes) / nome).read_text(encoding="utf-8").split()
        caminhos = (x.split("/", 1) for x in linhas)
        (Path(preparado) / nome).write_text(
            "".join(f"./{vista}/images/{resto}\n" for vista, resto in caminhos), encoding="utf-8"
        )


def verificar_vista(config: str | Path, preparado: str | Path, saida: str | Path,
                    imagens: int = 4, device="cpu") -> tuple[int, int]:
    """Roda `model.val` em poucas imagens do teste, com um modelo sem treino.

    Confere só a ligação entre YAML, listas, imagens e rótulos; as métricas não são usadas.
    """
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    config, preparado, saida = Path(config), Path(preparado).resolve(), Path(saida).resolve()
    dados = carregar_config(config)
    nomes = {int(k): v for k, v in dados["names"].items()}

    def rotulo(linha):
        vista, resto = linha.removeprefix("./").split("/images/", 1)
        return (preparado / vista / "labels" / resto).with_suffix(".txt")

    linhas = (preparado / dados["test"]).read_text(encoding="utf-8").split()
    linhas = [x for x in linhas if rotulo(x).stat().st_size][:imagens]
    # Ultralytics descarta caixas repetidas; a contagem esperada também.
    caixas = sum(len(set(rotulo(x).read_text().splitlines())) for x in linhas)
    amostra = f"_amostra_{config.stem}.txt"
    (preparado / amostra).write_text("".join(f"{x}\n" for x in linhas), encoding="utf-8")
    saida.mkdir(parents=True, exist_ok=True)
    temporario = saida / config.name
    temporario.write_text(
        f"path: {preparado.as_posix()}\ntrain: {amostra}\nval: {dados['val']}\ntest: {amostra}\n"
        "names:\n" + "".join(f"  {k}: {v}\n" for k, v in nomes.items()), encoding="utf-8"
    )
    modelo = YOLO("yolo26n.yaml")
    modelo.model = DetectionModel("yolo26n.yaml", nc=len(nomes), verbose=False)
    modelo.model.names = nomes
    lidas = []
    modelo.add_callback("on_val_end", lambda v: lidas.extend(v.dataloader.dataset.im_files))
    metricas = modelo.val(data=str(temporario), split="test", imgsz=640, batch=imagens, workers=0,
                          plots=False, device=device, project=str(saida), name=config.stem,
                          exist_ok=True, verbose=False)
    obtido = (len(lidas), int(metricas.nt_per_class.sum()))
    if obtido != (len(linhas), caixas) or not caixas:
        raise ValueError(f"{config.name}: lidas {obtido} imagens e caixas, esperado "
                         f"{(len(linhas), caixas)}")
    return obtido


def folhas_de_contato(celulas: list[tuple[Path, str]], destino: str | Path, prefixo: str,
                      colunas: int, linhas: int, lado: int = 420) -> list[Path]:
    """Grades de miniaturas com legenda, para inspeção visual."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    for antiga in destino.glob(f"{prefixo}_*.jpg"):
        antiga.unlink()
    arquivos = []
    for inicio in range(0, len(celulas), colunas * linhas):
        lote = celulas[inicio:inicio + colunas * linhas]
        altura = -(-len(lote) // colunas) * (lado + 20)
        folha = Image.new("RGB", (colunas * lado, altura), "white")
        desenho = ImageDraw.Draw(folha)
        for n, (caminho, legenda) in enumerate(lote):
            x, y = n % colunas * lado, n // colunas * (lado + 20)
            with Image.open(caminho) as imagem:
                imagem.thumbnail((lado - 8, lado - 8))
                folha.paste(imagem, (x + 4, y + 20))
            desenho.text((x + 4, y + 4), legenda, fill="black")
        arquivos.append(destino / f"{prefixo}_{len(arquivos) + 1:02d}.jpg")
        folha.save(arquivos[-1], quality=90)
    return arquivos
