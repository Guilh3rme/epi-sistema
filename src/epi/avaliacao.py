"""Avaliação por classe e por tamanho (Fase 4): desenvolvimento e, uma única vez, os testes.

A execução grava predições, marcações e curvas; as tabelas saem só desses arquivos, e podem
ser refeitas sem nova inferência. Chamar `configurar_ultralytics()` antes.
"""

import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from epi import dados, treino
from epi.ambiente import RAIZ, registrar

EXECUCOES = {  # nome: (experimento, YAML em configs/, lista do teste)
    "e1_interno": ("e1", "base_e1.yaml", "e1_teste.txt"),
    "e1_cruzado": ("e1", "base_e1_cppe.yaml", "e1_teste_cruzado.txt"),
    "e2_interno": ("e2", "base_e2.yaml", "e2_teste.txt"),
    "e2_cruzado": ("e2", "base_e2_sh17.yaml", "e2_teste_cruzado.txt"),
}
FAIXAS = ("pequeno", "medio", "grande")
CORTES = (0, 32**2, 96**2, 1e10)  # px² com o lado maior da imagem em LADO (cortes do COCO)
LADO = 640
MAX_DET = 300  # o mesmo da validação do Ultralytics; o padrão do COCO é 100
MARCADOR = "concluido.json"
MEDIA = "média das principais"


def _gravar_json(caminho: Path, conteudo) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    parcial = caminho.with_name(caminho.name + ".parte")
    parcial.write_text(json.dumps(conteudo, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(parcial, caminho)


def _ler_json(caminho: Path):
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def preparar_yaml(execucao: str, split: str, preparado, divisoes, destino) -> tuple[Path, list[str]]:
    """YAML de uma execução e a lista de imagens do `split`. Em `val`, sem a chave `test`."""
    from ultralytics.utils import YAML

    experimento, arquivo, lista_teste = EXECUCOES[execucao]
    base = dados.carregar_config(RAIZ / "configs" / arquivo)
    esperado = {"train": f"{experimento}_treino.txt", "val": f"{experimento}_desenvolvimento.txt"}
    if split == "test":
        esperado["test"] = lista_teste
    obtido = {k: base.get(k) for k in esperado}
    if obtido != esperado:
        raise ValueError(f"listas inesperadas em {arquivo}: {obtido}")
    preparado = Path(preparado).resolve()
    dados.listas_ultralytics(divisoes, preparado)
    destino = Path(destino).resolve()
    destino.parent.mkdir(parents=True, exist_ok=True)
    YAML.save(destino, {"path": preparado.as_posix(), **esperado, "names": base["names"]})
    return destino, (preparado / esperado[split]).read_text(encoding="utf-8").split()


def marcacoes(preparado, lista: list[str]) -> list[dict]:
    """Dimensões e caixas marcadas (classe, cx, cy, l, a, normalizados) de cada imagem."""
    preparado = Path(preparado)
    saida = []
    for linha in lista:
        vista, resto = linha.removeprefix("./").split("/images/", 1)
        with Image.open(preparado / vista / "images" / resto) as imagem:
            largura, altura = imagem.size
        rotulo = (preparado / vista / "labels" / resto).with_suffix(".txt")
        # O Ultralytics descarta linhas repetidas; aqui também.
        caixas = sorted({tuple(map(float, x.split()))
                         for x in rotulo.read_text(encoding="utf-8").splitlines() if x.strip()})
        saida.append({"nome": Path(resto).name, "largura": largura, "altura": altura,
                      "caixas": [list(c) for c in caixas]})
    if len({m["nome"] for m in saida}) != len(saida):  # as predições são ligadas pelo nome
        raise ValueError("nomes de imagem repetidos na lista")
    return saida


def curvas(m) -> dict:
    """Por classe do modelo: contagens e, se houver caixas, AP e curvas P, R e F1."""
    caixa = m.box
    saida = {nome: {"id": int(i), "imagens": int(m.nt_per_image[i]), "caixas": int(m.nt_per_class[i])}
             for i, nome in m.names.items()}
    for linha, i in enumerate(caixa.ap_class_index):
        saida[m.names[int(i)]] |= {
            "ap50": float(caixa.ap50[linha]), "ap": float(caixa.ap[linha]),
            "p": np.asarray(caixa.p_curve[linha]).tolist(),
            "r": np.asarray(caixa.r_curve[linha]).tolist(),
            "f1": np.asarray(caixa.f1_curve[linha]).tolist(),
        }
    return saida


def _grade(pontos: int) -> np.ndarray:
    return np.linspace(0, 1, pontos)  # confianças das curvas do Ultralytics


def escolher_limiar(por_classe: dict, principais: list[str]) -> tuple[float, float]:
    """(confiança, F1) de maior F1 médio das classes principais, suavizado como no Ultralytics."""
    from ultralytics.utils.metrics import smooth

    faltam = [c for c in principais if "f1" not in por_classe.get(c, {})]
    if faltam:
        raise ValueError(f"sem caixas no desenvolvimento: {faltam}")
    media = np.mean([por_classe[c]["f1"] for c in principais], axis=0)
    i = int(smooth(media, 0.1).argmax())
    return float(_grade(len(media))[i]), float(media[i])


def diferencas(antes: dict, agora: dict) -> tuple[float, float]:
    """Maiores diferenças (de AP, de P e R) entre duas `metricas_desenvolvimento`."""
    if set(antes["por_classe"]) != set(agora["por_classe"]):
        raise RuntimeError("classes diferentes das registradas na F3")
    pares = [(antes["geral"], agora["geral"])]
    pares += [(v, agora["por_classe"][c]) for c, v in antes["por_classe"].items()]
    return tuple(max(abs(a[k] - b[k]) for a, b in pares for k in chaves)
                 for chaves in (("map50", "map50_95"), ("precisao", "revocacao")))


def desenvolvimento(experimento: str, cfg: dict, cfg_treino: dict, preparado, divisoes, saida,
                    copia, peso, referencia: dict | None = None, device=None) -> dict:
    """Valida no desenvolvimento, compara com a F3 e escolhe o limiar do experimento.

    `referencia` é o resumo da F3; sem ele (modo curto) nada é comparado e o teste não é
    liberado.
    """
    saida, copia = Path(saida).resolve(), Path(copia)
    sha = dados.impressao(peso)
    if referencia and sha != referencia["pesos_sha256"]["best.pt"]:
        raise ValueError(f"{experimento}: o peso não é o registrado na F3")
    arquivo = copia / "desenvolvimento.json"
    todos = _ler_json(arquivo) if arquivo.exists() else {}
    feito = todos.get(experimento)
    # O limiar já escolhido é mantido: recalculá-lo em outra sessão poderia diferir do publicado.
    if feito and feito["peso_sha256"] == sha and feito["reproduzido"] == (referencia is not None):
        print(f"{experimento}: desenvolvimento já concluído")
        return feito
    yaml, lista = preparar_yaml(f"{experimento}_interno", "val", preparado, divisoes,
                                saida / f"{experimento}_desenvolvimento.yaml")
    sessao = registrar(copia / "ambiente", semente=cfg_treino["treino"]["seed"])
    lidas = []
    m = treino.validar(peso, yaml, cfg_treino, saida, device, split="val",
                       nome=f"{experimento}_desenvolvimento", lidas=lidas, save_json=True)
    if len(lidas) != len(lista):
        raise RuntimeError(f"{experimento}: {len(lidas)} imagens lidas, esperado {len(lista)}")
    maior = maior_pr = None
    if referencia:
        # Só o AP tem tolerância: P e R são lidos no ponto de F1 máximo, que se desloca com
        # variações mínimas de confiança. A diferença deles é apenas registrada.
        maior, maior_pr = diferencas(referencia["metricas_desenvolvimento"],
                                     treino.resumir_metricas(m))
        if not maior <= cfg["tolerancia_f3"]:
            raise RuntimeError(f"{experimento}: AP do desenvolvimento difere da F3 em {maior:.4f}")
    limiar, f1 = escolher_limiar(curvas(m), cfg["principais"])
    resultado = {"limiar": limiar, "f1_medio": f1, "peso_sha256": sha,
                 "reproduzido": referencia is not None, "maior_diferenca_ap": maior,
                 "maior_diferenca_pr": maior_pr,
                 "imagens": len(lista), "sessao": sessao}
    _gravar_json(arquivo, todos | {experimento: resultado})
    return resultado


def _conferir_teste(experimento: str, cfg: dict, copia: Path, sha: str) -> None:
    """Recusa o teste sem limiar publicado e sem o desenvolvimento reproduzido com o mesmo peso."""
    limiar = cfg["limiares"].get(experimento)
    if limiar is None:
        raise RuntimeError(f"{experimento}: grave o limiar em configs/avaliacao.yaml antes do teste")
    arquivo = copia / "desenvolvimento.json"
    dev = (_ler_json(arquivo) if arquivo.exists() else {}).get(experimento)
    if not dev or not dev["reproduzido"] or dev["peso_sha256"] != sha:
        raise RuntimeError(f"{experimento}: desenvolvimento não reproduzido com este peso")
    if abs(dev["limiar"] - limiar) > 1e-6:
        raise RuntimeError(f"{experimento}: limiar {limiar} difere do escolhido, {dev['limiar']}")


def avaliar(execucao: str, split: str, cfg: dict, cfg_treino: dict, preparado, divisoes, saida,
            copia, peso, device=None) -> dict:
    """Roda uma execução e grava o marcador por último. Com o marcador, não roda de novo."""
    if split not in ("val", "test"):
        raise ValueError(f"divisão inesperada: {split}")
    experimento = EXECUCOES[execucao][0]
    saida, copia = Path(saida).resolve(), Path(copia)
    pasta = copia / execucao
    marcador = pasta / MARCADOR
    if marcador.exists():
        feito = _ler_json(marcador)
        if feito["split"] != split:
            raise RuntimeError(f"{execucao}: já há resultado de outra divisão nesta pasta")
        print(f"{execucao}: já concluída")
        return feito
    sha = dados.impressao(peso)
    if split == "test":
        _conferir_teste(experimento, cfg, copia, sha)
    yaml, lista = preparar_yaml(execucao, split, preparado, divisoes, saida / f"{execucao}.yaml")
    sessao = registrar(pasta / "ambiente", semente=cfg_treino["treino"]["seed"])
    if split == "test" and sessao["codigo_modificado"] is not False:
        raise RuntimeError("o teste exige o código publicado, sem modificações")
    gt = marcacoes(preparado, lista)  # antes da inferência: erro aqui não gasta o teste

    predicoes = saida / execucao / "predictions.json"
    predicoes.unlink(missing_ok=True)  # sobra de tentativa interrompida
    lidas = []
    m = treino.validar(peso, yaml, cfg_treino, saida, device, split=split, nome=execucao,
                       lidas=lidas, save_json=True)
    if len(lidas) != len(lista):
        raise RuntimeError(f"{execucao}: {len(lidas)} imagens lidas, esperado {len(lista)}")
    por_classe = curvas(m)
    contadas = np.bincount([int(c[0]) for i in gt for c in i["caixas"]], minlength=len(m.names))
    if any(contadas[v["id"]] != v["caixas"] for v in por_classe.values()):
        raise RuntimeError(f"{execucao}: caixas lidas diferem das marcações")

    pasta.mkdir(parents=True, exist_ok=True)
    if predicoes.exists():
        shutil.copyfile(predicoes, pasta / "predicoes.json")
    else:  # o Ultralytics não grava o arquivo quando não há nenhuma detecção
        _gravar_json(pasta / "predicoes.json", [])
    _gravar_json(pasta / "marcacoes.json", gt)
    _gravar_json(pasta / "curvas.json", por_classe)
    resumo = {
        "execucao": execucao, "experimento": experimento, "split": split,
        "peso_sha256": sha, "limiar": cfg["limiares"].get(experimento),
        "relatorio_divisoes_sha256": dados._sha_texto(Path(divisoes) / "relatorio.json"),
        "imagens": len(lista),
        "sha256": {n: dados.impressao(pasta / n)
                   for n in ("predicoes.json", "marcacoes.json", "curvas.json")},
        "sessoes": [_ler_json(p) for p in sorted((pasta / "ambiente").glob("*.json"))],
    }
    _gravar_json(marcador, resumo)  # por último
    return resumo


def por_tamanho(gt: list[dict], predicoes: list[dict], nomes: dict[int, str], limiar: float) -> dict:
    """{(classe, faixa): caixas, revocação no limiar, AP50 e AP50-95} pela regra de área do COCO.

    Cada imagem é reescalada para o lado maior `LADO`, e a área das caixas é medida nela.
    Valores `None` indicam faixa sem caixas.
    """
    from faster_coco_eval import COCO, COCOeval_faster

    imagens, anotacoes, escala = [], [], {}
    for n, item in enumerate(gt, 1):
        s = LADO / max(item["largura"], item["altura"])
        escala[item["nome"]] = (n, s)
        imagens.append({"id": n, "width": item["largura"] * s, "height": item["altura"] * s})
        for classe, cx, cy, w, h in item["caixas"]:
            w, h = w * item["largura"] * s, h * item["altura"] * s
            x, y = cx * item["largura"] * s - w / 2, cy * item["altura"] * s - h / 2
            anotacoes.append({"id": len(anotacoes) + 1, "image_id": n, "category_id": int(classe) + 1,
                              "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
    categorias = [{"id": i + 1, "name": nome} for i, nome in sorted(nomes.items())]

    def rodar(escolhidas):
        base = COCO({"images": imagens, "annotations": [dict(a) for a in anotacoes],
                     "categories": categorias})
        caixas = []
        for p in escolhidas:
            n, s = escala[p["file_name"]]
            caixas.append({"image_id": n, "category_id": p["category_id"], "score": p["score"],
                           "bbox": [v * s for v in p["bbox"]]})
        ev = COCOeval_faster(base, base.loadRes(caixas), iouType="bbox",
                             print_function=lambda *a, **k: None)
        ev.params.maxDets = [1, 10, MAX_DET]
        ev.evaluate()
        ev.accumulate()
        return ev

    todas = rodar(predicoes) if predicoes else None
    acima = [p for p in predicoes if p["score"] >= limiar]
    no_limiar = rodar(acima) if acima else None
    resultado = {}
    for k, categoria in enumerate(categorias):
        areas = [a["area"] for a in anotacoes if a["category_id"] == categoria["id"]]
        for a, faixa in enumerate(FAIXAS, 1):
            n = sum(CORTES[a - 1] <= area <= CORTES[a] for area in areas)  # limites como no COCO
            r = ap50 = ap = None
            if n:
                r = float(no_limiar.eval["recall"][0, k, a, -1]) if no_limiar else 0.0
                if todas:
                    precisao = todas.eval["precision"][:, :, k, a, -1]  # [IoU, revocação]
                    ap50, ap = float(precisao[0].mean()), float(precisao.mean())
                else:
                    ap50 = ap = 0.0
            resultado[categoria["name"], faixa] = {
                "caixas": n, "revocacao": r, "ap50": ap50, "ap50_95": ap}
    return resultado


def _blocos(execucao: str, cfg: dict, por_classe: dict) -> list[tuple[str, str]]:
    """(bloco, classe) na ordem das tabelas; classes de ausência só no teste interno de E2."""
    blocos = [("principal", c) for c in cfg["principais"]] + [("principal", MEDIA)]
    blocos += [("exploratória", c) for c in cfg["exploratorias"]]
    if execucao == "e2_interno":
        blocos += [("ausência", c) for c in por_classe if c.startswith("no_")]
    return blocos


def _media(linhas: list[dict], campos) -> dict:
    return {k: None if any(x[k] is None for x in linhas) else float(np.mean([x[k] for x in linhas]))
            for k in campos}


def linhas_por_classe(execucao: str, cfg: dict, por_classe: dict, limiar: float) -> list[dict]:
    linhas, principais = [], []
    for bloco, classe in _blocos(execucao, cfg, por_classe):
        if classe == MEDIA:
            valores = {"imagens": None, "caixas": None} | _media(
                principais, ("precisao", "revocacao", "f1", "map50", "map50_95"))
        else:
            c = por_classe[classe]
            valores = {"imagens": c["imagens"], "caixas": c["caixas"]}
            if "f1" in c:
                i = int(np.abs(_grade(len(c["f1"])) - limiar).argmin())
                valores |= {"precisao": c["p"][i], "revocacao": c["r"][i], "f1": c["f1"][i],
                            "map50": c["ap50"], "map50_95": c["ap"]}
            else:
                valores |= dict.fromkeys(("precisao", "revocacao", "f1", "map50", "map50_95"))
        linha = {"execucao": execucao, "bloco": bloco, "classe": classe} | valores
        linhas.append(linha)
        if bloco == "principal" and classe != MEDIA:
            principais.append(linha)
    return linhas


def linhas_por_tamanho(execucao: str, cfg: dict, por_classe: dict, faixas: dict) -> list[dict]:
    linhas, principais = [], []
    for bloco, classe in _blocos(execucao, cfg, por_classe):
        for faixa in FAIXAS:
            if classe == MEDIA:
                grupo = [x for x in principais if x["faixa"] == faixa]
                valores = {"caixas": None} | _media(grupo, ("revocacao", "ap50", "ap50_95"))
            else:
                valores = faixas[classe, faixa]
            linha = {"execucao": execucao, "bloco": bloco, "classe": classe, "faixa": faixa} | valores
            linhas.append(linha)
            if bloco == "principal" and classe != MEDIA:
                principais.append(linha)
    return linhas


def _markdown(linhas: list[dict]) -> str:
    def texto(v):
        if v is None:
            return "—"
        return f"{v:.3f}".replace(".", ",") if isinstance(v, float) else str(v)

    campos = list(linhas[0])
    tabela = [campos, ["---"] * len(campos)] + [[texto(x[c]) for c in campos] for x in linhas]
    return "".join(f"| {' | '.join(x)} |\n" for x in tabela)


def tabelas(copia, destino, cfg: dict) -> list[str]:
    """Monta CSV e Markdown das execuções concluídas, a partir dos arquivos salvos."""
    copia, destino = Path(copia), Path(destino)
    classe, tamanho, resumos = [], [], {}
    for execucao in EXECUCOES:
        pasta = copia / execucao
        if not (pasta / MARCADOR).exists():
            continue
        resumo = _ler_json(pasta / MARCADOR)
        for nome, sha in resumo["sha256"].items():
            if dados.impressao(pasta / nome) != sha:
                raise ValueError(f"{execucao}/{nome} difere do registrado")
        # Fora do teste (modo curto) vale o limiar do desenvolvimento salvo.
        limiar = resumo["limiar"]
        if resumo["split"] == "val" and limiar is None:
            limiar = _ler_json(copia / "desenvolvimento.json")[resumo["experimento"]]["limiar"]
        por_classe = _ler_json(pasta / "curvas.json")
        faixas = por_tamanho(_ler_json(pasta / "marcacoes.json"), _ler_json(pasta / "predicoes.json"),
                             {v["id"]: n for n, v in por_classe.items()}, limiar)
        classe += linhas_por_classe(execucao, cfg, por_classe, limiar)
        tamanho += linhas_por_tamanho(execucao, cfg, por_classe, faixas)
        resumos[execucao] = resumo | {"limiar": limiar}
    if not resumos:
        raise ValueError("nenhuma execução concluída")

    destino.mkdir(parents=True, exist_ok=True)
    splits = {r["split"] for r in resumos.values()}
    aviso = "" if splits == {"test"} else "**Desenvolvimento: não é resultado de teste.**\n\n"
    limiares = ", ".join(sorted({f"{r['experimento'].upper()} = {r['limiar']:.3f}".replace(".", ",")
                                 for r in resumos.values()}))
    notas = {
        "por_classe": f"P, R e F1 na confiança fixada no desenvolvimento ({limiares}). "
                      "Calçado e óculos são exploratórios.",
        "por_tamanho": "Faixas pela área da caixa com o lado maior da imagem em 640 px: cortes "
                       f"32² e 96². R na confiança fixada no desenvolvimento ({limiares}). AP com "
                       "a interpolação do COCO: não é somável nem comparável diretamente ao AP "
                       "por classe.",
    }
    for nome, linhas in (("por_classe", classe), ("por_tamanho", tamanho)):
        with open(destino / f"{nome}.csv", "w", encoding="utf-8", newline="") as arquivo:
            escritor = csv.DictWriter(arquivo, list(linhas[0]), lineterminator="\n")
            escritor.writeheader()
            escritor.writerows(linhas)
        (destino / f"{nome}.md").write_text(
            f"{aviso}{notas[nome]}\n\n{_markdown(linhas)}", encoding="utf-8", newline="\n")
    (destino / "resumo.json").write_text(
        json.dumps(resumos, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return list(resumos)


def entregar(copia, destino) -> Path:
    """Zip com predições, curvas, marcações, registros e tabelas (sem imagens nem pesos)."""
    copia = Path(copia)
    return Path(shutil.make_archive(str(Path(destino).with_suffix("")), "zip",
                                    root_dir=copia.parent, base_dir=copia.name))
