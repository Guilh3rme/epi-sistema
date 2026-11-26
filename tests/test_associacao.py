import subprocess
import sys

import pytest

from epi.associacao import (
    EPIS, Deteccao, Estado, associar_a, associar_b, no_portao, regioes, selecionar_pessoa,
)

P, A, I = Estado.PRESENTE, Estado.AUSENTE, Estado.INCERTO
PAR = {
    "conf_min": 0.25, "conf_ok": 0.5, "conf_ponto": 0.5, "margem_borda": 0.01, "altura_min": 0.5,
    "capacete_largura": 1.0, "capacete_altura": 0.8, "oculos_margem": 0.25,
    "oculos_meia_altura": 0.15, "colete_margem": 0.15, "luva_avanco": 0.4, "luva_lado": 0.9,
    "calcado_lado": 0.7, "iou_duplicata": 0.5,
}
FORMA = (600, 400, 3)
PESSOA = Deteccao("pessoa", 0.9, (80, 40, 320, 580))


def pose(invisiveis=(), **mover):
    """Corpo de frente, em pé; o lado esquerdo da pessoa fica à direita na imagem."""
    xy = [
        (200, 90), (210, 80), (190, 80), (225, 85), (175, 85),
        (240, 150), (160, 150), (260, 220), (140, 220), (270, 290), (130, 290),
        (225, 300), (175, 300), (225, 420), (175, 420), (225, 530), (175, 530),
    ]
    return [(x, y, 0.1 if i in invisiveis else 0.9) for i, (x, y) in enumerate(xy)]


def item(classe, x, y, conf=0.9):
    return Deteccao(classe, conf, (x - 8, y - 8, x + 8, y + 8))


def completo():
    return [
        item("capacete", 200, 55), item("oculos", 200, 80), item("colete", 200, 220),
        item("luvas", 280, 315), item("luvas", 120, 315),
        item("calcado", 225, 550), item("calcado", 175, 550),
    ]


def sem(classe, dets):
    return [d for d in dets if d.classe != classe]


def ambas(dets, pontos=None):
    return (
        associar_a(PESSOA, dets, PAR),
        associar_b(PESSOA, pontos or pose(), dets, FORMA, PAR),
    )


def test_modulos_puros_nao_importam_torch():
    codigo = (
        "import sys, epi.associacao, epi.decisao; "
        "print({'torch', 'ultralytics'} & set(sys.modules))"
    )
    saida = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, check=True)
    assert saida.stdout.strip() == "set()"


def test_selecionar_pessoa():
    fraca = Deteccao("pessoa", 0.4, PESSOA.caixa)
    assert selecionar_pessoa([], PAR) == (False, None)
    assert selecionar_pessoa([PESSOA], PAR) == (True, PESSOA)
    assert selecionar_pessoa([fraca], PAR) == (True, None)
    assert selecionar_pessoa([PESSOA, fraca], PAR) == (True, None)  # duas pessoas
    assert selecionar_pessoa([Deteccao("pessoa", 0.1, PESSOA.caixa)], PAR) == (False, None)


def test_sem_pessoa_valida_ou_sem_pose_fica_tudo_incerto():
    incerto = dict.fromkeys(EPIS, I)
    assert associar_a(None, completo(), PAR) == incerto
    assert associar_b(None, pose(), completo(), FORMA, PAR) == incerto
    assert associar_b(PESSOA, None, completo(), FORMA, PAR) == incerto


def test_todos_os_itens_no_lugar():
    a, b = ambas(completo())
    assert a == b == dict.fromkeys(EPIS, P)


def test_sem_itens_fica_ausente():
    a, b = ambas([])
    assert a == b == dict.fromkeys(EPIS, A)


@pytest.mark.parametrize("classe", ["capacete", "oculos", "colete"])
def test_confianca_intermediaria_e_incerta(classe):
    dets = [Deteccao(d.classe, 0.3, d.caixa) if d.classe == classe else d for d in completo()]
    a, b = ambas(dets)
    assert a[classe] == b[classe] == I


def test_item_na_mao_presente_em_a_e_ausente_em_b():
    na_mao = [item("capacete", 130, 300), *sem("capacete", completo())]
    a, b = ambas(na_mao)
    assert (a["capacete"], b["capacete"]) == (P, A)


def test_item_fora_da_caixa_da_pessoa_nao_conta():
    a, b = ambas([item("capacete", 360, 55)])
    assert a["capacete"] == b["capacete"] == A


@pytest.mark.parametrize("classe", ["luvas", "calcado"])
def test_um_so_lado_nao_libera(classe):
    um = [d for d in completo() if d.classe != classe]
    um.append(next(d for d in completo() if d.classe == classe))
    a, b = ambas(um)
    assert a[classe] == b[classe] == A


@pytest.mark.parametrize("classe", ["luvas", "calcado"])
def test_caixas_repetidas_do_mesmo_item_nao_formam_um_par(classe):
    um = next(d for d in completo() if d.classe == classe)
    x1, y1, x2, y2 = um.caixa
    repetida = Deteccao(classe, 0.8, (x1 + 1, y1, x2 + 1, y2))
    a, b = ambas([*sem(classe, completo()), um, repetida])
    assert a[classe] == b[classe] == A


@pytest.mark.parametrize("classe", ["luvas", "calcado"])
def test_um_lado_com_confianca_baixa_e_incerto(classe):
    dets = completo()
    i = next(i for i, d in enumerate(dets) if d.classe == classe)
    dets[i] = Deteccao(classe, 0.3, dets[i].caixa)
    a, b = ambas(dets)
    assert a[classe] == b[classe] == I


def pes_juntos():
    pontos = pose()
    pontos[15], pontos[16] = (205, 530, 0.9), (195, 530, 0.9)
    return pontos


def test_mesma_deteccao_nao_vale_pelos_dois_lados():
    pontos = pes_juntos()
    esquerda, direita = regioes(pontos, FORMA, PAR)["calcado"]
    nos_dois = item("calcado", 200, 550)
    for caixa in (esquerda, direita):
        assert caixa[0] <= 200 <= caixa[2] and caixa[1] <= 550 <= caixa[3]
    assert associar_b(PESSOA, pontos, [nos_dois], FORMA, PAR)["calcado"] == A


def test_distribuicao_nao_gulosa_entre_os_lados():
    # A melhor detecção cabe nos dois lados; a outra, só no esquerdo.
    pontos = pes_juntos()
    nos_dois = item("calcado", 200, 550, 0.95)
    so_esquerda = item("calcado", 235, 550, 0.6)
    for dets in ([nos_dois, so_esquerda], [so_esquerda, nos_dois]):
        assert associar_b(PESSOA, pontos, dets, FORMA, PAR)["calcado"] == P


def test_parte_nao_visivel_fica_incerta_so_em_b():
    pontos = pose(invisiveis=(10,))  # punho direito
    a, b = ambas(completo(), pontos)
    assert (a["luvas"], b["luvas"]) == (P, I)
    # O lado visível sem luva continua ausente.
    assert associar_b(PESSOA, pontos, [], FORMA, PAR)["luvas"] == A


def test_deteccao_unica_com_um_lado_invisivel_nao_libera():
    pontos = pose(invisiveis=(10,))
    assert associar_b(PESSOA, pontos, [item("luvas", 280, 315)], FORMA, PAR)["luvas"] == I


def test_ponto_fora_do_quadro_nao_e_visivel():
    pontos = pose()
    pontos[15] = (225, 700, 0.9)
    assert regioes(pontos, FORMA, PAR)["calcado"][0] is None


@pytest.mark.parametrize(
    "invisiveis, epi",
    [((5,), "capacete"), (range(5), "capacete"), ((1, 2), "oculos"), ((5,), "oculos"),
     ((11,), "colete"), ((13,), "calcado"), ((7,), "luvas")],
)
def test_pontos_exigidos_por_regiao(invisiveis, epi):
    assert associar_b(PESSOA, pose(invisiveis), completo(), FORMA, PAR)[epi] == I


@pytest.mark.parametrize(
    "caixa, passa",
    [
        ((80, 40, 320, 580), True),
        ((0, 40, 200, 580), False),    # encostada na borda esquerda
        ((80, 40, 320, 598), False),   # dentro da margem inferior
        ((80, 300, 320, 580), False),  # baixa demais
    ],
)
def test_portao(caixa, passa):
    assert no_portao([Deteccao("pessoa", 0.9, caixa)], FORMA, PAR) is passa


def test_portao_sem_pessoa():
    assert not no_portao([item("capacete", 200, 55)], FORMA, PAR)
