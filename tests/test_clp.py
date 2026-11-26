import socket
import subprocess
import sys

import pytest

from epi import clp
from epi.decisao import BLOQUEADO, LIBERADO


def porta_livre():
    with socket.socket() as s:
        s.bind((clp.HOST, 0))
        return s.getsockname()[1]


@pytest.fixture
def simulador():
    porta = porta_livre()
    processo = subprocess.Popen(
        [sys.executable, "-m", "epi.clp", "--porta", str(porta)],
        stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
    )
    assert "CLP simulado" in processo.stdout.readline()
    yield porta, processo
    processo.kill()
    processo.wait()


def test_escreve_rele_e_zera(simulador):
    porta, processo = simulador
    cliente = clp.cliente(porta)

    escrita = clp.enviar(cliente, BLOQUEADO, 7, ("capacete", "colete"))
    assert escrita["ok"] and escrita["latencia_ms"] > 0
    assert clp.ler(cliente) == [2, 7, 0b10001]

    assert clp.enviar(cliente, LIBERADO, 8)["ok"]
    assert clp.ler(cliente) == [1, 8, 0]

    assert clp.enviar(cliente, None, 8)["ok"]
    assert clp.ler(cliente) == [0, 8, 0]
    cliente.close()

    processo.kill()
    impresso = processo.stdout.read()
    assert "[2, 7, 17]" in impresso and "[1, 8, 0]" in impresso and "[0, 8, 0]" in impresso


def test_sequencia_cabe_no_registrador(simulador):
    cliente = clp.cliente(simulador[0])
    assert clp.enviar(cliente, LIBERADO, 65537)["ok"]
    assert clp.ler(cliente) == [1, 1, 0]
    cliente.close()


def test_segundo_simulador_na_mesma_porta_recusa(simulador):
    comando = [sys.executable, "-m", "epi.clp", "--porta", str(simulador[0])]
    assert subprocess.run(comando, capture_output=True, timeout=30).returncode == 1


def test_servidor_parado_gera_falha_registrada(simulador):
    porta, processo = simulador
    cliente = clp.cliente(porta)
    assert clp.enviar(cliente, BLOQUEADO, 1)["ok"]
    processo.kill()
    processo.wait()
    for _ in range(2):  # conexão perdida e, depois, recusada
        escrita = clp.enviar(cliente, LIBERADO, 2)
        assert not escrita["ok"] and escrita["erro"]
    assert clp.ler(cliente) is None
    cliente.close()
