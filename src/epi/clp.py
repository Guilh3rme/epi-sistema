"""CLP simulado (servidor Modbus TCP, só em 127.0.0.1) e cliente.

Simulador: python -m epi.clp [--porta 5020]
"""

import argparse
import asyncio
import socket
import time
from datetime import datetime

from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException
from pymodbus.server import ModbusTcpServer
from pymodbus.simulator import DataType, SimData, SimDevice

from epi.associacao import EPIS
from epi.decisao import BLOQUEADO, INDETERMINADO, LIBERADO

HOST = "127.0.0.1"
PORTA = 5020
CODIGOS = {None: 0, LIBERADO: 1, BLOQUEADO: 2, INDETERMINADO: 3}


def cliente(porta=PORTA) -> ModbusTcpClient:
    # Espera curta e sem repetição: CLP fora do ar não pode travar o laço.
    return ModbusTcpClient(HOST, port=porta, timeout=0.5, retries=0)


def enviar(cliente, decisao, seq, ausentes=()) -> dict:
    """Escreve decisão, sequência e bits de ausentes em uma operação; falha não levanta exceção."""
    valores = [CODIGOS[decisao], seq % 65536, sum(1 << EPIS.index(e) for e in ausentes)]
    latencia = erro = None
    try:
        if not cliente.connect():
            erro = "sem conexão com o CLP"
        else:
            inicio = time.perf_counter()
            resposta = cliente.write_registers(0, valores, device_id=1)
            latencia = (time.perf_counter() - inicio) * 1000
            if resposta.isError():
                erro = str(resposta)
    except (ModbusException, OSError) as falha:
        erro = f"{type(falha).__name__}: {falha}"
    if erro:
        cliente.close()  # a próxima escrita reconecta
    return {"ok": erro is None, "valores": valores, "latencia_ms": latencia, "erro": erro}


def ler(cliente) -> list | None:
    """Os três registradores, ou None se o CLP não responder."""
    try:
        resposta = cliente.read_holding_registers(0, count=3, device_id=1)
    except (ModbusException, OSError):
        cliente.close()
        return None
    return None if resposta.isError() else list(resposta.registers)


async def _escrita(funcao, inicio, endereco, quantidade, registradores, valores):
    if valores is not None:
        agora = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"{agora} escrita a partir do registrador {endereco}: {list(valores)}", flush=True)


async def servir(porta=PORTA):
    # O Windows aceita dois servidores na mesma porta; as escritas iriam para um deles.
    with socket.socket() as sonda:
        sonda.settimeout(0.5)
        if sonda.connect_ex((HOST, porta)) == 0:
            raise SystemExit(f"Já há um servidor em {HOST}:{porta}.")
    registradores = SimData(0, count=3, values=0, datatype=DataType.REGISTERS)
    servidor = ModbusTcpServer(
        SimDevice(1, simdata=[registradores], action=_escrita), address=(HOST, porta)
    )
    await servidor.serve_forever(background=True)
    print(f"CLP simulado em {HOST}:{porta}", flush=True)
    await asyncio.Event().wait()


if __name__ == "__main__":
    argumentos = argparse.ArgumentParser(description=__doc__)
    argumentos.add_argument("--porta", type=int, default=PORTA)
    try:
        asyncio.run(servir(argumentos.parse_args().porta))
    except KeyboardInterrupt:
        pass
