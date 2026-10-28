# TinyKWS Lab

**Reconhecimento de comandos de voz com TinyML em um microcontrolador de recursos limitados.**

O TinyKWS Lab investiga como executar modelos compactos de aprendizado de máquina diretamente na **BitDogLab**, baseada no **Raspberry Pi Pico W (RP2040)**. A proposta é reconhecer comandos curtos, como *yes*, *no* e *stop*, sem depender de serviços na nuvem durante a inferência.

## Pergunta de pesquisa

**Como a arquitetura da rede neural, a ordem de aplicação de poda e quantização e a otimização dos kernels computacionais influenciam o equilíbrio entre acurácia, uso de memória e latência de inferência no RP2040?**

## Abordagem

O experimento compara duas famílias de modelos — uma **MLP** compacta e um **TinyTransformer** — sob diferentes estratégias de compressão: modelo de referência, quantização INT8, poda seguida de quantização e quantização seguida de poda. Também avalia implementações convencionais e otimizadas de operações computacionais, respeitando as limitações do Cortex-M0+.

Os modelos são desenvolvidos a partir do **Google Speech Commands**, com extração de características de áudio por **MFCC**. A avaliação considera tanto dados de referência quanto gravações capturadas pelo microfone da BitDogLab.

## Pipeline geral

**Áudio → Pré-processamento e MFCC → Treinamento (MLP / TinyTransformer) → Compressão → Inferência no RP2040 → Avaliação de acurácia, memória e latência**

## Resultado esperado

Uma demonstração funcional de reconhecimento local de comandos de voz, acompanhada de uma comparação experimental reproduzível que mostre os compromissos entre **qualidade de classificação**, **tamanho dos modelos**, **consumo de memória** e **tempo de resposta**.

Os resultados serão medidos experimentalmente; eventuais ganhos de compressão ou otimização não são pressupostos. O projeto também distingue técnicas descritas na literatura de adaptações específicas para o RP2040.

## Tecnologias e plataforma

Python para preparação de dados e treinamento; MicroPython e C/C++ para prototipagem e inferência embarcada; BitDogLab com Raspberry Pi Pico W (RP2040) como plataforma de avaliação.

## Hardware de avaliação

O hardware-alvo é o microcontrolador **RP2040**, usado na BitDogLab/Raspberry Pi Pico W:

- **Processador:** dual-core ARM Cortex-M0+ operando a até 133 MHz ([datasheet do RP2040](https://pip.raspberrypi.com/documents/RP-008371-DS-rp2040-datasheet.pdf), [referência da placa](https://www.eletrogate.com/placa-de-desenvolvimento-rp2040)).
- **Memória RAM:** 264 KB de SRAM, dividida em seis bancos ([datasheet do RP2040](https://pip.raspberrypi.com/documents/RP-008371-DS-rp2040-datasheet.pdf), [visão geral](https://en.wikipedia.org/wiki/RP2040)).
- **Memória Flash:** o RP2040 não possui Flash interna; o armazenamento de programa usa memória externa via QSPI. A capacidade da placa depende do módulo utilizado, geralmente entre 2 MB e 16 MB ([referência da placa](https://www.eletrogate.com/placa-de-desenvolvimento-rp2040), [exemplo de descrição](https://www.youtube.com/watch?v=cAcFSxFx5DU)).
- **Periféricos:** 30 pinos GPIO, sendo 26 multifuncionais; 2 SPI; 2 I²C; 2 UART; 3 entradas ADC externas de 12 bits; 16 canais PWM; USB 1.1 com funções host/device; e máquinas de estados de E/S programáveis (PIO) ([datasheet do RP2040](https://pip.raspberrypi.com/documents/RP-008371-DS-rp2040-datasheet.pdf), [referência da placa](https://www.eletrogate.com/placa-de-desenvolvimento-rp2040), [visão geral](https://en.wikipedia.org/wiki/RP2040)).

As medições de latência, SRAM, Flash e comportamento dos modelos serão feitas na placa. Estimativas produzidas no host não serão apresentadas como medições do RP2040.

## Referências de base

- **Google Speech Commands:** conjunto público de dados para reconhecimento de palavras-chave.
- **A Microcontroller is All You Need: Enabling Transformer Execution on Low-Power IoT Endnodes:** referência sobre execução eficiente de Transformers em microcontroladores.
- **Trabalhos sobre compressão consciente de redes neurais (UFRN):** referências para estudar poda, quantização e sua ordem de aplicação.
