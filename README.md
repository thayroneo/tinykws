# VoiceTiny RP2040

**Reconhecimento de comandos de voz com TinyML em um microcontrolador de recursos limitados.**

O VoiceTiny RP2040 investiga como executar modelos compactos de aprendizado de máquina diretamente na **BitDogLab**, baseada no **Raspberry Pi Pico W (RP2040)**. A proposta é reconhecer comandos curtos, como *yes*, *no* e *stop*, sem depender de serviços na nuvem durante a inferência.

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

## Referências de base

- **Google Speech Commands:** conjunto público de dados para reconhecimento de palavras-chave.
- **A Microcontroller is All You Need: Enabling Transformer Execution on Low-Power IoT Endnodes:** referência sobre execução eficiente de Transformers em microcontroladores.
- **Trabalhos sobre compressão consciente de redes neurais (UFRN):** referências para estudar poda, quantização e sua ordem de aplicação.
