cmake_minimum_required(VERSION 3.13)

include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)

project(tinykws_adc_dma_capture C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

add_executable(adc_dma_capture
    main.c
)

target_link_libraries(adc_dma_capture
    pico_stdlib
    hardware_adc
    hardware_dma
    hardware_clocks
)

pico_enable_stdio_usb(adc_dma_capture 1)
pico_enable_stdio_uart(adc_dma_capture 0)
pico_add_extra_outputs(adc_dma_capture)
