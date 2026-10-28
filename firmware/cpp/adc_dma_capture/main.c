#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hardware/adc.h"
#include "hardware/dma.h"
#include "pico/stdlib.h"

#define ADC_GPIO 28
#define ADC_INPUT 2
#define DEFAULT_RATE_HZ 16000u
#define DEFAULT_DURATION_MS 1000u
#define MAX_SAMPLES 16000u
#define ADC_CLOCK_HZ 48000000.0f

static uint16_t samples[MAX_SAMPLES];

static void configure_adc_dma(uint sample_rate_hz, int dma_channel, dma_channel_config *config) {
    adc_init();
    adc_gpio_init(ADC_GPIO);
    adc_select_input(ADC_INPUT);
    adc_fifo_setup(
        true,  /* write each conversion to the FIFO */
        true,  /* enable DMA data request */
        1,     /* request when at least one sample is present */
        false, /* no error bit in the FIFO value */
        false  /* retain the 12-bit value in the uint16_t container */
    );
    adc_set_clkdiv((ADC_CLOCK_HZ / (float)sample_rate_hz) - 1.0f);
    adc_fifo_drain();

    *config = dma_channel_get_default_config((uint)dma_channel);
    channel_config_set_transfer_data_size(config, DMA_SIZE_16);
    channel_config_set_read_increment(config, false);
    channel_config_set_write_increment(config, true);
    channel_config_set_dreq(config, DREQ_ADC);
}

static size_t capture(uint sample_rate_hz, uint duration_ms, uint64_t *elapsed_us) {
    size_t sample_count = ((size_t)sample_rate_hz * duration_ms) / 1000u;
    if (sample_count == 0 || sample_count > MAX_SAMPLES) {
        return 0;
    }

    int dma_channel = dma_claim_unused_channel(true);
    dma_channel_config config;
    configure_adc_dma(sample_rate_hz, dma_channel, &config);
    dma_channel_configure(
        (uint)dma_channel,
        &config,
        samples,
        &adc_hw->fifo,
        sample_count,
        false
    );

    uint64_t start_us = time_us_64();
    adc_run(true);
    dma_start_channel_mask(1u << (uint)dma_channel);
    dma_channel_wait_for_finish_blocking((uint)dma_channel);
    uint64_t end_us = time_us_64();
    adc_run(false);
    adc_fifo_drain();
    dma_channel_unclaim((uint)dma_channel);
    *elapsed_us = end_us - start_us;
    return sample_count;
}

static void handle_capture(uint rate_hz, uint duration_ms) {
    uint64_t elapsed_us = 0;
    size_t sample_count = capture(rate_hz, duration_ms, &elapsed_us);
    if (sample_count == 0) {
        printf("VTKWS_ERROR invalid_capture_parameters\n");
        fflush(stdout);
        return;
    }
    double effective_rate = 0.0;
    if (sample_count > 1 && elapsed_us > 0) {
        effective_rate = ((double)(sample_count - 1) * 1000000.0) / (double)elapsed_us;
    }
    printf(
        "VTKWS_READY {\"sample_count\":%u,\"requested_rate_hz\":%u,"
        "\"duration_ms\":%u,\"adc_gpio\":%u,\"adc_channel\":%u,"
        "\"adc_bits\":12,\"elapsed_us\":%llu,\"effective_rate_hz\":%.6f,"
        "\"sample_format\":\"uint16_le_adc12\"}\n",
        (unsigned)sample_count,
        (unsigned)rate_hz,
        (unsigned)duration_ms,
        ADC_GPIO,
        ADC_INPUT,
        (unsigned long long)elapsed_us,
        effective_rate
    );
    fflush(stdout);
    fwrite(samples, sizeof(uint16_t), sample_count, stdout);
    fflush(stdout);
    printf("VTKWS_DONE\n");
    fflush(stdout);
}

int main(void) {
    stdio_init_all();
    sleep_ms(1500);
    char command[64];
    size_t length = 0;
    printf("VTKWS_ADC_DMA_READY\n");
    fflush(stdout);
    while (true) {
        int character = getchar_timeout_us(1000);
        if (character < 0) {
            continue;
        }
        if (character == '\n' || character == '\r') {
            command[length] = '\0';
            if (strncmp(command, "CAPTURE", 7) == 0) {
                unsigned rate = DEFAULT_RATE_HZ;
                unsigned duration = DEFAULT_DURATION_MS;
                (void)sscanf(command + 7, "%u %u", &rate, &duration);
                handle_capture(rate, duration);
            } else if (strcmp(command, "PING") == 0) {
                printf("VTKWS_PONG\n");
                fflush(stdout);
            } else if (length > 0) {
                printf("VTKWS_ERROR unknown_command\n");
                fflush(stdout);
            }
            length = 0;
        } else if (length + 1 < sizeof(command)) {
            command[length++] = (char)character;
        } else {
            length = 0;
        }
    }
}
