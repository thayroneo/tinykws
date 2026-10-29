#ifndef TINY_TRANSFORMER_H
#define TINY_TRANSFORMER_H

#include <stddef.h>

#define TT_TOKENS 40
#define TT_INPUT 13
#define TT_MODEL 32
#define TT_HEADS 2
#define TT_HEAD_DIM 16
#define TT_FFN 64
#define TT_CLASSES 5

typedef struct {
    const float *input_projection_weight;
    const float *input_projection_bias;
    const float *position;
    const float *attention_in_proj_weight;
    const float *attention_in_proj_bias;
    const float *attention_out_proj_weight;
    const float *attention_out_proj_bias;
    const float *norm1_weight;
    const float *norm1_bias;
    const float *ffn0_weight;
    const float *ffn0_bias;
    const float *ffn2_weight;
    const float *ffn2_bias;
    const float *norm2_weight;
    const float *norm2_bias;
    const float *classifier_weight;
    const float *classifier_bias;
} TinyTransformerWeights;

typedef struct {
    float h[TT_TOKENS * TT_MODEL];
    float q[TT_TOKENS * TT_MODEL];
    float k[TT_TOKENS * TT_MODEL];
    float v[TT_TOKENS * TT_MODEL];
    float attended[TT_TOKENS * TT_MODEL];
    float scores[TT_HEADS * TT_TOKENS * TT_TOKENS];
    float ffn[TT_TOKENS * TT_FFN];
} TinyTransformerWorkspace;

size_t tiny_transformer_workspace_bytes(void);
void tiny_transformer_infer(const float input[TT_TOKENS][TT_INPUT],
                            const TinyTransformerWeights *weights,
                            TinyTransformerWorkspace *workspace,
                            float logits[TT_CLASSES]);

#endif
