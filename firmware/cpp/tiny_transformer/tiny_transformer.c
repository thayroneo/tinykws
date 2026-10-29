#include "tiny_transformer.h"

#include <math.h>

static float linear_row(const float *weight, const float *bias, size_t row,
                        const float *input, size_t input_dim) {
    float value = bias[row];
    const float *weights = weight + row * input_dim;
    for (size_t col = 0; col < input_dim; ++col) value += weights[col] * input[col];
    return value;
}

static void layer_norm(float *data, const float *weight, const float *bias) {
    for (size_t token = 0; token < TT_TOKENS; ++token) {
        float *row = data + token * TT_MODEL;
        float mean = 0.0f;
        for (size_t col = 0; col < TT_MODEL; ++col) mean += row[col];
        mean /= (float)TT_MODEL;
        float variance = 0.0f;
        for (size_t col = 0; col < TT_MODEL; ++col) {
            float delta = row[col] - mean;
            variance += delta * delta;
        }
        variance /= (float)TT_MODEL;
        float inverse_std = 1.0f / sqrtf(variance + 1.0e-5f);
        for (size_t col = 0; col < TT_MODEL; ++col)
            row[col] = (row[col] - mean) * inverse_std * weight[col] + bias[col];
    }
}

size_t tiny_transformer_workspace_bytes(void) {
    return sizeof(TinyTransformerWorkspace);
}

void tiny_transformer_infer(const float input[TT_TOKENS][TT_INPUT],
                            const TinyTransformerWeights *weights,
                            TinyTransformerWorkspace *workspace,
                            float logits[TT_CLASSES]) {
    for (size_t token = 0; token < TT_TOKENS; ++token) {
        for (size_t col = 0; col < TT_MODEL; ++col) {
            workspace->h[token * TT_MODEL + col] =
                linear_row(weights->input_projection_weight,
                           weights->input_projection_bias, col, input[token], TT_INPUT) +
                weights->position[token * TT_MODEL + col];
        }
    }

    for (size_t token = 0; token < TT_TOKENS; ++token) {
        const float *row = workspace->h + token * TT_MODEL;
        for (size_t component = 0; component < 3 * TT_MODEL; ++component) {
            float value = linear_row(weights->attention_in_proj_weight,
                                     weights->attention_in_proj_bias, component,
                                     row, TT_MODEL);
            float *target = component < TT_MODEL ? workspace->q :
                            (component < 2 * TT_MODEL ? workspace->k : workspace->v);
            size_t offset = component % TT_MODEL;
            target[token * TT_MODEL + offset] = value;
        }
    }

    for (size_t head = 0; head < TT_HEADS; ++head) {
        for (size_t query = 0; query < TT_TOKENS; ++query) {
            float maximum = -INFINITY;
            for (size_t key = 0; key < TT_TOKENS; ++key) {
                float score = 0.0f;
                for (size_t col = 0; col < TT_HEAD_DIM; ++col)
                    score += workspace->q[query * TT_MODEL + head * TT_HEAD_DIM + col] *
                             workspace->k[key * TT_MODEL + head * TT_HEAD_DIM + col];
                score /= sqrtf((float)TT_HEAD_DIM);
                workspace->scores[(head * TT_TOKENS + query) * TT_TOKENS + key] = score;
                if (score > maximum) maximum = score;
            }
            float denominator = 0.0f;
            for (size_t key = 0; key < TT_TOKENS; ++key) {
                float *score = &workspace->scores[(head * TT_TOKENS + query) * TT_TOKENS + key];
                *score = expf(*score - maximum);
                denominator += *score;
            }
            for (size_t key = 0; key < TT_TOKENS; ++key)
                workspace->scores[(head * TT_TOKENS + query) * TT_TOKENS + key] /= denominator;
        }
    }

    for (size_t token = 0; token < TT_TOKENS; ++token) {
        for (size_t col = 0; col < TT_MODEL; ++col) {
            size_t head = col / TT_HEAD_DIM;
            size_t head_col = col % TT_HEAD_DIM;
            float value = 0.0f;
            for (size_t key = 0; key < TT_TOKENS; ++key)
                value += workspace->scores[(head * TT_TOKENS + token) * TT_TOKENS + key] *
                         workspace->v[key * TT_MODEL + head * TT_HEAD_DIM + head_col];
            workspace->attended[token * TT_MODEL + col] = value;
        }
    }

    for (size_t token = 0; token < TT_TOKENS; ++token) {
        float projected[TT_MODEL];
        for (size_t col = 0; col < TT_MODEL; ++col)
            projected[col] = linear_row(weights->attention_out_proj_weight,
                                        weights->attention_out_proj_bias, col,
                                        workspace->attended + token * TT_MODEL, TT_MODEL);
        for (size_t col = 0; col < TT_MODEL; ++col)
            workspace->h[token * TT_MODEL + col] += projected[col];
    }
    layer_norm(workspace->h, weights->norm1_weight, weights->norm1_bias);

    for (size_t token = 0; token < TT_TOKENS; ++token) {
        for (size_t row = 0; row < TT_FFN; ++row) {
            float value = linear_row(weights->ffn0_weight, weights->ffn0_bias, row,
                                     workspace->h + token * TT_MODEL, TT_MODEL);
            workspace->ffn[token * TT_FFN + row] = value > 0.0f ? value : 0.0f;
        }
        float projected[TT_MODEL];
        for (size_t col = 0; col < TT_MODEL; ++col)
            projected[col] = linear_row(weights->ffn2_weight, weights->ffn2_bias, col,
                                        workspace->ffn + token * TT_FFN, TT_FFN);
        for (size_t col = 0; col < TT_MODEL; ++col)
            workspace->h[token * TT_MODEL + col] += projected[col];
    }
    layer_norm(workspace->h, weights->norm2_weight, weights->norm2_bias);

    float pooled[TT_MODEL] = {0};
    for (size_t token = 0; token < TT_TOKENS; ++token)
        for (size_t col = 0; col < TT_MODEL; ++col)
            pooled[col] += workspace->h[token * TT_MODEL + col] / (float)TT_TOKENS;
    for (size_t row = 0; row < TT_CLASSES; ++row)
        logits[row] = linear_row(weights->classifier_weight, weights->classifier_bias,
                                 row, pooled, TT_MODEL);
}
