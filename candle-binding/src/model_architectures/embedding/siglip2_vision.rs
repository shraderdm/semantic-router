//! SigLIP2 vision encoder (V2 slot for `MultiModalEmbeddingModel`).
//!
//! This file is a **thin shim** around
//! `candle_transformers::models::siglip2_naflex::VisionModel`, David's already-
//! validated SigLIP2 implementation from candle PR #3510 (cosine 0.9996 vs the
//! HuggingFace `Siglip2VisionModel` PyTorch reference). The reference impl lives
//! at `~/cua-vsr-scoping/candle-cua-demo/candle-transformers/src/models/siglip2_naflex.rs`
//! and is wired into this crate via the `[patch."https://github.com/huggingface/candle"]`
//! block in `candle-binding/Cargo.toml`.
//!
//! ## Why the swap
//!
//! The earlier overnight version of this file (393 LOC) was a from-scratch port
//! of SigLIP2 patterns written before the Cargo `[patch]` was added. It compiled
//! and ran structurally but had no parity validation. Now that `siglip2_naflex`
//! is importable, V2 uses the reference impl directly.
//!
//! ## Fixed-resolution-through-naflex
//!
//! `siglip2_naflex::VisionModel` is the **NaFlex variant** (variable-resolution
//! native-aspect-ratio). For the v0.1 CUA demo we drive it at a fixed
//! `(image_size, image_size)` resolution by passing `target_uniform=Some((h_patches,
//! w_patches))` where `h_patches == w_patches == image_size / patch_size`. When
//! the target equals the base grid, the bilinear position-embedding resize at
//! `siglip2_naflex.rs:514` short-circuits to a no-op.
//!
//! ## Input transformation
//!
//! NaFlex expects pre-patchified `(B, num_patches, C*P*P)` input. The caller in
//! `multimodal_embedding.rs` still hands us conv-style `(B, C, H, W)` pixel
//! values, so this shim patchifies them inline. The transform is shape-only
//! (no learnable weights); it must produce patches in row-major (h, w) order to
//! match the order in which the base position embedding indexes them.
//!
//! ## Checkpoint compatibility
//!
//! The NaFlex VisionModel weight tree differs from the fixed-resolution v2
//! tree the previous shim targeted. The expected checkpoint switches from
//! `google/siglip2-base-patch16-256` to `google/siglip2-base-patch16-naflex`.
//! Key differences:
//! - `embeddings.patch_embedding` is a `Linear` (`[hidden, C*P*P]`) instead of
//!   a `Conv2d` (`[hidden, C, P, P]`).
//! - `embeddings.position_embedding` is `[num_patches, hidden]` (already that
//!   shape, but the previous fixed-res shim assumed it would be indexed by
//!   absolute patch id; naflex's resize-and-add path handles the same lookup
//!   identically when target == base).
//! - Encoder layers + post_layernorm + attentional-probe head have identical
//!   weight names and shapes; the difference is purely the embedding head.
//!
//! The HANDOVER doc (`~/cua-vsr-scoping/v0.1-build/HANDOVER.md` decisions 3 + 4)
//! captures this checkpoint switch.

use candle_core::{Result, Tensor};
use candle_nn::{Activation, VarBuilder};
use candle_transformers::models::siglip2_naflex::{VisionConfig, VisionModel};

use super::multimodal_embedding::MultiModalEmbeddingConfig;

/// V2 image encoder slot. Wraps the validated naflex reference impl.
#[derive(Clone)]
pub struct Siglip2VisionEncoder {
    inner: VisionModel,
    /// Patch grid width = height for fixed-resolution use. Cached so `forward`
    /// can drive `target_uniform` without re-deriving it from pixel shape.
    grid_size: usize,
    patch_size: usize,
}

impl Siglip2VisionEncoder {
    /// Build a `siglip2_naflex::VisionConfig` from the multimodal config and
    /// load the underlying naflex `VisionModel` from `vb`. Weight prefix follows
    /// the caller's existing convention: `vb` is rooted at
    /// `image_encoder.vision_encoder`.
    pub fn load(vb: VarBuilder, config: &MultiModalEmbeddingConfig) -> Result<Self> {
        let grid_size = config.image_size / config.image_patch_size;
        let num_patches = grid_size * grid_size;
        let inner_cfg = VisionConfig {
            hidden_size: config.image_hidden_size,
            intermediate_size: config.image_intermediate_size,
            num_hidden_layers: config.image_num_layers,
            num_attention_heads: config.image_num_heads,
            num_channels: 3,
            patch_size: config.image_patch_size,
            num_patches,
            hidden_act: Activation::GeluPytorchTanh,
            layer_norm_eps: 1e-6,
        };
        let inner = VisionModel::new(&inner_cfg, vb)?;
        Ok(Self {
            inner,
            grid_size,
            patch_size: config.image_patch_size,
        })
    }

    /// Returns pooler output `(batch, image_hidden_size)`.
    ///
    /// `pixel_values` is `(B, 3, H, W)` with `H == W == grid_size * patch_size`.
    /// This shim patchifies in-place to `(B, num_patches, 3*P*P)` then dispatches
    /// to `naflex::VisionModel::forward` in uniform mode.
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let patches = patchify(pixel_values, self.patch_size)?;
        self.inner.forward(
            &patches,
            None,                                   // pixel_attention_mask: all patches are real
            None,                                   // spatial_shapes: uniform mode
            Some((self.grid_size, self.grid_size)), // target_uniform
        )
    }
}

/// Convert `(B, C, H, W)` to `(B, num_patches, C*P*P)` in row-major patch order.
///
/// Matches the layout `siglip2_naflex::VisionEmbeddings::forward_uniform` expects:
/// row 0 of the position grid corresponds to patches `[0..grid_w]`, row 1 to
/// `[grid_w..2*grid_w]`, and so on.
fn patchify(pixel_values: &Tensor, patch_size: usize) -> Result<Tensor> {
    let (b, c, h, w) = pixel_values.dims4()?;
    if h % patch_size != 0 || w % patch_size != 0 {
        candle_core::bail!(
            "patchify: spatial dims ({h}, {w}) must be divisible by patch_size {patch_size}"
        );
    }
    let h_p = h / patch_size;
    let w_p = w / patch_size;
    // (B, C, h_p, P, w_p, P) -> permute to (B, h_p, w_p, C, P, P) -> reshape.
    pixel_values
        .reshape((b, c, h_p, patch_size, w_p, patch_size))?
        .permute((0, 2, 4, 1, 3, 5))?
        .contiguous()?
        .reshape((b, h_p * w_p, c * patch_size * patch_size))
}

// Sanity-check construction + forward of the encoder from random weights. This
// validates the patchify transform, the VisionConfig mapping, and that the
// naflex VisionModel accepts the inputs the shim produces. Numeric parity is
// inherited from the underlying impl (cosine 0.9996 vs PyTorch on the upstream
// reference test) and not retested here.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_architectures::embedding::multimodal_embedding::VisionVariant;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    fn small_config() -> MultiModalEmbeddingConfig {
        MultiModalEmbeddingConfig {
            embedding_dim: 384,
            text_hidden_size: 384,
            text_num_layers: 6,
            text_num_heads: 12,
            text_intermediate_size: 1536,
            text_vocab_size: 30522,
            text_max_position_embeddings: 512,
            text_type_vocab_size: 2,
            text_layer_norm_eps: 1e-12,
            // SigLIP2-base-patch16-naflex dimensions (base grid 16x16 = 256 patches).
            image_hidden_size: 768,
            image_patch_size: 16,
            image_size: 256,
            image_num_layers: 2, // shrink for the test
            image_num_heads: 12,
            image_intermediate_size: 3072,
            audio_hidden_size: 384,
            audio_num_layers: 4,
            audio_num_heads: 6,
            audio_num_mel_bins: 80,
            audio_max_source_positions: 1500,
            matryoshka_dims: vec![384, 256, 128, 64, 32],
            vision_variant: VisionVariant::V2,
        }
    }

    #[test]
    fn encoder_forward_shape() -> Result<()> {
        let device = Device::Cpu;
        let cfg = small_config();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let encoder = Siglip2VisionEncoder::load(vb, &cfg)?;
        let pixels = Tensor::zeros((2, 3, cfg.image_size, cfg.image_size), DType::F32, &device)?;
        let out = encoder.forward(&pixels)?;
        assert_eq!(out.dims(), &[2, cfg.image_hidden_size]);
        Ok(())
    }

    #[test]
    fn patchify_layout_is_row_major() -> Result<()> {
        // Build a (1, 1, 4, 4) tensor with row-major values 0..16; patchify with
        // patch_size=2 should give 4 patches of length 4 in row-major order:
        // patch (0,0): [0, 1, 4, 5]
        // patch (0,1): [2, 3, 6, 7]
        // patch (1,0): [8, 9, 12, 13]
        // patch (1,1): [10, 11, 14, 15]
        let device = Device::Cpu;
        let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let t = Tensor::from_vec(data, (1, 1, 4, 4), &device)?;
        let p = patchify(&t, 2)?;
        let vals = p.to_vec3::<f32>()?;
        assert_eq!(vals[0][0], vec![0.0, 1.0, 4.0, 5.0]);
        assert_eq!(vals[0][1], vec![2.0, 3.0, 6.0, 7.0]);
        assert_eq!(vals[0][2], vec![8.0, 9.0, 12.0, 13.0]);
        assert_eq!(vals[0][3], vec![10.0, 11.0, 14.0, 15.0]);
        Ok(())
    }
}
