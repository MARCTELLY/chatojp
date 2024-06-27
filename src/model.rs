//pub type  Embedding = Vec<f32>;
use candle_core::{D, DType, Result, Tensor};
use candle_nn::{Dropout, Embedding, Module, VarBuilder};
use serde::Deserialize;

pub const DTYPE: DType = DType::F32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum HiddenAct {
    Gelu,
    Relu,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
enum PositionEmbeddingType {
    #[default]
    Absolute,
}
#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    attention_probs_dropout_prob: f64,
    classifier_dropout: Option<f64>, //
    hidden_act: HiddenAct,           //
    hidden_dropout_prob: f64,        //
    hidden_size: usize,              //
    initializer_range: f64,          //
    intermediate_size: usize,        //
    layer_norm_eps: f64,             //
    max_position_embeddings: usize,  //
    model_type: Option<String>,      //
    num_attention_heads: usize,      //
    num_hidden_layers: usize,        //
    pad_token_id: usize,             //
    #[serde(default)]
    position_embedding_type: PositionEmbeddingType, //
    type_vocab_size: usize,          //
    #[serde(default)]
    use_cache: bool, //
    vocab_size: usize,               //
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            attention_probs_dropout_prob: 0.1,
            classifier_dropout: None,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: 0.1,
            hidden_size: 768,
            initializer_range: 0.02,
            intermediate_size: 3072,
            layer_norm_eps: 1e-12,
            max_position_embeddings: 514,
            model_type: Some("xlm-roberta".to_string()),
            num_attention_heads: 12,
            num_hidden_layers: 12,
            pad_token_id: 1,
            position_embedding_type: PositionEmbeddingType::Absolute,
            type_vocab_size: 1,
            use_cache: true,
            vocab_size: 250002,
        }
    }
}

#[derive(Debug)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
    span: tracing::Span,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "layer-norm");
        Self {
            weight,
            bias,
            eps,
            span,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let (_bsize, _seq_len, hidden_size) = x.dims3()?;
        let x = x.to_dtype(internal_dtype)?;
        let mean_x = (x.sum_keepdim(2)? / hidden_size as f64)?;
        let x = x.broadcast_sub(&mean_x)?;
        let norm_x = (x.sqr()?.sum_keepdim(2)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let x = x_normed
            .to_dtype(x_dtype)?
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)?;
        Ok(x)
    }
}

fn embedding(vocab_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get((vocab_size, hidden_size), "weight")?;
    Ok(Embedding::new(embeddings, hidden_size))
}

fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    let (weight, bias) = match (vb.get(size, "weight"), vb.get(size, "bias")) {
        (Ok(weight), Ok(bias)) => (weight, bias),
        (Err(err), _) | (_, Err(err)) => {
            if let (Ok(weight), Ok(bias)) = (vb.get(size, "bert.embeddings.LayerNorm.gamma"), vb.get(size, "bert.embeddings.LayerNorm.beta")) {
                (weight, bias)
            } else {
                return Err(err);
            }
        }
    };
    Ok(LayerNorm::new(weight, bias, eps))
}


pub struct EmbeddingBertModel {
    word_embeddings: Embedding,
    position_embeddings: Option<Embedding>,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    span: tracing::Span,
}

impl EmbeddingBertModel {
    pub async fn embed(&self, _text: &str) -> Result<Vec<Embedding>> {
        todo!()
    }
    pub fn load(vb: VarBuilder, config: &ModelConfig) -> Result<Self> {
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("bert.embeddings.word_embeddings"),
        )?;

        let position_embeddings = embedding(
            config.max_position_embeddings,
            config.hidden_size,
            vb.pp("bert.embeddings.position_embeddings"),
        )?;
        let token_type_embeddings = embedding(
            config.type_vocab_size,
            config.hidden_size,
            vb.pp("bert.embeddings.token_type_embeddings"),
        )?;


        let layer_norm = layer_norm(config.hidden_size, config.layer_norm_eps, vb)?;
        Ok(Self {
            word_embeddings,
            position_embeddings: Some(position_embeddings),
            token_type_embeddings,
            layer_norm,
            dropout: Dropout::new(config.hidden_dropout_prob as f32),
            span: tracing::span!(tracing::Level::TRACE, "embeddings"),
        })
    }

    pub(crate) fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (_bsize, seq_len) = input_ids.dims2()?;
        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;
        let mut embeddings = (&input_embeddings + token_type_embeddings)?;
        if let Some(position_embeddings) = &self.position_embeddings {
            // TODO: Proper absolute positions?
            let position_ids = (0..seq_len as u32).collect::<Vec<_>>();
            let position_ids = Tensor::new(&position_ids[..], input_ids.device())?;
            embeddings = embeddings.broadcast_add(&position_embeddings.forward(&position_ids)?)?
        }
        let embeddings = self.layer_norm.forward(&embeddings)?;
        let embeddings = self.dropout.forward(&embeddings, false)?;
        Ok(embeddings)
    }
}
