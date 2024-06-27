use candle_core::Tensor;
use std::path::PathBuf;

use crate::{
    contents::File,
    model::{EmbeddingBertModel, ModelConfig, DTYPE},
};
use anyhow::{Error, Result};
use candle_core::Device;
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};

pub fn get_model_path(model: String) -> (PathBuf, PathBuf, PathBuf) {
    let api = Api::new().unwrap();
    let repo = api.model(model);
    (
        repo.get("model.safetensors").unwrap_or(PathBuf::default()),
        repo.get("tokenizer.json").unwrap_or(PathBuf::default()),
        repo.get("config.json").unwrap_or(PathBuf::default()),
    )
}

/// Get the bert model for embedding
pub fn get_embedding_model_and_tokenizer() -> Result<(EmbeddingBertModel, Tokenizer)> {
    let (weights_path, tokenizer_path, config_path) =
        get_model_path("google-bert/bert-base-multilingual-cased".to_string());

    // config
    let config = std::fs::read_to_string(config_path)?;
    let config: ModelConfig = serde_json::from_str(&config)?;

    // tokenizer
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(Error::msg)?;

    // weights
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &Device::Cpu)?
    };
    let model = EmbeddingBertModel::load(vb, &config)?;
    Ok((model, tokenizer))
}

pub fn embed_sentence(prompt: &str) -> Result<Tensor> {
    let span = tracing::span!(tracing::Level::TRACE, "sentence");
    let (model, mut tokenizer) = get_embedding_model_and_tokenizer().unwrap();
    let tokenizer = tokenizer.with_padding(None).with_truncation(None).unwrap();
    let tokens = tokenizer
        .encode(prompt, true)
        .map_err(Error::msg)
        .unwrap()
        .get_ids()
        .to_vec();
    let token_ids = Tensor::new(&tokens[..], &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let token_type_ids = token_ids.zeros_like().unwrap();

    let embeddings = model.forward(&token_ids, &token_type_ids)?;
    tracing::event!(tracing::Level::TRACE, "generated embeddings {:?}", embeddings.shape());
    Ok(embeddings)
}

pub async fn embed_file(file: &File) -> Result<Tensor> {
    let span = tracing::span!(tracing::Level::TRACE, "embedding files");
    let _enter = span.enter();
    let sentences_as_str : Vec<&str> = file.sentences.iter().map(|s| s.as_str()).collect();
    let pp = PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        ..Default::default()
    };
    let (model, mut tokenizer) = get_embedding_model_and_tokenizer().unwrap();
    tokenizer.with_padding(Some(pp));
    let tokens = tokenizer
        .encode_batch(sentences_as_str.to_vec(), true)
        .map_err(Error::msg)?;
    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Ok(Tensor::new(tokens.as_slice(), &Device::Cpu)?)
        })
        .collect::<Result<Vec<_>>>()?;
    let token_ids = Tensor::stack(&token_ids, 0)?;
    let token_type_ids= token_ids.zeros_like()?;
    tracing::event!(tracing::Level::TRACE, "running embedding on {:?}", token_ids.shape());
    let embeddings = model.forward(&token_ids, &token_type_ids)?;
    tracing::event!(tracing::Level::TRACE, "generated embeddings {:?}", embeddings.shape());
    Ok(embeddings)
}

#[cfg(test)]
mod tests {
    use candle_core::IndexOp;
    use crate::embedding::embed_sentence;

    #[test]
    fn test_emb(){
        let emb = embed_sentence("Je suis moi").unwrap();
        println!(" {emb}");
        let vec = emb.i((0, 0, ..)).unwrap();

        println!(" {:#?}", vec.to_vec1::<f32>())
    }
}