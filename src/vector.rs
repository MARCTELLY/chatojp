use candle_core::{IndexOp, Tensor};
use qdrant_client::client::{Payload, QdrantClient};
use crate::contents::File;
use crate::errors::EmbeddingError;
use anyhow::Result;
use candle_nn::Embedding;
use qdrant_client::qdrant::PointStruct;
use serde_json::json;

const COLLECTION: &str = "docs";


pub struct VectorDB {
    client: QdrantClient,
    id: u64
}

impl VectorDB {
    pub fn new(client: QdrantClient) -> Self {
        Self {client, id: 0}
    }

    pub async fn upsert_embedding(&mut self, embedding: Tensor, file: &File) -> Result<()> {
        let span = tracing::span!(tracing::Level::TRACE, "upsert_embedding method");
        let _enter = span.enter();
        let payload: Payload = json!({
            "id": file.path.clone(),
        })
            .try_into()
            .map_err(|_| EmbeddingError {})?;

        tracing::event!(tracing::Level::TRACE, "Embedded: {}", file.path);

        let vec = embedding.i((0, 0, ..)).unwrap().to_vec1::<f32>().unwrap();
        let points = vec![PointStruct::new(self.id, vec, payload)];
        self.client.upsert_points(COLLECTION, None, points, None).await?;
        self.id += 1;
        Ok(())
    }

}
