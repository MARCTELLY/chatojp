use std::sync::Arc;
use axum::Router;
use axum::routing::get;
use qdrant_client::client::QdrantClient;
use safetensors::View;

use crate::contents::File;
use crate::embedding::embed_file;
use crate::vector::VectorDB;

mod contents;
mod vector;
mod errors;
mod embedding;
mod model;

struct AppState {
    files: Vec<File>,
    vector_db: VectorDB,
}

async fn embed_documentation(vector_db: &mut VectorDB, files: &Vec<File>) -> anyhow::Result<()> {
    for file in files {
        let embeddings = embed_file(file).await?;
        tracing::event!(tracing::Level::TRACE, "Embedding: {:?}", file.path);
        vector_db.upsert_embedding(embeddings, file).await?;
    }
    Ok(())
}

async fn hello_world() -> &'static str {
    "Hello World!"
}

fn init_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(hello_world))
        .with_state(state)
}

async fn make_client() -> anyhow::Result<QdrantClient> {
    let client = QdrantClient::from_url("https://addeb953-f2f0-4113-af5b-27c0145484f0.us-east4-0.gcp.cloud.qdrant.io:6333")
        .with_api_key("Qb2MVEtWOYZ4Hi876mx4P2ztn_i8JYuGAnyv3c_5l7slLceBzy5BvA")
        .build().unwrap();
    Ok(client)
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    // create Qdrant client
    let q_client = make_client().await.unwrap();

    // create the app state
    let shared_state = Arc::new(AppState {
        files: vec![],
        vector_db: VectorDB::new(q_client),
    });

    let app = init_router(shared_state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .unwrap();

    axum::serve(listener, app).await.unwrap()
}
