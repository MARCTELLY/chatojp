# Chatojp

Chatojp is a Rust-based web application that leverages the power of vector embeddings and the Qdrant vector database to embed and manage document files. The application is built using the Axum web framework, and includes features for embedding document content and storing the embeddings in a vector database for efficient retrieval.

## Features

- Embed document contents using a custom embedding function.
- Store and manage document embeddings in Qdrant vector database.
- Simple "Hello World!" endpoint for testing the web server.

## Getting Started

### Prerequisites

- Rust (latest stable version)
- Cargo (latest stable version)
- Qdrant (running instance with an API key)
- Tokio (for asynchronous runtime)
- Axum (for the web server)
- Tracing (for logging)

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/chatojp.git
    cd chatojp
    ```

2. **Build the project:**

    ```bash
    cargo build --release
    ```

3. **Run the project:**

    ```bash
    cargo run --release
    ```

## Usage

### Running the Server

By default, the server will start and listen on `127.0.0.1:3000`. You can test the server by navigating to `http://127.0.0.1:3000` in your web browser or using a tool like `curl`:

```bash
curl http://127.0.0.1:3000
```

This should return:

```plaintext
Hello World!
```

### Embedding Documentation

The `embed_documentation` function processes each file, generates embeddings using the `embed_file` function, and upserts the embeddings into the Qdrant vector database.

### Code Structure

- **contents**: Contains modules for handling file contents.
- **embedding**: Contains the embedding logic.
- **errors**: Error handling module.
- **vector**: Abstractions for interacting with the vector database.
- **model**: Definitions for data models used in the application.

### Main Components

- `AppState`: Holds the state of the application, including the list of files and the vector database instance.
- `embed_documentation`: Asynchronously processes files and stores their embeddings.
- `hello_world`: Simple test endpoint.
- `init_router`: Initializes the Axum router with the application state and routes.
- `make_client`: Creates a new Qdrant client.
- `main`: Entry point of the application, initializes logging, creates the application state, and starts the server.

## Configuration

### Environment Variables

- `QDRANT_API_KEY`: The API key for authenticating with the Qdrant instance.
- `QDRANT_URL`: The URL of the Qdrant instance.

These environment variables can be set in your shell or in a `.env` file in the root of your project (using `dotenv` crate is recommended for managing environment variables).

## Contributing

Contributions are welcome! Please fork the repository and open a pull request to contribute.

## License

This project is licensed under the MIT License.

## Contact

For any questions or suggestions, please open an issue on the GitHub repository or contact the maintainer at gabin.mberik@riives.com
