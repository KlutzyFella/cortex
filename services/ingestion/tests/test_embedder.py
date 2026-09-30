"""Unit tests for the vector embedder."""



class TestEmbedChunks:
    def test_single_chunk_returns_one_vector(self, embedding_model) -> None:
        from embedder import embed_chunks

        result = embed_chunks(embedding_model, ["Hello world"])
        assert len(result) == 1

    def test_vector_has_384_dimensions(self, embedding_model) -> None:
        from embedder import embed_chunks

        result = embed_chunks(embedding_model, ["Hello world"])
        assert len(result[0]) == 384

    def test_n_chunks_returns_n_vectors(self, embedding_model) -> None:
        from embedder import embed_chunks

        chunks = ["First chunk.", "Second chunk.", "Third chunk."]
        result = embed_chunks(embedding_model, chunks)
        assert len(result) == len(chunks)

    def test_each_vector_has_correct_dimension(self, embedding_model) -> None:
        from embedder import embed_chunks

        chunks = ["chunk one", "chunk two", "chunk three"]
        result = embed_chunks(embedding_model, chunks)
        for vec in result:
            assert len(vec) == 384

    def test_all_values_are_floats(self, embedding_model) -> None:
        from embedder import embed_chunks

        result = embed_chunks(embedding_model, ["test text"])
        for value in result[0]:
            assert isinstance(value, float)

    def test_empty_list_returns_empty_list(self, embedding_model) -> None:
        from embedder import embed_chunks

        result = embed_chunks(embedding_model, [])
        assert result == []

    def test_different_texts_produce_different_vectors(self, embedding_model) -> None:
        from embedder import embed_chunks

        results = embed_chunks(embedding_model, ["cat", "database engineering"])
        assert results[0] != results[1]
