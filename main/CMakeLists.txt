idf_component_register(
    SRCS "main.c" "inference.c"
    INCLUDE_DIRS "."
    EMBED_FILES "stories1M.bin" "tokenizer.bin"
)