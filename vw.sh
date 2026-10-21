uv run soundscape metrics --site config/sites/vineyard.toml && \
    uv run soundscape ambient --site config/sites/vineyard.toml && \
    uv run soundscape kurtosis --site config/sites/vineyard.toml && \
    uv run soundscape report --site config/sites/vineyard.toml
