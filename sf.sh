uv run soundscape metrics --site config/sites/southfork.toml && \
    uv run soundscape ambient --site config/sites/southfork.toml && \
    uv run soundscape kurtosis --site config/sites/southfork.toml && \
    uv run soundscape report --site config/sites/southfork.toml
