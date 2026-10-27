SITE=config/sites/southfork.toml
STEPS="metrics psd background kurtosis interpulse report supplement"

for STEP in $STEPS; do
    uv run soundscape $STEP --site $SITE && \
    :
done
