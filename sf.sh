SITE=config/sites/southfork.toml
STEPS="metrics psd background kurtosis interpulse report supplement"

uv run soundscape detect --reselect --site $SITE

for STEP in $STEPS; do
    uv run soundscape $STEP --site $SITE && \
    :
done
