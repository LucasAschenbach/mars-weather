# Forecast Data

Keep raw model outputs out of `public/` and publish browser-ready frames into `public/forecasts/`.
The app can load either a single legacy forecast at `public/forecasts/latest/` or a multi-forecast
catalog at `public/forecasts/catalog.json`.

Recommended layout:

```text
mas_weather_app/
  data/
    forecasts/
      raw/
        latest.nc
  public/
    forecasts/
      catalog.json
      pretrained/
        manifest.json
        frames/
          lead-000.i16
          lead-002.i16
      random-init/
        manifest.json
        frames/
          lead-000.i16
          lead-002.i16
      ground-truth/
        manifest.json
        frames/
          lead-000.i16
          lead-002.i16
```

Export app-readable files from NetCDF rollouts:

```bash
python scripts/export_forecast.py ../artifacts/app_rollouts/pretrained_7day_rollout.nc \
  --output public/forecasts/pretrained \
  --max-lead-hours 168 \
  --source "Pretrained Aurora"
```

The catalog lists each selectable forecast:

```json
{
  "schema": "mars-weather-forecast-catalog-v1",
  "forecasts": [
    {
      "id": "pretrained",
      "label": "Pretrained",
      "manifestPath": "/forecasts/pretrained/manifest.json"
    }
  ]
}
```

The app first tries to load:

```text
/forecasts/catalog.json
```

If the catalog is missing, it falls back to the legacy single-forecast path:

```text
/forecasts/latest/manifest.json
```

If no forecast manifest or frame can be loaded, it falls back to the deterministic preview data.

The exported binary frame order is:

```text
ps, tsurf, co2ice, dustcol, u, v, temp
```

Surface variables are stored as `lat, lon`; atmospheric variables are stored as `lev, lat, lon`.
Values are quantized as signed 16-bit integers. The manifest stores each variable's `min`
and `scale` for decoding in the browser.

For aligned model-vs-truth comparisons, the exported manifests should share the same `baseTime`,
`solarLongitude`, `marsYear`, and lead-hour frame sequence. The current 7-day app rollouts use
85 frames: lead 0 plus +2h through +168h.
