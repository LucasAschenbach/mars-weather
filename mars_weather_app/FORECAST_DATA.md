# Forecast Data

Keep raw model outputs out of `public/` and publish browser-ready frames into `public/forecasts/latest/`.

Recommended layout:

```text
mas_weather_app/
  data/
    forecasts/
      raw/
        latest.nc
  public/
    forecasts/
      latest/
        manifest.json
        frames/
          lead-000.i16
          lead-002.i16
          lead-004.i16
```

Put the final Aurora/OpenMARS NetCDF prediction at:

```text
mas_weather_app/data/forecasts/raw/latest.nc
```

Then export the app-readable files from the app directory:

```bash
python scripts/export_forecast.py data/forecasts/raw/latest.nc
```

The app automatically tries to load:

```text
/forecasts/latest/manifest.json
```

If that file is missing or a frame fails to load, it falls back to the deterministic preview data.

The exported binary frame order is:

```text
ps, tsurf, co2ice, dustcol, u, v, temp
```

Surface variables are stored as `lat, lon`; atmospheric variables are stored as `lev, lat, lon`.
Values are quantized as signed 16-bit integers. The manifest stores each variable's `min`
and `scale` for decoding in the browser.
