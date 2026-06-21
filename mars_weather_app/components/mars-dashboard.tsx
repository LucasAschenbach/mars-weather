"use client"

import { useEffect, useMemo, useState } from "react"
import {
  frameForLead,
  loadForecastFrame,
  loadForecastManifest,
  pointAt,
  type ForecastFrame,
  type ForecastManifest,
} from "@/lib/forecast-file"
import {
  ATMOS_LEVEL_COUNT,
  co2Ice,
  dustcol,
  forecastDate,
  LAYERS,
  ps,
  temp,
  tsurf,
  windU,
  windV,
  type Layer,
} from "@/lib/mars-data"
import { ForecastHeader } from "@/components/forecast-header"
import { LayerControls, MapLegend } from "@/components/layer-controls"
import { LeadTimeSlider } from "@/components/lead-time-slider"
import { MarsMap, type GeoPoint } from "@/components/mars-map"
import { MetricCard } from "@/components/metric-card"
import { VerticalLevelSelector } from "@/components/vertical-level-selector"

function fmtCoord(p: GeoPoint): string {
  const ns = p.lat >= 0 ? "N" : "S"
  const ew = p.lon >= 0 ? "E" : "W"
  return `${Math.abs(p.lat).toFixed(0)}°${ns} ${Math.abs(p.lon).toFixed(0)}°${ew}`
}

export function MarsDashboard() {
  const [layer, setLayer] = useState<Layer>("tsurf")
  const [leadHours, setLeadHours] = useState(48)
  const [level, setLevel] = useState(0)
  const [selected, setSelected] = useState<GeoPoint>({ lon: 0, lat: 0 })
  const [forecastManifest, setForecastManifest] = useState<ForecastManifest | null>(null)
  const [forecastFrame, setForecastFrame] = useState<ForecastFrame | null>(null)

  useEffect(() => {
    let cancelled = false
    loadForecastManifest()
      .then((manifest) => {
        if (!cancelled) setForecastManifest(manifest)
      })
      .catch(() => {
        if (!cancelled) setForecastManifest(null)
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!forecastManifest) {
      return
    }

    let cancelled = false
    const ref = frameForLead(forecastManifest, leadHours)
    loadForecastFrame(forecastManifest, ref)
      .then((frame) => {
        if (!cancelled) setForecastFrame(frame)
      })
      .catch(() => {
        if (!cancelled) setForecastFrame(null)
      })
    return () => {
      cancelled = true
    }
  }, [forecastManifest, leadHours])

  const m = useMemo(() => {
    const { lon, lat } = selected
    if (forecastManifest && forecastFrame) {
      const point = pointAt(forecastManifest, forecastFrame, lon, lat, level)
      return {
        tsurf: point.tsurf,
        ps: point.ps,
        dustcol: point.dustcol,
        ice: point.co2ice,
        temp: point.temp,
        u: point.u,
        v: point.v,
      }
    }

    return {
      tsurf: tsurf(lon, lat, leadHours),
      ps: ps(lon, lat, leadHours),
      dustcol: dustcol(lon, lat, leadHours),
      ice: co2Ice(lon, lat, leadHours),
      temp: temp(lon, lat, leadHours, level),
      u: windU(lon, lat, leadHours, level),
      v: windV(lon, lat, leadHours, level),
    }
  }, [forecastFrame, forecastManifest, selected, leadHours, level])

  const windSpeed = Math.sqrt(m.u * m.u + m.v * m.v)
  const windDir = ((Math.atan2(-m.u, -m.v) * 180) / Math.PI + 360) % 360
  const levelLabel = `L${level + 1}/${ATMOS_LEVEL_COUNT}`

  return (
    <main className="relative h-dvh w-full overflow-hidden bg-background">
      <MarsMap
        layer={layer}
        leadHours={leadHours}
        selected={selected}
        forecast={
          forecastManifest && forecastFrame
            ? { manifest: forecastManifest, frame: forecastFrame }
            : null
        }
        onSelect={setSelected}
      />

      {/* Top bar: title + controls (controls wrap below title on mobile) */}
      <div className="pointer-events-none absolute inset-x-0 top-0 z-10 flex flex-wrap items-start justify-between gap-2 p-3 sm:p-4">
        <div className="pointer-events-auto">
          <ForecastHeader date={forecastDate(leadHours)} coord={fmtCoord(selected)} />
        </div>
        <div className="pointer-events-auto flex w-full flex-col gap-2 sm:w-auto sm:items-end">
          <LayerControls active={layer} onChange={setLayer} />
          <MapLegend meta={LAYERS[layer]} />
        </div>
      </div>

      {/* Bottom: compact control pills + scrollable metric cards */}
      <div className="absolute inset-x-0 bottom-0 z-10 p-3 sm:p-4">
        <div className="flex flex-col gap-2 sm:gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <LeadTimeSlider value={leadHours} onChange={setLeadHours} />
            <VerticalLevelSelector level={level} onChange={setLevel} />
          </div>
          <div className="flex snap-x snap-mandatory gap-2 overflow-x-auto pb-1 sm:gap-3 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
            <MetricCard label="tsurf" value={m.tsurf.toFixed(1)} unit="K" />
            <MetricCard label="ps" value={m.ps.toFixed(0)} unit="Pa" />
            <MetricCard
              label="temp"
              value={m.temp.toFixed(1)}
              unit="K"
              sub={levelLabel}
            />
            <MetricCard
              label="u/v"
              value={windSpeed.toFixed(1)}
              unit="m/s"
              sub={levelLabel}
            >
              <p className="mt-1 font-mono text-[11px] tabular-nums text-muted-foreground">
                {windDir.toFixed(0)}° · U {m.u >= 0 ? "+" : ""}
                {m.u.toFixed(1)} · V {m.v >= 0 ? "+" : ""}
                {m.v.toFixed(1)}
              </p>
            </MetricCard>
            <MetricCard label="dustcol" value={m.dustcol.toFixed(2)} unit="opacity" />
            <MetricCard label="co2ice" value={m.ice.toFixed(0)} unit="kg/m²" />
          </div>
        </div>
      </div>
    </main>
  )
}
