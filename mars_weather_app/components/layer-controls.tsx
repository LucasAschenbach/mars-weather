"use client"

import { LAYERS, sampleColor, type Layer, type LayerMeta } from "@/lib/mars-data"

const ORDER: Layer[] = ["tsurf", "ps", "dustcol"]
const FULL_LABELS: Record<Layer, string> = {
  tsurf: "Surface Temp",
  ps: "Pressure",
  dustcol: "Dust Opacity",
}

interface LayerControlsProps {
  active: Layer
  onChange: (layer: Layer) => void
}

export function LayerControls({ active, onChange }: LayerControlsProps) {
  return (
    <div className="flex w-full flex-col gap-2 rounded-lg border border-border/60 bg-card/55 p-1.5 backdrop-blur-md sm:w-auto sm:p-2">
      <span className="px-1 text-[10px] font-medium uppercase tracking-widest text-muted-foreground sr-only sm:not-sr-only">
        Map Layer
      </span>
      <div className="flex gap-1">
        {ORDER.map((key) => {
          const isActive = key === active
          return (
            <button
              key={key}
              type="button"
              onClick={() => onChange(key)}
              aria-pressed={isActive}
              className={`flex-1 whitespace-nowrap rounded-md px-2.5 py-1 text-[11px] font-medium uppercase tracking-wide transition-colors sm:flex-initial sm:px-3 sm:py-1.5 sm:text-xs ${
                isActive
                  ? "bg-mars text-mars-foreground"
                  : "text-muted-foreground hover:bg-foreground/10 hover:text-foreground"
              }`}
            >
              <span className="sm:hidden">{LAYERS[key].label}</span>
              <span className="hidden sm:inline">{FULL_LABELS[key]}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

function gradientCss(meta: LayerMeta): string {
  const stops = meta.stops
    .map(([p, c]) => `rgb(${c[0]} ${c[1]} ${c[2]}) ${Math.round(p * 100)}%`)
    .join(", ")
  return `linear-gradient(to right, ${stops})`
}

interface MapLegendProps {
  meta: LayerMeta
}

export function MapLegend({ meta }: MapLegendProps) {
  const [lo, hi] = meta.domain
  // Use sampleColor so the legend swatch matches the painted overlay exactly.
  const mid = sampleColor(meta, 0.5)
  return (
    <div className="flex items-center gap-2 rounded-lg border border-border/60 bg-card/55 px-2.5 py-1.5 backdrop-blur-md sm:flex-col sm:items-stretch sm:gap-0 sm:px-3 sm:py-3">
      <span className="shrink-0 text-[10px] font-medium uppercase tracking-widest text-muted-foreground sm:hidden">
        {meta.label}
      </span>
      <div className="hidden items-center justify-between gap-3 sm:flex">
        <span className="text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
          {FULL_LABELS[meta.key]}
        </span>
        <span
          className="font-mono text-[10px] text-muted-foreground"
          style={{ color: `rgb(${mid[0]} ${mid[1]} ${mid[2]})` }}
        >
          {meta.unit}
        </span>
      </div>
      <span className="shrink-0 font-mono text-[10px] tabular-nums text-muted-foreground sm:hidden">
        {lo}
      </span>
      <div
        className="h-2.5 w-20 shrink-0 rounded-full sm:mt-2 sm:w-44"
        style={{ backgroundImage: gradientCss(meta) }}
        aria-hidden="true"
      />
      <span className="shrink-0 font-mono text-[10px] tabular-nums text-muted-foreground sm:hidden">
        {hi}
      </span>
      <div className="mt-1 hidden justify-between font-mono text-[10px] tabular-nums text-muted-foreground sm:flex">
        <span>{lo}</span>
        <span>{hi}</span>
      </div>
    </div>
  )
}
