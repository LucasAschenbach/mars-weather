"use client"

import { Database } from "lucide-react"
import type { LoadedForecastSource } from "@/lib/forecast-file"

interface ForecastSourcePickerProps {
  sources: LoadedForecastSource[]
  activeId: string | null
  onChange: (id: string) => void
}

export function ForecastSourcePicker({
  sources,
  activeId,
  onChange,
}: ForecastSourcePickerProps) {
  if (sources.length <= 1) return null

  return (
    <label className="flex items-center gap-2 rounded-lg border border-border/60 bg-card/55 px-3 py-2 backdrop-blur-md transition-colors hover:bg-card/75">
      <Database className="size-3.5 text-mars" aria-hidden="true" />
      <span className="text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
        Forecast
      </span>
      <select
        value={activeId ?? sources[0]?.id ?? ""}
        onChange={(event) => onChange(event.target.value)}
        className="max-w-48 bg-transparent font-mono text-xs font-semibold text-mars outline-none"
        aria-label="Forecast source"
      >
        {sources.map((source) => (
          <option key={source.id} value={source.id} className="bg-background text-foreground">
            {source.label}
          </option>
        ))}
      </select>
    </label>
  )
}
