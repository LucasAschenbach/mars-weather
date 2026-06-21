"use client"

import { Layers3 } from "lucide-react"
import { ATMOS_LEVEL_COUNT } from "@/lib/mars-data"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"

interface VerticalLevelSelectorProps {
  level: number
  onChange: (level: number) => void
}

export function VerticalLevelSelector({ level, onChange }: VerticalLevelSelectorProps) {
  const maxLevel = ATMOS_LEVEL_COUNT - 1
  const pct = (level / maxLevel) * 100
  const label = `L${level + 1}/${ATMOS_LEVEL_COUNT}`

  return (
    <Popover>
      <PopoverTrigger className="flex items-center gap-2 rounded-lg border border-border/60 bg-card/55 px-3 py-2 backdrop-blur-md transition-colors hover:bg-card/75">
        <Layers3 className="size-3.5 text-foreground" aria-hidden="true" />
        <span className="text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
          Level
        </span>
        <span className="font-mono text-xs font-semibold tabular-nums text-foreground">
          {label}
        </span>
      </PopoverTrigger>
      <PopoverContent
        side="top"
        align="start"
        sideOffset={8}
        className="w-64 border-border/60 bg-card/90 backdrop-blur-md"
      >
        <div className="flex items-baseline justify-between gap-2">
          <span className="text-[11px] font-medium uppercase tracking-widest text-muted-foreground">
            Model Level
          </span>
          <span className="font-mono text-sm font-semibold tabular-nums text-foreground">
            {label}
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={maxLevel}
          step={1}
          value={level}
          onChange={(e) => onChange(Number(e.target.value))}
          aria-label="OpenMARS vertical level for temp, u, and v"
          className="mt-3 h-1.5 w-full cursor-pointer appearance-none rounded-full outline-none [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:cursor-pointer [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:bg-foreground [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-foreground"
          style={{
            background: `linear-gradient(to right, color-mix(in oklch, var(--foreground) 70%, transparent) ${pct}%, color-mix(in oklch, var(--foreground) 18%, transparent) ${pct}%)`,
          }}
        />
        <div className="mt-1.5 flex justify-between font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
          <span>L1</span>
          <span>L{ATMOS_LEVEL_COUNT}</span>
        </div>
      </PopoverContent>
    </Popover>
  )
}
