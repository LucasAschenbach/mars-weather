"use client"

import { Clock } from "lucide-react"
import { formatLead, LEAD_STEP_HOURS, MAX_LEAD_HOURS } from "@/lib/mars-data"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"

interface LeadTimeSliderProps {
  value: number
  onChange: (hours: number) => void
}

export function LeadTimeSlider({ value, onChange }: LeadTimeSliderProps) {
  const pct = (value / MAX_LEAD_HOURS) * 100

  return (
    <Popover>
      <PopoverTrigger className="flex items-center gap-2 rounded-lg border border-border/60 bg-card/55 px-3 py-2 backdrop-blur-md transition-colors hover:bg-card/75">
        <Clock className="size-3.5 text-mars" aria-hidden="true" />
        <span className="text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
          Lead Time
        </span>
        <span className="font-mono text-xs font-semibold tabular-nums text-mars">
          {formatLead(value)}
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
            Lead Time
          </span>
          <span className="font-mono text-sm font-semibold tabular-nums text-mars">
            {formatLead(value)}
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={MAX_LEAD_HOURS}
          step={LEAD_STEP_HOURS}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          aria-label="Forecast lead time in hours"
          className="mt-3 h-1.5 w-full cursor-pointer appearance-none rounded-full outline-none [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:cursor-pointer [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:bg-mars [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-mars [&::-webkit-slider-thumb]:shadow-[0_0_8px_1px] [&::-webkit-slider-thumb]:shadow-mars/60"
          style={{
            background: `linear-gradient(to right, var(--mars) ${pct}%, color-mix(in oklch, var(--foreground) 18%, transparent) ${pct}%)`,
          }}
        />
        <div className="mt-1.5 flex justify-between font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
          <span>Now</span>
          <span>+7 Days</span>
        </div>
      </PopoverContent>
    </Popover>
  )
}
