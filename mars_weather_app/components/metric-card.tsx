import type { ReactNode } from "react"

interface MetricCardProps {
  label: string
  value: string
  unit?: string
  sub?: string
  children?: ReactNode
}

export function MetricCard({ label, value, unit, sub, children }: MetricCardProps) {
  return (
    <div className="min-w-[8.25rem] flex-1 snap-start rounded-lg border border-border/60 bg-card/55 px-3 py-2.5 backdrop-blur-md sm:min-w-44 sm:px-4 sm:py-3">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[11px] font-medium uppercase tracking-widest text-muted-foreground">
          {label}
        </span>
        {sub ? (
          <span className="text-[10px] uppercase tracking-wider text-muted-foreground/80">{sub}</span>
        ) : null}
      </div>
      <div className="mt-1.5 flex items-baseline gap-1">
        <span className="font-mono text-xl font-semibold tabular-nums text-card-foreground sm:text-2xl">
          {value}
        </span>
        {unit ? <span className="font-mono text-sm text-muted-foreground">{unit}</span> : null}
      </div>
      {children}
    </div>
  )
}
