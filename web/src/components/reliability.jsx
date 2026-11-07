// Two small charts for the About page, drawn from abstain.json (held-out test split):
// how often a label was right at each score level, and how accuracy rises as the app
// labels fewer comments and marks the rest "check by hand".
import abstain from '../abstain.json';

const W = 320;
const H = 200;
const PAD = { l: 40, r: 18, t: 22, b: 34 };
const iw = W - PAD.l - PAD.r;
const ih = H - PAD.t - PAD.b;
const pctText = (v) => `${Math.round(v * 100)}%`;

function Axes({ xTicks, yTicks, x, y, xLabel, yLabel }) {
  return (
    <g className="text-ink-faint dark:text-ube-300">
      {yTicks.map((t) => (
        <g key={`y${t}`}>
          <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} className="stroke-current" strokeOpacity="0.18" />
          <text x={PAD.l - 6} y={y(t) + 3.5} textAnchor="end" className="fill-current text-[10px]">{pctText(t)}</text>
        </g>
      ))}
      {xTicks.map((t) => (
        <text key={`x${t}`} x={x(t)} y={H - PAD.b + 14} textAnchor="middle" className="fill-current text-[10px]">{pctText(t)}</text>
      ))}
      <text x={PAD.l + iw / 2} y={H - 4} textAnchor="middle" className="fill-current text-[10.5px]">{xLabel}</text>
      <text x={PAD.l - 6} y={10} textAnchor="start" className="fill-current text-[10.5px]">{yLabel}</text>
    </g>
  );
}

/** Bars: accuracy within each top-score band; the dashed line is where score would equal accuracy. */
export function ReliabilityChart() {
  const bins = abstain.reliability.filter((b) => b.count >= 10);
  const x = (v) => PAD.l + ((v - 1 / 3) / (2 / 3)) * iw;
  const y = (v) => PAD.t + (1 - v) * ih;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img"
      aria-label="Accuracy against top score on held-out test comments: higher scores were right more often, close to the diagonal">
      <Axes xTicks={[0.4, 0.6, 0.8, 1]} yTicks={[0, 0.5, 1]} x={x} y={y} xLabel="Top score" yLabel="Share of labels that were right" />
      <line x1={x(1 / 3)} y1={y(1 / 3)} x2={x(1)} y2={y(1)} className="stroke-ink-faint dark:stroke-ube-300" strokeDasharray="4 4" />
      {bins.map((b) => (
        <g key={b.from}>
          <rect x={x(b.from) + 1.5} width={Math.max(1, x(b.to) - x(b.from) - 3)} y={y(b.accuracy)} height={y(0) - y(b.accuracy)}
            rx="3" className="fill-ube-500/70 dark:fill-ube-300/70" />
          <title>{`Top score ${pctText(b.from)}–${pctText(b.to)}: ${pctText(b.accuracy)} right (${b.count} comments)`}</title>
        </g>
      ))}
    </svg>
  );
}

/** Line: accuracy of the labelled comments when only the most confident share is labelled. */
export function CoverageChart() {
  const pts = abstain.curve;
  const x = (v) => PAD.l + ((1 - v) / 0.5) * iw; // 100% labelled on the left, 50% on the right
  const lo = 0.75;
  const y = (v) => PAD.t + (1 - (v - lo) / (1 - lo)) * ih;
  const path = pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.coverage).toFixed(1)},${y(p.answered_accuracy).toFixed(1)}`).join(' ');
  const t = abstain.test;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img"
      aria-label={`Accuracy rises from ${pctText(pts[0].answered_accuracy)} with every comment labelled to ${pctText(t.answered_accuracy)} when ${pctText(t.coverage)} are labelled`}>
      <Axes xTicks={[1, 0.9, 0.8, 0.7, 0.6, 0.5]} yTicks={[0.8, 0.9, 1]} x={x} y={y} xLabel="Share of comments labelled" yLabel="Share of labels that were right" />
      <path d={path} fill="none" className="stroke-jambu-500" strokeWidth="2.5" strokeLinejoin="round" />
      <circle cx={x(t.coverage)} cy={y(t.answered_accuracy)} r="5" className="fill-jambu-500 stroke-white dark:stroke-night" strokeWidth="2" />
      <text x={x(t.coverage) - 8} y={y(t.answered_accuracy) - 10} textAnchor="end" className="fill-ink text-[10.5px] font-semibold dark:fill-ube-50">
        This app&apos;s cut-off
      </text>
    </svg>
  );
}
