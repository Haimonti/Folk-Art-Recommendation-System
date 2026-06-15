const ROMAN: Record<string, string> = { "1": "I", "2": "II" };

export function panelName(id: string): string {
  const parts = String(id).split("_");
  if (parts.length !== 3) return `Panel ${id}`;
  const [survey, scroll, panel] = parts;
  const sv = ROMAN[survey] || survey;
  return `Scroll ${sv}-${scroll} Panel ${panel}`;
}
