// Readable names for the model's features. The API reports features by their column names
// (usd_goal_real, main_category_Technology, title term 'smart'); the UI shows plain English and keeps
// the raw name available (tooltips), so nothing is hidden — only translated.

const NAMED: Record<string, string> = {
  usd_goal_real: "Funding goal",
  goal: "Funding goal",
  goal_per_day: "Goal per campaign day",
  campaign_duration: "Campaign length",
  category_freq: "Category popularity",
  launch_year: "Launch year",
  launch_month: "Launch month",
  launch_weekday: "Launch weekday",
  launch_quarter: "Launch quarter",
  is_weekend_launch: "Weekend launch",
  title_length: "Title length",
  word_count: "Words in title",
  avg_word_length: "Average word length",
  uppercase_ratio: "Capital letters in title",
  has_number: "Number in title",
  title: "Title",
};

const PREFIXES: [string, string][] = [
  ["main_category_", "Category"],
  ["currency_", "Currency"],
  ["country_", "Country"],
];

/** "usd_goal_real" -> "Funding goal"; "title term 'smart'" -> "Title word “smart”". */
export function featureLabel(raw: string): string {
  if (NAMED[raw]) return NAMED[raw];
  for (const [p, label] of PREFIXES) if (raw.startsWith(p)) return `${label}: ${raw.slice(p.length)}`;
  const term = raw.match(/^title term '(.+)'( \(absent\))?$/);
  if (term) return term[2] ? `No “${term[1]}” in title` : `Title word “${term[1]}”`;
  return raw.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}
