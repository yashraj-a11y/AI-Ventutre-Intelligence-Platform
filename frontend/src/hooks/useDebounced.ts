// Day 19 · Phase F — debounce a rapidly-changing value (e.g. a slider) so we don't spam /predict.
import { useEffect, useState } from "react";

export function useDebounced<T>(value: T, ms = 450): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return debounced;
}
